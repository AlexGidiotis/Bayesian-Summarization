import logging
import os
import sys

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_metric

import transformers
from filelock import FileLock
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import EarlyStoppingCallback

from src.common.loaders import load_datasets, load_model
from src.summarization.generation import generate_summaries
from src.summarization.utils import SUMMARIZATION_NAME_MAPPING, parse_kargs

with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Basic class for summary generation.

    Example:
    ```
    summarizer = Summarizer(...)
    test_loader = torch.utils.data.DataLoader(...)

    model, tokenizer = summarizer.load_model_tokenizer()
    generated_sums, target_sums, article_ids = summarizer.predict(test_loader, model, tokenizer)
    ```
    """
    def __init__(self, **kwargs):
        self.model_args, self.data_args, self.training_args = parse_kargs(**kwargs)
        self.setup_loggers()

        set_seed(self.training_args.seed)

    def load_model_tokenizer(self):
        model, tokenizer = load_model(
            model_name_or_path=self.model_args.model_name_or_path,
            config_name=self.model_args.config_name,
            cache_dir=self.model_args.cache_dir,
            model_revision=self.model_args.model_revision,
            use_auth_token=self.model_args.use_auth_token,
            tokenizer_name=self.model_args.tokenizer_name,
            use_fast_tokenizer=self.model_args.use_fast_tokenizer)

        return model, tokenizer

    def setup_loggers(self):
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.setLevel(logging.INFO if is_main_process(self.training_args.local_rank) else logging.WARN)

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, "
            f"device: {self.training_args.device}, "
            f"n_gpu: {self.training_args.n_gpu}, "
            f"distributed training: {bool(self.training_args.local_rank != -1)}, "
            f"16-bits training: {self.training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(self.training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
        logger.info("Training/evaluation parameters %s", self.training_args)

    def predict(self, dataloader, model, tokenizer, *args, **kwargs):
        generated_sums, target_sums, article_ids = generate_summaries(
            dataloader=dataloader,
            model=model,
            tokenizer=tokenizer,
            device=self.training_args.device,
            text_column=self.data_args.text_column,
            summary_column=self.data_args.summary_column,
            max_source_length=self.data_args.max_source_length,
            num_beams=self.data_args.num_beams)

        return generated_sums, target_sums, article_ids


class TrainerSummarizer(Summarizer):
    """
    Basic class that runs summarization training and prediction using datasets.

    Example:
    ```
    sum_trainer = TrainerSummarizer(...)

    sum_trainer.init_sum()

    train_metrics = sum_trainer.train()
    eval_metrics = sum_trainer.evaluate()
    test_metrics = sum_trainer.predict()
    ```
    """
    def __init__(self, **kwargs):
        self.last_checkpoint = None
        self.datasets = None
        self.model = None
        self.tokenizer = None
        self.prefix = None
        self.column_names = None
        self.preprocess_function = None
        self.train_dataset = None
        self.eval_dataset = None
        self.eot_eval_dataset = None
        self.test_dataset = None
        self.data_collator = None
        self.metric = None
        self.trainer = None
        super(TrainerSummarizer, self).__init__(**kwargs)

    def init_sum(self):
        self.last_checkpoint = self.detect_checkpoint()
        self.datasets = load_datasets(
            dataset_name=self.data_args.dataset_name,
            dataset_config_name=self.data_args.dataset_config_name,
            train_file=self.data_args.train_file,
            validation_file=self.data_args.validation_file,
            test_file=self.data_args.test_file)
        self.model, self.tokenizer = self.load_model_tokenizer()

        self.prefix = self.init_decoder(model=self.model)
        (
            self.train_dataset, self.eval_dataset, self.eot_eval_dataset, self.test_dataset
        ) = self.init_datasets(datasets=self.datasets, model=self.model, tokenizer=self.tokenizer, prefix=self.prefix)

        self.data_collator = self.init_collocator(tokenizer=self.tokenizer)

        self.metric = load_metric("rouge")

    def detect_checkpoint(self):
        """Detecting last checkpoint."""
        last_checkpoint = None
        if os.path.isdir(self.training_args.output_dir) and not self.training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(self.training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        return last_checkpoint

    def init_decoder(self, model):
        """Set decoder_start_token_id"""
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        return prefix

    def init_datasets(self, datasets, model, tokenizer, prefix):
        """
        Initializes loaded datasets for preprocessing, training and prediction.

        :returns
        train_dataset:
        eval_dataset:
        eot_eval_dataset:
        test_dataset:
        """
        # We need to tokenize inputs and targets.
        column_names = datasets["train"].column_names

        # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
        # them all).
        text_column, summary_column = None, None

        # Get the column names for input/target.
        dataset_columns = SUMMARIZATION_NAME_MAPPING.get(self.data_args.dataset_name, None)
        if self.data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = self.data_args.text_column
            if text_column not in column_names:
                raise ValueError(
                    f"--text_column' value '{self.data_args.text_column}' needs to be one of: {', '.join(column_names)}"
                )
        if self.data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = self.data_args.summary_column
            if summary_column not in column_names:
                raise ValueError(
                    f"--summary_column' value '{self.data_args.summary_column}' "
                    f"needs to be one of: {', '.join(column_names)}"
                )

        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = "max_length" if self.data_args.pad_to_max_length else False

        if self.training_args.label_smoothing_factor > 0 and not hasattr(
                model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. "
                "This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples):
            inputs = examples[text_column]
            targets = examples[summary_column]
            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(
                inputs,
                max_length=self.data_args.max_source_length,
                padding=padding,
                truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_target_length,
                    padding=padding,
                    truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        if self.training_args.do_train:
            if "train" not in datasets:
                raise ValueError("training requires a train dataset")
            train_dataset = datasets["train"]
            if self.data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.data_args.max_train_samples))
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
        else:
            train_dataset = None

        if self.training_args.do_eval:
            max_target_length = self.data_args.val_max_target_length
            if "validation" not in datasets:
                raise ValueError("evaluation requires a validation dataset")
            eval_dataset = datasets["validation"]
            if self.data_args.max_val_samples is not None:
                eval_dataset = eval_dataset.select(range(self.data_args.max_val_samples))
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
            eot_eval_dataset = datasets["validation"]  # end-of-training evaluation with more data
            if self.data_args.max_test_samples is not None:
                eot_eval_dataset = eot_eval_dataset.select(range(self.data_args.max_test_samples))
            eot_eval_dataset = eot_eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
        else:
            eval_dataset = None
            eot_eval_dataset = None

        if self.training_args.do_predict:
            max_target_length = self.data_args.val_max_target_length
            if "test" not in datasets:
                raise ValueError("prediction requires a test dataset")
            test_dataset = datasets["test"]
            if self.data_args.max_test_samples is not None:
                test_dataset = test_dataset.select(range(self.data_args.max_test_samples))
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
        else:
            test_dataset = None

        return train_dataset, eval_dataset, eot_eval_dataset, test_dataset

    def init_collocator(self, tokenizer):
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if self.training_args.fp16 else None,
            )

        return data_collator

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = postprocess_text(decoded_preds)
        decoded_labels = postprocess_text(decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def train(self):
        es_callback = EarlyStoppingCallback(early_stopping_patience=3)
        # Initialize our Trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[es_callback]
        )

        # Training
        if self.last_checkpoint is not None:
            checkpoint = self.last_checkpoint
        elif os.path.isdir(self.model_args.model_name_or_path):
            checkpoint = self.model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples
            if self.data_args.max_train_samples is not None else len(self.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        train_metrics = metrics

        return train_metrics

    def evaluate(self):
        if self.trainer is None:
            self.trainer = Seq2SeqTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[]
            )
        metrics = self.trainer.evaluate(
            eval_dataset=self.eot_eval_dataset,
            max_length=self.data_args.val_max_target_length,
            num_beams=self.data_args.num_beams,
            metric_key_prefix="eval"
        )
        max_eot_val_samples = (
            self.data_args.max_test_samples
            if self.data_args.max_test_samples is not None else len(self.eot_eval_dataset)
        )
        metrics["eval_samples"] = min(max_eot_val_samples, len(self.eot_eval_dataset))

        eval_metrics = metrics

        return eval_metrics

    def predict(self, *args, **kwargs):
        if self.trainer is None:
            self.trainer = Seq2SeqTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[]
            )
        test_results = self.trainer.predict(
            self.test_dataset,
            metric_key_prefix="test",
            max_length=self.data_args.val_max_target_length,
            num_beams=self.data_args.num_beams,
        )
        metrics = test_results.metrics
        max_test_samples = (
            self.data_args.max_test_samples if self.data_args.max_test_samples is not None else len(self.test_dataset)
        )
        metrics["test_samples"] = min(max_test_samples, len(self.test_dataset))

        if self.trainer.is_world_process_zero():
            if self.training_args.predict_with_generate:
                test_preds = self.tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                output_test_preds_file = os.path.join(self.training_args.output_dir, "test_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))
        test_metrics = metrics

        return test_metrics


def postprocess_text(texts):
    texts = [text.strip() for text in texts]

    # rougeLSum expects newline after each sentence
    texts = ["\n".join(nltk.sent_tokenize(text)) for text in texts]

    return texts
