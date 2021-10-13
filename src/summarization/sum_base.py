import logging
import os
import re
import gc
import sys

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import torch
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import EarlyStoppingCallback

from src.summarization.utils import ModelArguments, DataTrainingArguments, SUMMARIZATION_NAME_MAPPING

with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


class Summarizer:
    """
    """
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.last_checkpoint = self.detect_checkpoint()

        self.setup_loggers()

        set_seed(self.training_args.seed)

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

    def init_sum(self):
        self.datasets = self.load_dataset()
        self.model, self.tokenizer = self.load_model()

        self.prefix = self.init_decoder()
        self.column_names, self.preprocess_function = self.init_preprocessing()
        self.train_dataset, self.eval_dataset, self.eot_eval_dataset, self.test_dataset = self.init_dataset()

        self.data_collator = self.init_collocator()

        self.metric = load_metric("rouge")

    def detect_checkpoint(self):
        """Detecting last checkpoint."""
        last_checkpoint = None
        if os.path.isdir(self.training_args.output_dir) and self.training_args.do_train and not self.training_args.overwrite_output_dir:
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
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(self.training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
        logger.info("Training/evaluation parameters %s", self.training_args)

    def load_dataset(self):
        """
        Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
        or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        (the dataset will be downloaded automatically from the datasets Hub).

        For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
        second column for the summaries (unless you specify column names for this with the `text_column` and
        `summary_column` arguments).
        For translation, only JSON files are supported, with one field named "translation" containing two keys for the
        source and target languages (unless you adapt what follows).

        In distributed training, the load_dataset function guarantee that only one local process can concurrently
        download the dataset.

        See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        https://huggingface.co/docs/datasets/loading_datasets.html.
        """
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name)
        else:
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
                extension = self.data_args.train_file.split(".")[-1]
            if self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
                extension = self.data_args.validation_file.split(".")[-1]
            if self.data_args.test_file is not None:
                data_files["test"] = self.data_args.test_file
                extension = self.data_args.test_file.split(".")[-1]
            datasets = load_dataset(extension, data_files=data_files)

        return datasets

    def load_model(self):
        """
        Load pretrained model and tokenizer

        Distributed training:
        The .from_pretrained methods guarantee that only one local process can concurrently
        download model & vocab.
        """
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        return model, tokenizer

    def init_decoder(self):
        """Set decoder_start_token_id"""
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            assert (
                    self.data_args.target_lang is not None and self.data_args.source_lang is not None
            ), "mBart requires --target_lang and --source_lang"
            if isinstance(self.tokenizer, MBartTokenizer):
                self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.data_args.target_lang]
            else:
                self.model.config.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.data_args.target_lang)

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        return prefix

    def init_dataset(self):
        # We need to tokenize inputs and targets.
        if self.training_args.do_train:
            column_names = self.datasets["train"].column_names
        elif self.training_args.do_eval:
            column_names = self.datasets["validation"].column_names
        elif self.training_args.do_predict:
            column_names = self.datasets["test"].column_names
        else:
            logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
            return

        # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
        # them all).
        source_lang, target_lang, text_column, summary_column = None, None, None, None

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
                    f"--summary_column' value '{self.data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = "max_length" if self.data_args.pad_to_max_length else False

        if self.training_args.label_smoothing_factor > 0 and not hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{self.model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples):
            inputs = examples[text_column]
            targets = examples[summary_column]
            inputs = [self.prefix + inp for inp in inputs]
            model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_dataset = None
        eval_dataset = None
        eot_eval_dataset = None
        test_dataset = None
        if self.training_args.do_train:
            train_dataset = self.datasets["train"]
            if "train" not in self.datasets:
                raise ValueError("--do_train requires a train dataset")
            if self.data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.data_args.max_train_samples))
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        if self.training_args.do_eval:
            max_target_length = self.data_args.val_max_target_length
            if "validation" not in self.datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = self.datasets["validation"]
            if self.data_args.max_val_samples is not None:
                eval_dataset = eval_dataset.select(range(self.data_args.max_val_samples))
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
            eot_eval_dataset = self.datasets["validation"]  # end-of-training evaluation with more data
            if self.data_args.max_val_samples is not None:
                eot_val_samples = 5000
                eot_eval_dataset = eot_eval_dataset.select(range(eot_val_samples))
            eot_eval_dataset = eot_eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        if self.training_args.do_predict:
            max_target_length = self.data_args.val_max_target_length
            if "test" not in self.datasets:
                raise ValueError("--do_predict requires a test dataset")
            test_dataset = self.datasets["test"]
            if self.data_args.max_test_samples is not None:
                test_dataset = test_dataset.select(range(self.data_args.max_test_samples))
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        return train_dataset, eval_dataset, eot_eval_dataset, test_dataset

    def init_collocator(self):
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
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
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    summarizer = Summarizer(model_args, data_args, training_args)
    summarizer.init_sum()

    # WIP

    es_callback = EarlyStoppingCallback(early_stopping_patience=3)
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[es_callback]
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    #         trainer.log_metrics("train", metrics)
    #         trainer.save_metrics("train", metrics)
    #         trainer.save_state()
    train_metrics = metrics

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            eval_dataset=eot_eval_dataset if training_args.do_eval else None,
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_val_samples = eot_val_samples if data_args.max_val_samples is not None else len(eot_eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eot_eval_dataset))

    #         trainer.log_metrics("eval", metrics)
    #         trainer.save_metrics("eval", metrics)
    eval_metrics = metrics
    print(eval_metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = test_results.metrics
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        #         trainer.log_metrics("test", metrics)
        #         trainer.save_metrics("test", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "test_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    test_metrics = metrics

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return train_metrics, eval_metrics, test_metrics


if __name__ == "__main__":
    main()