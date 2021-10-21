import os.path
import tempfile
import unittest
import mock
from datasets import DatasetDict, Dataset, Metric

from transformers import AutoTokenizer, PegasusForConditionalGeneration, PegasusConfig, PreTrainedTokenizerFast, \
    DataCollatorForSeq2Seq

from src.common.loaders import load_datasets
from src.summarization.sum_base import Summarizer, postprocess_text
from test.test_loaders import create_test_loader, write_test_json
from test.testing_common_utils import values_tensor
from src.bayesian_summarization.bayesian import BayesianSummarizer


class TestSummarizer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @mock.patch("src.summarization.sum_base.load_model")
    def test_init_sum(self, mock_model_loader):
        args = {
            "model_path": "model_path",
            "train_file": "training_file.json",
            "output_dir": "output_dir",
            "source_len": 20,
            "target_len": 10,
            "metric": "rouge1",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, args["model_path"])
            train_file = os.path.join(temp_dir, args["train_file"])
            output_dir = os.path.join(temp_dir, args["output_dir"])

            write_test_json(train_file, n=4)

            summarizer = Summarizer(
                model_name_or_path=model_path,
                train_file=train_file,
                output_dir=output_dir,
                max_source_length=args["source_len"],
                max_target_length=args["target_len"],
                val_max_target_length=args["target_len"],
                evaluation_strategy="no",
                metric_for_best_model=args["metric"],
                pad_to_max_length=True,
                do_train=True,
                do_eval=False,
                predict_with_generate=True,
                do_predict=False,)

            dummy_config = PegasusConfig()
            model = PegasusForConditionalGeneration(config=dummy_config)
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
            mock_model_loader.return_value = model, tokenizer

            summarizer.init_sum()

        self.assertIsInstance(summarizer.model, PegasusForConditionalGeneration)
        self.assertIsInstance(summarizer.tokenizer, PreTrainedTokenizerFast)
        self.assertIsInstance(summarizer.datasets, DatasetDict)
        self.assertIsInstance(summarizer.train_dataset, Dataset)
        self.assertIsNone(summarizer.eval_dataset)
        self.assertIsNone(summarizer.eot_eval_dataset)
        self.assertIsNone(summarizer.test_dataset)
        self.assertIsInstance(summarizer.metric, Metric)

        del summarizer

    def test_load_init_datasets(self):
        args = {
            "model_path": "model_path",
            "train_file": "training_file.json",
            "validation_file": "val_file.json",
            "test_file": "test_file.json",
            "output_dir": "output_dir",
            "source_len": 20,
            "target_len": 10,
            "max_train_samples": 2,
            "max_val_samples": 2,
            "max_test_samples": 2,
            "metric": "rouge1",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, args["model_path"])
            train_file = os.path.join(temp_dir, args["train_file"])
            validation_file = os.path.join(temp_dir, args["validation_file"])
            test_file = os.path.join(temp_dir, args["test_file"])
            output_dir = os.path.join(temp_dir, args["output_dir"])

            write_test_json(train_file, n=4)
            write_test_json(validation_file, n=4)
            write_test_json(test_file, n=4)

            summarizer = Summarizer(
                model_name_or_path=model_path,
                train_file=train_file,
                validation_file=validation_file,
                test_file=test_file,
                output_dir=output_dir,
                max_source_length=args["source_len"],
                max_target_length=args["target_len"],
                val_max_target_length=args["target_len"],
                evaluation_strategy="epoch",
                metric_for_best_model=args["metric"],
                pad_to_max_length=True,
                max_train_samples=args["max_train_samples"],
                max_val_samples=args["max_val_samples"],
                max_test_samples=args["max_test_samples"],
                do_train=True,
                do_eval=True,
                predict_with_generate=True,
                do_predict=True,)

            datasets = load_datasets(
                dataset_name=summarizer.data_args.dataset_name,
                dataset_config_name=summarizer.data_args.dataset_config_name,
                train_file=summarizer.data_args.train_file,
                validation_file=summarizer.data_args.validation_file,
                test_file=summarizer.data_args.test_file)

            self.assertIsInstance(datasets, DatasetDict)

            dummy_config = PegasusConfig()
            model = PegasusForConditionalGeneration(config=dummy_config)
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
            prefix = ""

            train_dataset, eval_dataset, eot_eval_dataset, test_dataset = summarizer.init_datasets(
                datasets=datasets, model=model, tokenizer=tokenizer, prefix=prefix
            )
            self.assertIsInstance(train_dataset, Dataset)
            self.assertEqual(train_dataset.num_rows, 2)
            self.assertIsInstance(eval_dataset, Dataset)
            self.assertEqual(eval_dataset.num_rows, 2)
            self.assertIsInstance(eot_eval_dataset, Dataset)
            self.assertEqual(eot_eval_dataset.num_rows, 2)
            self.assertIsInstance(test_dataset, Dataset)
            self.assertEqual(test_dataset.num_rows, 2)

        del summarizer, model, tokenizer, prefix, datasets, train_dataset, eval_dataset, eot_eval_dataset, test_dataset

    def test_load_init_datasets_with_columns(self):
        args = {
            "model_path": "model_path",
            "train_file": "training_file.json",
            "validation_file": "val_file.json",
            "test_file": "test_file.json",
            "output_dir": "output_dir",
            "text_column": "document",
            "summary_column": "summary",
            "source_len": 20,
            "target_len": 10,
            "metric": "rouge1",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, args["model_path"])
            train_file = os.path.join(temp_dir, args["train_file"])
            validation_file = os.path.join(temp_dir, args["validation_file"])
            test_file = os.path.join(temp_dir, args["test_file"])
            output_dir = os.path.join(temp_dir, args["output_dir"])

            write_test_json(train_file, n=4)
            write_test_json(validation_file, n=4)
            write_test_json(test_file, n=4)

            summarizer = Summarizer(
                model_name_or_path=model_path,
                train_file=train_file,
                validation_file=validation_file,
                test_file=test_file,
                output_dir=output_dir,
                max_source_length=args["source_len"],
                max_target_length=args["target_len"],
                val_max_target_length=args["target_len"],
                evaluation_strategy="epoch",
                metric_for_best_model=args["metric"],
                text_column=args["text_column"],
                summary_column=args["summary_column"],
                pad_to_max_length=True,
                do_train=True,
                do_eval=True,
                predict_with_generate=True,
                do_predict=True,)

            datasets = load_datasets(
                dataset_name=summarizer.data_args.dataset_name,
                dataset_config_name=summarizer.data_args.dataset_config_name,
                train_file=summarizer.data_args.train_file,
                validation_file=summarizer.data_args.validation_file,
                test_file=summarizer.data_args.test_file)

            self.assertIsInstance(datasets, DatasetDict)

            dummy_config = PegasusConfig()
            model = PegasusForConditionalGeneration(config=dummy_config)
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
            prefix = ""

            train_dataset, eval_dataset, eot_eval_dataset, test_dataset = summarizer.init_datasets(
                datasets=datasets, model=model, tokenizer=tokenizer, prefix=prefix
            )

            self.assertIsInstance(train_dataset, Dataset)
            self.assertEqual(train_dataset.num_rows, 4)
            self.assertIsInstance(eval_dataset, Dataset)
            self.assertEqual(eval_dataset.num_rows, 4)
            self.assertIsInstance(eot_eval_dataset, Dataset)
            self.assertEqual(eot_eval_dataset.num_rows, 4)
            self.assertIsInstance(test_dataset, Dataset)
            self.assertEqual(test_dataset.num_rows, 4)

        del summarizer, model, tokenizer, prefix, datasets, train_dataset, eval_dataset, eot_eval_dataset, test_dataset

    def test_init_collocator_no_pad(self):
        args = {
            "model_path": "model_path",
            "train_file": "training_file.json",
            "validation_file": "val_file.json",
            "test_file": "test_file.json",
            "output_dir": "output_dir",
            "source_len": 20,
            "target_len": 10,
            "metric": "rouge1",
        }

        summarizer = Summarizer(
            model_name_or_path=args["model_path"],
            train_file=args["train_file"],
            validation_file=args["validation_file"],
            test_file=args["test_file"],
            output_dir=args["output_dir"],
            max_source_length=args["source_len"],
            max_target_length=args["target_len"],
            val_max_target_length=args["target_len"],
            evaluation_strategy="epoch",
            metric_for_best_model=args["metric"],
            pad_to_max_length=False,
            do_train=True,
            do_eval=True,
            predict_with_generate=True,
            do_predict=True,)

        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

        data_collator = summarizer.init_collocator(tokenizer=tokenizer)
        self.assertIsInstance(data_collator, DataCollatorForSeq2Seq)

        del summarizer, tokenizer, data_collator

    def test_init_collocator(self):
        args = {
            "model_path": "model_path",
            "train_file": "training_file.json",
            "validation_file": "val_file.json",
            "test_file": "test_file.json",
            "output_dir": "output_dir",
            "source_len": 20,
            "target_len": 10,
            "metric": "rouge1",
        }

        summarizer = Summarizer(
            model_name_or_path=args["model_path"],
            train_file=args["train_file"],
            validation_file=args["validation_file"],
            test_file=args["test_file"],
            output_dir=args["output_dir"],
            max_source_length=args["source_len"],
            max_target_length=args["target_len"],
            val_max_target_length=args["target_len"],
            evaluation_strategy="epoch",
            metric_for_best_model=args["metric"],
            pad_to_max_length=True,
            do_train=True,
            do_eval=True,
            predict_with_generate=True,
            do_predict=True,)

        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

        data_collator = summarizer.init_collocator(tokenizer=tokenizer)
        self.assertTrue(callable(data_collator))

        del summarizer, tokenizer, data_collator

    def test_init_decoder(self):
        args = {
            "model_path": "model_path",
            "train_file": "training_file.json",
            "validation_file": "val_file.json",
            "test_file": "test_file.json",
            "output_dir": "output_dir",
            "source_len": 20,
            "target_len": 10,
            "metric": "rouge1",
        }

        summarizer = Summarizer(
            model_name_or_path=args["model_path"],
            train_file=args["train_file"],
            validation_file=args["validation_file"],
            test_file=args["test_file"],
            output_dir=args["output_dir"],
            max_source_length=args["source_len"],
            max_target_length=args["target_len"],
            val_max_target_length=args["target_len"],
            evaluation_strategy="epoch",
            metric_for_best_model=args["metric"],
            pad_to_max_length=True,
            do_train=True,
            do_eval=True,
            predict_with_generate=True,
            do_predict=True,)

        dummy_config = PegasusConfig()
        model = PegasusForConditionalGeneration(config=dummy_config)
        prefix = summarizer.init_decoder(model)

        self.assertEqual(prefix, "")

        del summarizer, model, prefix

    def test_text_postprocessing(self):
        test_texts = ["This is the first sentence. This is the second sentence.    This is the third sentence.     "]
        processed_texts = postprocess_text(test_texts)
        self.assertNotEqual(test_texts[0], processed_texts[0])
