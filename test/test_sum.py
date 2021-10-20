import os.path
import tempfile
import unittest
import mock
from datasets import DatasetDict, Dataset, Metric

from transformers import AutoTokenizer, PegasusForConditionalGeneration, PegasusConfig, PreTrainedTokenizerFast

from src.summarization.sum_base import Summarizer
from test.test_loaders import create_test_loader, write_test_json
from test.testing_common_utils import values_tensor
from src.bayesian_summarization.bayesian import BayesianSummarizer


class TestSummarizer(unittest.TestCase):
    @mock.patch("src.summarization.sum_base.load_model")
    def setUp(self, mock_model_loader) -> None:
        self.args = {
            "model_path": "model_path",
            "train_file": "training_file.json",
            "validation_file": "val_file.json",
            "test_file": "test_file.json",
            "output_dir": "output_dir",
            "source_len": 20,
            "target_len": 10,
            "metric": "rouge1",
        }

        self.temp_dir = tempfile.TemporaryDirectory()
        model_path = os.path.join(self.temp_dir.name, self.args["model_path"])
        train_file = os.path.join(self.temp_dir.name, self.args["train_file"])
        validation_file = os.path.join(self.temp_dir.name, self.args["validation_file"])
        test_file = os.path.join(self.temp_dir.name, self.args["test_file"])
        output_dir = os.path.join(self.temp_dir.name, self.args["output_dir"])

        write_test_json(train_file, n=4)
        write_test_json(validation_file, n=4)
        write_test_json(test_file, n=4)

        self.summarizer = Summarizer(
            model_name_or_path=model_path,
            train_file=train_file,
            validation_file=validation_file,
            test_file=test_file,
            output_dir=output_dir,
            max_source_length=self.args["source_len"],
            max_target_length=self.args["target_len"],
            val_max_target_length=self.args["target_len"],
            evaluation_strategy="epoch",
            metric_for_best_model=self.args["metric"],
            pad_to_max_length=True,
            do_train=True,
            do_eval=True,
            predict_with_generate=True,
            do_predict=True,)

        dummy_config = PegasusConfig()
        model = PegasusForConditionalGeneration(config=dummy_config)
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        mock_model_loader.return_value = model, tokenizer

        self.summarizer.init_sum()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        del self.summarizer

    def test_init_sum(self):
        self.assertIsInstance(self.summarizer.model, PegasusForConditionalGeneration)
        self.assertIsInstance(self.summarizer.tokenizer, PreTrainedTokenizerFast)
        self.assertIsInstance(self.summarizer.datasets, DatasetDict)
        self.assertIsInstance(self.summarizer.train_dataset, Dataset)
        self.assertIsInstance(self.summarizer.eval_dataset, Dataset)
        self.assertIsInstance(self.summarizer.eot_eval_dataset, Dataset)
        self.assertIsInstance(self.summarizer.test_dataset, Dataset)
        self.assertIsInstance(self.summarizer.metric, Metric)
