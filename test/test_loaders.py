import json
import os
import tempfile
import unittest
import mock
from transformers import PegasusConfig, PegasusForConditionalGeneration, PreTrainedTokenizerFast

from src.common.loaders import init_loader, init_dataset, create_loader, load_datasets, load_model


def write_test_json(datafile, n):
    with open(datafile, "w") as jwf:
        for i in range(n):
            json.dump({
                "document": f"Test document {i}",
                "summary": f"Test summary {i}",
                "id": i
            }, fp=jwf)


def create_test_loader(args, tmp_dir):
    datafile = os.path.join(tmp_dir, "test.json")
    args["data_path"] = datafile
    write_test_json(datafile, n=args["max_test_samples"])
    test_loader = init_loader(
        test_batch_size=args["test_batch_size"],
        split="test", data_path=args["data_path"],
        dataset_name=args["dataset_name"],
        dataset_config_name=args["dataset_config_name"],
        max_test_samples=args["max_test_samples"])
    return test_loader


class TestDataLoader(unittest.TestCase):
    def test_load_datasets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datafile = os.path.join(tmp_dir, "test.json")
            args = {
                "test_file": datafile,
                "dataset_name": None,
                "dataset_config_name": "",
            }
            write_test_json(datafile, n=4)

            datasets = load_datasets(
                dataset_name=args["dataset_name"],
                dataset_config_name=args["dataset_config_name"],
                test_file=args["test_file"])

            self.assertIn("test", datasets)
            self.assertEqual(len(datasets["test"]), 4)
            del datasets

    def test_init_loader(self):
        args = {
            "test_batch_size": 1,
            "dataset_name": None,
            "dataset_config_name": "",
            "max_test_samples": 4,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_loader = create_test_loader(args=args, tmp_dir=tmp_dir)
            self.assertEqual(len(test_loader), 4)
            del test_loader

    def test_init_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datafile = os.path.join(tmp_dir, "test.json")
            args = {
                "data_path": datafile,
                "dataset_name": None,
                "dataset_config_name": "",
            }
            write_test_json(datafile, n=4)

            test_ds = init_dataset(
                data_path=args["data_path"],
                dataset_name=args["dataset_name"],
                dataset_config_name=args["dataset_config_name"],
                split="test")

            self.assertEqual(len(test_ds), 4)
            del test_ds

    def test_create_loader(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datafile = os.path.join(tmp_dir, "test.json")
            args = {
                "test_batch_size": 1,
                "data_path": datafile,
                "dataset_name": None,
                "dataset_config_name": "",
                "max_test_samples": 4,
            }
            write_test_json(datafile, n=4)

            test_ds = init_dataset(
                data_path=args["data_path"],
                dataset_name=args["dataset_name"],
                dataset_config_name=args["dataset_config_name"],
                split="test")

            test_loader = create_loader(
                dataset=test_ds,
                batch_size=args["test_batch_size"],
                sample=[0, 2])

            self.assertEqual(len(test_loader), 2)
            del test_loader, test_ds


class TestModelLoader(unittest.TestCase):
    @mock.patch("transformers.AutoConfig.from_pretrained")
    @mock.patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    def test_load_model(self, mock_model, mock_config):
        args = {
            "model_path": "test_pegasus",
            "tokenizer_name": "google/pegasus-xsum",
        }

        dummy_config = PegasusConfig()
        mock_config.return_value = dummy_config
        mock_model.return_value = PegasusForConditionalGeneration(config=dummy_config)
        model, tokenizer = load_model(model_name_or_path=args["model_path"], tokenizer_name=args["tokenizer_name"])

        self.assertIsInstance(model, PegasusForConditionalGeneration)
        self.assertIsInstance(tokenizer, PreTrainedTokenizerFast)

        del model, dummy_config, tokenizer
