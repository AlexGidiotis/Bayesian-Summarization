import json
import os
import tempfile
import unittest

from src.loaders import init_loader, init_dataset, create_loader


def write_test_json(datafile, n):
    with open(datafile, "w") as jwf:
        for i in range(n):
            json.dump({
                "document": f"Test document {i}",
                "summary": f"Test summary {i}",
                "id": i
            }, fp=jwf)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init_loader(self):
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

            test_loader = init_loader(
                test_batch_size=args["test_batch_size"],
                split="test", data_path=args["data_path"],
                dataset_name=args["dataset_name"],
                dataset_config_name=args["dataset_config_name"],
                max_test_samples=args["max_test_samples"])

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
            del test_loader
