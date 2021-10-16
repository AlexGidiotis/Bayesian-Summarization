import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from datasets import load_dataset


logger = logging.getLogger(__name__)


def load_datasets(
        dataset_name=None,
        dataset_config_name=None,
        train_file=None,
        validation_file=None,
        test_file=None):
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
    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(dataset_name, dataset_config_name)
    else:
        data_files = {}
        extension = ""
        if train_file is not None:
            data_files["train"] = train_file
            extension = train_file.split(".")[-1]
        if validation_file is not None:
            data_files["validation"] = validation_file
            extension = validation_file.split(".")[-1]
        if test_file is not None:
            data_files["test"] = test_file
            extension = test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    return datasets


def init_loader(
        test_batch_size,
        split,
        data_path=None,
        dataset_name=None,
        dataset_config_name=None,
        max_test_samples=None):
    """Initialize test DataLoader"""
    datasets = {}
    if split == "train":
        datasets = load_datasets(dataset_name, dataset_config_name, train_file=data_path)
    elif split == "val":
        datasets = load_datasets(dataset_name, dataset_config_name, validation_file=data_path)
    elif split == "test":
        datasets = load_datasets(dataset_name, dataset_config_name, test_file=data_path)
    else:
        ValueError(f"split needs to be one of train/val/test but {split} was given")

    test_dataset = datasets[split]
    if max_test_samples is not None:
        test_dataset = test_dataset.select(range(max_test_samples))

    params = {
        'batch_size': test_batch_size,
        'shuffle': False,
    }
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    return test_loader


def init_dataset(data_path=None, dataset_name=None, dataset_config_name=None, split="train"):
    """Initialize test Dataset"""
    datasets = {}
    if split == "train":
        datasets = load_datasets(dataset_name, dataset_config_name, train_file=data_path)
    elif split == "val":
        datasets = load_datasets(dataset_name, dataset_config_name, validation_file=data_path)
    elif split == "test":
        datasets = load_datasets(dataset_name, dataset_config_name, test_file=data_path)
    else:
        ValueError(f"split needs to be one of train/val/test but {split} was given")

    return datasets[split]


def create_loader(dataset, batch_size, sample):
    """Takes a torch Dataset and a list of indexes and creates a DataLoader of it.

    The DataLoader will only use the subset of articles specified in sample list of indexes.
    """
    params = {
        'batch_size': batch_size,
        'shuffle': False,
        'sampler': sample,
    }
    data_loader = torch.utils.data.DataLoader(dataset, **params)
    return data_loader


def load_model(device, model_path, tokenizer_name=None):
    """Load model and tokenizer"""
    logger.info(f"Loading tokenizer {tokenizer_name if tokenizer_name else model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_path)

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path).to(device)

    return model, tokenizer
