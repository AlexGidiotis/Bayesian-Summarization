from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from datasets import load_dataset


def init_loader(test_batch_size, data_path=None, dataset_name=None, dataset_config_name=None, max_test_samples=None):
    """Initialize test DataLoader"""
    if dataset_name is not None:
        datasets = load_dataset(dataset_name, dataset_config_name)
    else:
        data_files = {"test": data_path}
        extension = data_path.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    test_dataset = datasets["test"]
    if max_test_samples is not None:
        test_dataset = test_dataset.select(range(max_test_samples))
    
    params = {
        'batch_size': test_batch_size,
        'shuffle': False,
    }
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)
    
    return test_loader


def load_model(device, model_path, tokenizer_name=None):
    """Load model and tokenizer"""
    print(f"Loading tokenizer {tokenizer_name if tokenizer_name else model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_path)

    print(f"Loading model from {model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path).to(device)

    return model, tokenizer


def init_dataset(args, split="train"):
    """Initialize test Dataset"""
    if args.dataset_name is not None:
        datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {split: args.data_path}
        extension = args.data_path.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    dataset = datasets[split]

    return dataset


def create_loader(data_sampler, batch_size, sample):
    """Takes a torch Dataset and a list of indexes and creates a DataLoader of it.

    The DataLoader will only use the subset of articles specified in sample list of indexes.
    """
    dataset = data_sampler.dataset
    params = {
        'batch_size': batch_size,
        'shuffle': False,
        'sampler': sample,
    }
    data_loader = torch.utils.data.DataLoader(dataset, **params)
    return data_loader
