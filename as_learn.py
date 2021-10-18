import argparse
import logging
import os
import sys
import random
import time
import re

import torch

from src.active_summarization import active_sum
from src.common.loaders import init_dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],)

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


set_global_logging_level(logging.ERROR, ["transformers"])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--train_file", type=str, help="")
    parser.add_argument("--validation_file", type=str, help="")
    parser.add_argument("--dataset_name", type=str, help="")
    parser.add_argument("--dataset_config_name", type=str, help="")
    parser.add_argument("--text_column", type=str, help="")
    parser.add_argument("--summary_column", type=str, help="")
    parser.add_argument("--seed", type=int, default=10, help="")
    parser.add_argument("--resume", type=int, default=0, help="")

    # AS args
    parser.add_argument("--N", type=int, default=10, help="")
    parser.add_argument("--L", type=int, default=20, help="")
    parser.add_argument("--K", type=int, default=100, help="")
    parser.add_argument("--S", type=int, default=10, help="")
    parser.add_argument("--steps", type=int, default=10, help="")
    parser.add_argument("--acquisition", type=str, default="bayesian", choices=["bayesian", "random"], help="")

    # Training args
    parser.add_argument("--init_model", type=str, help="")
    parser.add_argument("--max_source_length", type=int, default=256, help="")
    parser.add_argument("--max_summary_length", type=int, default=62, help="")
    parser.add_argument("--max_val_samples", type=int, help="")
    parser.add_argument("--max_test_samples", type=int, help="")
    parser.add_argument("--batch_size", type=int, default=8, help="")
    parser.add_argument("--num_beams", type=int, default=3, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="")
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument("--save_step", type=int, default=100, help="")
    parser.add_argument("--save_limit", type=int, default=1, help="")
    parser.add_argument("--metric_for_best_model", type=str, default="rouge1", choices=["rouge1", "rouge2", "rougeL"],
                        help="")

    args, unknown = parser.parse_known_args()

    return args, unknown


def main():
    # CUDA for PyTorch
    st = time.time()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    args, unknown = read_args()

    args.resume = bool(args.resume)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    data_path = os.path.join(args.output_path, "data")
    if args.resume:
        logger.info("Resuming session")
        if not os.path.exists(data_path):
            logger.info("Data dir not found. Unable to resume session.")
            sys.exit()
    else:
        if os.path.exists(data_path):
            logger.info("Data dir already exists. Aborting.")
            sys.exit()
        os.mkdir(data_path)

    train_model = os.path.join(args.output_path, "models")

    train_dataset = init_dataset(
        data_path=args.train_file,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split="train")
    train_sampler = active_sum.DataSampler(train_dataset, split="train")

    logger.info(f"{args.acquisition} L: {args.L} K: {args.K} S: {args.S} steps: {args.steps}")

    if args.acquisition == "bayesian":
        active_learner = active_sum.BAS(
            train_sampler,
            device=device,
            doc_col=args.text_column,
            sum_col=args.summary_column,
            seed=args.seed,
            py_module=__name__,
            init_model=args.init_model,
            source_len=args.max_source_length,
            target_len=args.max_summary_length,
            val_samples=args.max_val_samples,
            test_samples=args.max_test_samples,
            batch_size=args.batch_size,
            beams=args.num_beams,
            lr=args.learning_rate,
            save_step=args.save_step,
            save_limit=args.save_limit,
            metric=args.metric_for_best_model,
        )
    else:
        active_learner = active_sum.RandomActiveSum(
            train_sampler,
            device=device,
            doc_col=args.text_column,
            sum_col=args.summary_column,
            seed=args.seed,
            py_module=__name__,
            init_model=args.init_model,
            source_len=args.max_source_length,
            target_len=args.max_summary_length,
            val_samples=args.max_val_samples,
            test_samples=args.max_test_samples,
            batch_size=args.batch_size,
            beams=args.num_beams,
            lr=args.learning_rate,
            save_step=args.save_step,
            save_limit=args.save_limit,
            metric=args.metric_for_best_model,
        )

    if args.resume:
        active_learner.resume_learner(data_path)
    else:
        active_learner.init_learner(
            init_labeled=args.L,
            model_path=train_model,
            labeled_path=data_path,
            eval_path=args.validation_file,
            epochs=args.epochs)

    if args.acquisition == "bayesian":
        active_learner.learn(
            steps=args.steps,
            model_path=train_model,
            labeled_path=data_path,
            k=args.K, s=args.S, n=args.N,
            eval_path=args.validation_file,
            epochs=args.epochs)
    else:
        active_learner.learn(
            steps=args.steps,
            model_path=train_model,
            labeled_path=data_path,
            k=args.K, s=args.S,
            eval_path=args.validation_file,
            epochs=args.epochs)

    et = time.time()
    logger.info(f"Elapsed time: {et - st} sec.")


if __name__ == "__main__":
    main()