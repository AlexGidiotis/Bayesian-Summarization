import argparse
import os
import random

import torch

from src.summarization.generation import generate_summaries
from src.common.loaders import init_loader, load_model
from src.common.scoring import score_standard


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--dataset_name", type=str, help="")
    parser.add_argument("--dataset_config_name", type=str, help="")
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--text_column", type=str, help="")
    parser.add_argument("--summary_column", type=str, help="")

    parser.add_argument("--tokenizer_name", type=str, help="")
    parser.add_argument("--max_source_length", type=int, default=256, help="")
    parser.add_argument("--max_summary_length", type=int, default=128, help="")
    parser.add_argument("--max_test_samples", type=int, help="")
    parser.add_argument("--seed", type=int, default=10, help="")
    parser.add_argument("--test_batch_size", type=int, default=8, help="")
    parser.add_argument("--num_beams", type=int, default=3, help="")

    args, unknown = parser.parse_known_args()

    return args, unknown


def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    args, unknown = read_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    test_loader = init_loader(test_batch_size=args.test_batch_size, split="test", data_path=args.data_path, dataset_name=args.dataset_name, dataset_config_name=args.dataset_config_name, max_test_samples=args.max_test_samples)
    model, tokenizer = load_model(model_path=args.model_path, tokenizer_name=args.tokenizer_name, device=device)
    
    model.eval()
    generated_sums, target_sums, article_ids = generate_summaries(
        test_loader, model, tokenizer, device=device, args=args)

    metrics, mdf = score_standard(
        gen_sums=generated_sums,
        target_sums=target_sums,
        article_ids=article_ids)

    print(mdf)
    print(metrics)
    
    mdf.to_csv(os.path.join(args.output_path, "generated_sums.csv"), sep="\t", index=False)
    metrics.to_csv(os.path.join(args.output_path, "metrics.csv"), index=False)


if __name__ == "__main__":
    main()
