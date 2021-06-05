import argparse
import os
import random

import torch

from src.bayesian import BayesianSummarizer
from src.loaders import init_loader, load_model
from src.scoring import score_standard


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
    parser.add_argument("--mc_samples", type=int, default=10, help="")
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

    test_loader = init_loader(args)
    model, tokenizer = load_model(args, device=device)
    bayesian_summarizer = BayesianSummarizer(model=model, tokenizer=tokenizer)

    generated_sums, target_sums, article_ids, bleuvars = bayesian_summarizer.generate_bayesian_summaries(
        test_loader, device=device, args=args)

    metrics, mdf = score_standard(
        gen_sums=generated_sums,
        target_sums=target_sums,
        article_ids=article_ids)

    mdf["bleuvar"] = bleuvars
    print(mdf)
    print(metrics)
    
    mdf.to_csv(os.path.join(args.output_path, "generated_sums.csv"), sep="\t", index=False)
    metrics.to_csv(os.path.join(args.output_path, "metrics.csv"), index=False)


if __name__ == "__main__":
    main()
