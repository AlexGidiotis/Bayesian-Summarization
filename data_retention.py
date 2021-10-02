import argparse
import json
import os
import sys

import pandas as pd

from src.plotting import plot_rouge_retention, plot_increase, plot_bleuvar_retention


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs='+', help="")
    parser.add_argument("--root_path", type=str, help="")
    parser.add_argument("--models", nargs='+', help="")
    parser.add_argument("--n_list", nargs='+', type=int, help="")
    parser.add_argument("--bases", nargs='+', help="")

    args, unknown = parser.parse_known_args()

    return args, unknown


def metrics_change(df):
    """"""
    num_samples = len(df)
    percent_increase = int(num_samples / 1000)

    metrics_change_df = df.expanding().mean()
    metrics_change_df = metrics_change_df[1:].iloc[::percent_increase, :].reset_index(drop=True)

    return metrics_change_df


def load_runs(run_list, n_list):
    """"""
    run_dfs = []
    for run_path, n in zip(run_list, n_list):
        try:
            df = pd.read_csv(run_path, sep="\t")
            sorted_df = df.sort_values(by=['bleuvar'], ignore_index=True)
        except FileNotFoundError:
            sorted_df = None
        run_dfs.append(sorted_df)

    return run_dfs


def load_bases(base_list):
    """"""
    agg_base_dfs = []
    base_dfs = []
    for base_path in base_list:
        try:
            base_df = pd.read_csv(base_path, sep="\t")
            agg_base_df = base_df.agg(['mean', 'std'])
        except FileNotFoundError:
            base_df = None
            agg_base_df = base_df
        base_dfs.append(base_df)
        agg_base_dfs.append(agg_base_df)

    return base_dfs, agg_base_dfs


def norm_bleuvar(runs, n_list):
    """"""
    scaled_metrics_dfs = []
    for metrics_change_df, n in zip(runs, n_list):
        if metrics_change_df is not None:
            metrics_change_df["bleuvar_scaled"] = metrics_change_df["bleuvar"].apply(lambda x: x / (n * (n - 1)))

        scaled_metrics_dfs.append(metrics_change_df)

    return scaled_metrics_dfs


def join_runs(runs, run_models, base_runs, base_models):
    """"""
    joined_dfs = []
    for model, metrics_df in zip(run_models, runs):
        base_df = base_runs[base_models.index(model)]
        if base_df is not None:
            join_df = metrics_df.merge(base_df, on="article_id", suffixes=["_var", "_std"])
        else:
            join_df = base_df
        joined_dfs.append(join_df)

    return joined_dfs


def compute_diffs(joined_dfs, metrics):
    """"""
    diff_dfs = []
    for join_df in joined_dfs:
        if join_df is not None:
            for metric in metrics:
                join_df[f"{metric}_diff"] = join_df[[f"{metric}_var", f"{metric}_std"]] \
                    .apply(lambda x: x[0] - x[1], axis=1) \
                    .clip(-1, 1)
        diff_dfs.append(join_df)

    return diff_dfs


def aggregate_runs(dfs):
    """"""
    agg_dfs = []
    for df in dfs:
        if df is not None:
            metrics_df = metrics_change(df)
        else:
            metrics_df = df
        agg_dfs.append(metrics_df)

    return agg_dfs


def increase_metrics(runs, run_models, n_list, fractions, metrics):
    """"""
    all_metrics = {}
    for run_df, model, n in zip(runs, run_models, n_list):
        if run_df is None:
            continue
        model_metrics = {}
        for i, frac in enumerate(fractions):
            discr_df = run_df.describe()
            frac_metrics_df = run_df[run_df["bleuvar"] >= discr_df["bleuvar"][frac]][
                ["rouge1_std", "rouge2_std", "rougeL_std"]] \
                .agg(['mean', 'std'])
            frac_metrics = {}
            for metric in metrics:
                frac_increase = (
                    (discr_df[f"{metric}_std"]["mean"] - frac_metrics_df[f"{metric}_std"]["mean"])
                    / discr_df[f"{metric}_var"]["mean"]) * 100
                frac_metrics[metric] = frac_increase
            model_metrics[frac] = frac_metrics
        all_metrics[f"Var{model}-{n}"] = model_metrics

    return all_metrics


def main():
    args, unknown = read_args()

    data_list = args.data
    root_data_path = args.root_path
    model_list = args.models
    base_model_list = args.bases
    metrics_list = ["rouge1", "rouge2", "rougeL"]
    n_list = args.n_list
    fraction_list = ["75%", "50%", "25%"]

    data_runs = {}
    bases = {}
    diffs = {}
    for data in data_list:
        print(f"Running dataset {data}...")
        run_list = [
            f"{root_data_path}/var{model.lower()}{n}_{data}/generated_sums.csv" for model, n in zip(model_list, n_list)
        ]

        base_list = [
            f"{root_data_path}/{base_model.lower()}_{data}/generated_sums.csv" for base_model in base_model_list
        ]

        run_dfs = load_runs(run_list, n_list)
        base_dfs, agg_base_dfs = load_bases(base_list)
        bases[data] = agg_base_dfs
        join_dfs = join_runs(runs=run_dfs, run_models=model_list, base_runs=base_dfs, base_models=base_model_list)
        incr_metrics = increase_metrics(
            runs=join_dfs,
            run_models=model_list,
            n_list=n_list,
            fractions=fraction_list,
            metrics=metrics_list)

        with open(os.path.join(root_data_path, f"{data}_increase_metrics.json"), "w") as mf:
            json.dump(incr_metrics, mf)
            print(f"Exported {data}_increase_metrics.json")
        agg_run_dfs = aggregate_runs(join_dfs)

        norm_run_dfs = norm_bleuvar(runs=agg_run_dfs, n_list=n_list)
        data_runs[data] = norm_run_dfs

        diff_dfs = compute_diffs(joined_dfs=agg_run_dfs, metrics=metrics_list)
        diffs[data] = diff_dfs

    plot_rouge_retention(
        metrics=metrics_list,
        dataset_runs=data_runs,
        run_models=model_list,
        base_runs=bases,
        base_models=base_model_list,
        n_list=n_list,
        save_path=root_data_path,
        ascending=False,
    )

    plot_bleuvar_retention(
        dataset_runs=data_runs,
        run_models=model_list,
        n_list=n_list,
        save_path=root_data_path,
        ascending=True, )

    plot_increase(
        dataset_runs=data_runs,
        metrics=metrics_list,
        diff_runs=diffs,
        run_models=model_list,
        n_list=n_list,
        save_path=root_data_path,
        ascending=False, )


if __name__ == "__main__":
    main()
