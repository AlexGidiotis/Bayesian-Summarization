from tqdm import tqdm
import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        df = pd.read_csv(run_path, sep="\t")
        sorted_df = df.sort_values(by=['bleuvar'], ignore_index=True)

        run_dfs.append(sorted_df)

    return run_dfs


def load_bases(base_list):
    """"""
    agg_base_dfs = []
    base_dfs = []
    for base_path in base_list:
        base_df = pd.read_csv(base_path, sep="\t")
        agg_base_df = base_df.agg(['mean', 'std'])

        base_dfs.append(base_df)
        agg_base_dfs.append(agg_base_df)

    return base_dfs, agg_base_dfs


def norm_bleuvar(runs, n_list):
    """"""
    scaled_metrics_dfs = []
    for metrics_change_df, n in zip(runs, n_list):
        metrics_change_df["bleuvar_scaled"] = metrics_change_df["bleuvar"].apply(lambda x: x / (n * (n - 1)))
        scaled_metrics_dfs.append(metrics_change_df)

    return scaled_metrics_dfs


def join_runs(runs, run_models, base_runs, base_models):
    """"""
    joined_dfs = []
    for model, metrics_df in zip(run_models, runs):
        base_df = base_runs[base_models.index(model)]
        join_df = metrics_df.merge(base_df, on="article_id", suffixes=["_var", "_std"])
        joined_dfs.append(join_df)

    return joined_dfs


def compute_diffs(joined_dfs, metrics):
    """"""
    diff_dfs = []
    for join_df in joined_dfs:
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
        metrics_df = metrics_change(df)
        agg_dfs.append(metrics_df)

    return agg_dfs

def plot_rouge_retention(data, metrics, runs, run_models, n_list, base_runs, base_models, save_path):
    """"""
    plt.rcParams['figure.figsize'] = [18, 6]
    fig, axs = plt.subplots(1, 3)

    # plot metrics
    for metrics_change_df, model, n in zip(runs, run_models, n_list):
        for j, metric in enumerate(metrics):
            metrics_change_df[f"{metric}_var"].plot(label=f"Var{model}-{n}", ax=axs[j])

    num_steps = len(runs[0])

    # plot baselines
    color_map = {0: 'r', 1: 'b'}
    for i, (metrics_base, base_model) in enumerate(zip(base_runs, base_models)):
        for j, metric in enumerate(metrics):
            base_r = axs[j].axhline(y=metrics_base[metric]["mean"], color=color_map[i], linestyle='--')
            base_r.set_label(f"{base_model}")

    for j, metric in enumerate(metrics):
        axs[j].set_xticks(range(0, num_steps + 100, int(num_steps / 10)))
        locator_len = len(axs[j].xaxis.get_major_locator().locs)
        axs[j].set_xticklabels([x / 10 for x in range(0, locator_len)])
        axs[j].set_xlabel('Fraction of data retained')
        axs[j].set_ylabel(f"{metric}")

        axs[j].legend(loc='best')

    fig.text(0.5, 0.92, f'{data} dataset', ha='center', va='center', fontsize=15)

    fig.savefig(f'{save_path}/{data}_new_data.eps', format='eps')
    print(f"Exported {save_path}/{data}_new_data.eps")


def plot_bleuvar_retention(dataset_runs, run_models, n_list, save_path):
    """"""
    plt.rcParams['figure.figsize'] = [18, 6]

    fig, axs = plt.subplots(1, len(dataset_runs.keys()))
    for j, dataset in enumerate(dataset_runs.keys()):
        norm_run_dfs = dataset_runs[dataset]

        num_steps = len(norm_run_dfs[0])
        for norm_df, model, n in zip(norm_run_dfs, run_models, n_list):
            norm_df["bleuvar_scaled"].plot(label=f"Var{model}-{n}", ax=axs[j])

        axs[j].set_xticks(range(0, num_steps + 100, int(num_steps / 10)))
        locator_len = len(axs[j].xaxis.get_major_locator().locs)
        axs[j].set_xticklabels([x / 10 for x in range(0, locator_len)])
        axs[j].set_xlabel('Fraction of data retained')
        axs[j].set_ylabel(f"{dataset}")
        axs[j].legend(loc='best')

    fig.text(0.5, 0.92, 'BLEUVAR', ha='center', va='center', fontsize=15)
    fig.savefig(f'{save_path}/bleuvars_new.eps', format='eps')
    print(f"Exported {save_path}/bleuvars_new.eps")


def plot_increase(data, diff_runs, run_models, metrics, n_list, save_path):
    """"""
    plt.rcParams['figure.figsize'] = [18, 6]
    fig, axs = plt.subplots(1, 3)

    num_steps = len(diff_runs[0])
    for diff_df, model, n in zip(diff_runs, run_models, n_list):
        for j, metric in enumerate(metrics):
            diff_df[f"{metric}_diff"].plot(label=f"Var{model}-{n}", ax=axs[j])

    for j, metric in enumerate(metrics):
        axs[j].set_xticks(range(0, num_steps + 100, int(num_steps / 10)))
        locator_len = len(axs[j].xaxis.get_major_locator().locs)
        axs[j].set_xticklabels([x / 10 for x in range(0, locator_len)])
        axs[j].set_xlabel('Fraction of data retained')
        axs[j].set_ylabel(f"{metric} difference")

        axs[j].legend(loc='best')

    fig.text(0.5, 0.92, f'{data} dataset', ha='center', va='center', fontsize=15)
    fig.savefig(f'{save_path}/{data}_diff_new.eps', format='eps')
    print(f"Exported {save_path}/{data}_diff_new.eps")


if __name__ == "__main__":
    data_list = ["xsum", "cnn_dailymail"]
    root_data_path = "exp_runs"
    model_list = [
        "PEGASUS",
        "PEGASUS",
        "BART",
        "BART",
    ]
    base_model_list = [
        "PEGASUS",
        "BART",
    ]
    metrics_list = ["rouge1", "rouge2", "rougeL"]
    n_list = [10, 20, 10, 20]

    data_runs = {}
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
        join_dfs = join_runs(runs=run_dfs, run_models=model_list, base_runs=base_dfs, base_models=base_model_list)
        agg_run_dfs = aggregate_runs(join_dfs)

        norm_run_dfs = norm_bleuvar(runs=agg_run_dfs, n_list=n_list)
        data_runs[data] = norm_run_dfs

        plot_rouge_retention(
            data=data,
            metrics=metrics_list,
            runs=agg_run_dfs,
            run_models=model_list,
            base_runs=agg_base_dfs,
            base_models=base_model_list,
            n_list=n_list,
            save_path=root_data_path
        )

        diff_dfs = compute_diffs(joined_dfs=agg_run_dfs, metrics=metrics_list)

        plot_increase(
            data=data,
            metrics=metrics_list,
            diff_runs=diff_dfs,
            run_models=model_list,
            n_list=n_list,
            save_path=root_data_path)

    plot_bleuvar_retention(dataset_runs=data_runs, run_models=model_list, n_list=n_list, save_path=root_data_path)
