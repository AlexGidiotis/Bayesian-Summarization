import sys

import numpy as np

import matplotlib.pyplot as plt


def plot_rouge_retention(metrics, dataset_runs, run_models, n_list, base_runs, base_models, save_path, ascending=True):
    """"""
    plt.rcParams['figure.figsize'] = [18, 6]
    color_map = {0: 'r', 1: 'b'}
    # plot metrics
    for i, metric in enumerate(metrics):
        fig, axs = plt.subplots(1, len(dataset_runs))
        if len(dataset_runs) == 1:
            axs = [axs]
        for j, dataset in enumerate(dataset_runs.keys()):
            norm_runs = dataset_runs[dataset]
            for metrics_change_df, model, n in zip(norm_runs, run_models, n_list):
                if metrics_change_df is None:
                    continue
                metrics_change_df = metrics_change_df.sort_values(
                        "bleuvar_scaled", ascending=ascending, ignore_index=True) \
                    .rolling(3).mean()
                metrics_change_df[f"{metric}_std"].plot(label=f"Var{model}-{n}", ax=axs[j])

            num_steps = len(norm_runs[0])
            for k, (metrics_base, base_model) in enumerate(zip(base_runs[dataset], base_models)):
                if metrics_base is None:
                    continue
                base_r = axs[j].axhline(y=metrics_base[metric]["mean"], color=color_map[k], linestyle='--')
                base_r.set_label(f"{base_model}")

            axs[j].set_xticks(range(0, num_steps, int(num_steps / 10)))
            locator_len = len(axs[j].xaxis.get_major_locator().locs)
            axs[j].set_xticklabels([x / 10 for x in range(0, locator_len)])
            axs[j].set_xlabel('Fraction of data discarded')
            axs[j].set_ylabel(f"{metric}")

            axs[j].legend(loc='best')

            axs[j].set_title(f'{dataset}', fontsize=15)

        fig.savefig(f'{save_path}/{metric}.pdf', format='pdf')
        print(f"Exported {save_path}/{metric}.pdf")


def plot_bleuvar_retention(dataset_runs, run_models, n_list, save_path, ascending=True):
    """"""
    plt.rcParams['figure.figsize'] = [18, 6]

    fig, axs = plt.subplots(1, len(dataset_runs.keys()))
    for j, dataset in enumerate(dataset_runs.keys()):
        norm_run_dfs = dataset_runs[dataset]

        num_steps = len(norm_run_dfs[0])
        axs_j = axs[j] if isinstance(axs, np.ndarray) else axs
        for norm_df, model, n in zip(norm_run_dfs, run_models, n_list):
            if norm_df is None:
                continue
            norm_df = norm_df.sort_values("rouge1_std", ascending=ascending, ignore_index=True) \
                .rolling(3).mean()
            norm_df["bleuvar_scaled"].plot(label=f"Var{model}-{n}", ax=axs_j)

        axs_j.set_xticks(range(0, num_steps + 100, int(num_steps / 10)))
        locator_len = len(axs_j.xaxis.get_major_locator().locs)
        axs_j.set_xticklabels([x / 10 for x in range(0, locator_len)])
        axs_j.set_xlabel('Fraction of data discarded')
        axs_j.set_ylabel(f"{dataset}")
        axs_j.legend(loc='best')

    fig.text(0.5, 0.92, 'BLEUVAR', ha='center', va='center', fontsize=15)
    fig.savefig(f"{save_path}/bleuvars_{'_'.join(dataset_runs.keys())}.pdf", format="pdf")
    print(f"Exported {save_path}/bleuvars_{'_'.join(dataset_runs.keys())}.pdf")


def plot_increase(dataset_runs, run_models, metrics, n_list, save_path, ascending=True):
    """"""
    plt.rcParams['figure.figsize'] = [18, 6]
    for i, metric in enumerate(metrics):
        fig, axs = plt.subplots(1, len(dataset_runs))
        if len(dataset_runs) == 1:
            axs = [axs]
        for j, dataset in enumerate(dataset_runs.keys()):
            diff_runs = dataset_runs[dataset]
            for diff_df, model, n in zip(diff_runs, run_models, n_list):
                if diff_df is None:
                    continue

                diff_df = diff_df.sort_values("bleuvar_scaled", ascending=ascending, ignore_index=True) \
                    .rolling(3).mean()
                diff_df[f"{metric}_diff"].plot(label=f"Var{model}-{n}", ax=axs[j])

            num_steps = len(diff_runs[0])
            base_r = axs[j].axhline(y=0, color="k", linestyle='--')
            axs[j].set_xticks(range(0, num_steps + 100, int(num_steps / 10)))
            locator_len = len(axs[j].xaxis.get_major_locator().locs)
            axs[j].set_xticklabels([x / 10 for x in range(0, locator_len)])
            axs[j].set_xlabel('Fraction of data discarded')
            axs[j].set_ylabel(f"{metric} difference")

            axs[j].legend(loc='best')
            axs[j].set_title(f'{dataset}', fontsize=15)

        fig.savefig(f'{save_path}/{metric}_diff.pdf', format='pdf')
        print(f"Exported {save_path}/{metric}_diff.pdf")
