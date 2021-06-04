import numpy as np

import matplotlib.pyplot as plt


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
        axs_j = axs[j] if isinstance(axs, np.ndarray) else axs
        for norm_df, model, n in zip(norm_run_dfs, run_models, n_list):
            norm_df["bleuvar_scaled"].plot(label=f"Var{model}-{n}", ax=axs_j)

        axs_j.set_xticks(range(0, num_steps + 100, int(num_steps / 10)))
        locator_len = len(axs_j.xaxis.get_major_locator().locs)
        axs_j.set_xticklabels([x / 10 for x in range(0, locator_len)])
        axs_j.set_xlabel('Fraction of data retained')
        axs_j.set_ylabel(f"{dataset}")
        axs_j.legend(loc='best')

    fig.text(0.5, 0.92, 'BLEUVAR', ha='center', va='center', fontsize=15)
    fig.savefig(f"{save_path}/bleuvars_{'_'.join(dataset_runs.keys())}_new.eps", format="eps")
    print(f"Exported {save_path}/bleuvars_{'_'.join(dataset_runs.keys())}_new.eps")


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