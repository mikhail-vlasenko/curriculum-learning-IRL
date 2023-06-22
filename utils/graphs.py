import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def clean_df(df):
    columns = []
    for c in df.columns:
        if "__MIN" not in c and "__MAX" not in c:
            columns.append(c)
    df = df[columns]

    # useful_columns = []
    # for c in df.columns:
    #     for run_id in useful_runs:
    #         if str(run_id) in c:
    #             useful_columns.append(c)
    # df = df[useful_columns]

    df = df.fillna(method='ffill')
    return df


def process_and_plot(
        df, run_groups, group_names,
        smoothing_window=5, title='Non-discounted true returns in the target environment',
        vertical_lines: List[int] = [],
):
    processed_columns = []
    means = []
    stds = []

    # Process each group
    for i, group in enumerate(run_groups):
        group_columns = []

        # Filter group columns
        for c in df.columns:
            for run_id in group:
                if str(run_id) in c:
                    group_columns.append(c)

        # Smooth group columns
        for c in group_columns:
            df[c] = df[c].rolling(smoothing_window).mean()

        # Compute mean and standard deviation
        mean_col = f'group_{i}_mean'
        std_col = f'group_{i}_std'
        df[mean_col] = df[group_columns].mean(axis=1)
        df[std_col] = df[group_columns].std(axis=1)

        # Append processed columns and names
        processed_columns.append(group_columns)
        means.append(mean_col)
        stds.append(std_col)

    # Plot
    fig, ax = plt.subplots()
    fig.set_dpi(200)
    fig.set_size_inches(8, 6)
    for mean, std, name in zip(means, stds, group_names):
        ax.plot(df['Step'], df[mean], label=f'{name}')
        ax.fill_between(df['Step'], df[mean] - df[std], df[mean] + df[std], alpha=0.5)

    if len(vertical_lines) > 1:
        for i, line in enumerate(vertical_lines):
            ax.axvline(line, color=default_colors[i], linestyle='--')
    else:
        ax.axvline(vertical_lines[0], color='black', linestyle='--')

    # ax.axhline(-1.5, color='black', linestyle='--', alpha=0.5)
    # ax.axhline(3, color='black', linestyle='--', alpha=0.5)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Returns')
    plt.show()
    return df, processed_columns


def fixed_airl():
    lines = []
    df = pd.read_csv('../graph_data/fixed_airl.csv')
    old_runs = [121, 122, 123]
    new_runs = [203, 204, 210]
    group_names = ['using code from Peschl (2022)', 'corrected end reward estimation']
    run_groups = [old_runs, new_runs]
    title = 'Impact of end reward estimation correction on true returns'
    return df, run_groups, group_names, title, lines


def diff_swap_point():
    df = pd.read_csv('../graph_data/different_swap_point.csv')
    title = 'Non-discounted true returns in the target environment'

    # old_runs = [203, 204, 210]
    lines = []
    run_groups = []
    group_names = []
    # ------
    run_groups.append([214, 215, 216])
    lines.append(100000)
    group_names.append('env swap at 100k')

    run_groups.append([211, 212, 213])
    lines.append(150000)
    group_names.append('env swap at 150k')

    run_groups.append([206, 208, 209])
    lines.append(200000)
    group_names.append('env swap at 200k')

    run_groups.append([194, 195, 217])
    lines.append(300000)
    group_names.append('env swap at 300k')
    # ------
    df = df[df['Step'] <= 500000]
    return df, run_groups, group_names, title, lines


def fifty_demos():
    df = pd.read_csv('../graph_data/fifty_demos.csv')
    title = 'Non-discounted true returns in the target environment'

    lines = [200000]
    baseline = [224, 228, 229, 230, 231]
    cl = [225, 226, 227, 232, 233]
    run_groups = [baseline, cl]
    group_names = ['baseline', 'CL']
    return df, run_groups, group_names, title, lines


def main():
    plt.rcParams.update({'font.size': 15})

    smoothing_window = 20

    # df, run_groups, group_names, title, lines = fixed_airl()
    # df, run_groups, group_names, title, lines = diff_swap_point()
    df, run_groups, group_names, title, lines = fifty_demos()

    df = clean_df(df)
    process_and_plot(df, run_groups, group_names, smoothing_window, vertical_lines=lines, title=title)


if __name__ == '__main__':
    main()
