import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_df(df):
    columns = []
    for c in df.columns:
        if "__MIN" not in c and "__MAX" not in c:
            columns.append(c)
    df = df[columns]
    df = df.fillna(method='ffill')
    return df


def process_and_plot(
        df, run_groups, group_names,
        smoothing_window=5, title='True reward in the environment'
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
    for mean, std, name in zip(means, stds, group_names):
        ax.plot(df['Step'], df[mean], label=f'{name}')
        ax.fill_between(df['Step'], df[mean] - df[std], df[mean] + df[std], alpha=0.5)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    plt.show()
    return df, processed_columns


def main():
    df = pd.read_csv('graph_data/fixed_airl.csv')
    smoothing_window = 10

    df = clean_df(df)

    # fixed airl graph
    old_runs = [121, 122, 123]
    new_runs = [162, 163, 166]
    run_groups = [old_runs, new_runs]
    group_names = ['original implementation', 'fixed end reward estimation']

    process_and_plot(df, run_groups, group_names, smoothing_window)


if __name__ == '__main__':
    main()
