import globals as gl
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np


def plot_days(df, y=None):
    _, axs = plt.subplots()
    sns.pointplot(df, ax=axs, x='day', y=y, hue='chord', dodge=True, linestyle='none', errorbar='se',
                  palette=['red', 'blue'])


def plot_rep(df, y=None):

    _, axs = plt.subplots()
    sns.lineplot(data=df[df['chord'] == 'trained'], ax=axs, x='offset', y=y, hue='day', errorbar='se',
                 palette=['red'] * 5, marker='o', markeredgewidth=0, err_kws={'linewidth': 0})
    sns.lineplot(data=df[df['chord'] == 'untrained'], ax=axs, x='offset', y=y, hue='day', errorbar='se',
                 palette=['blue'] * 2, marker='o', markeredgewidth=0, err_kws={'linewidth': 0})
    axs.set_xticks(np.linspace(8, 28, 5))
    axs.set_xticklabels(np.linspace(1, 5, 5, dtype=int))
    axs.set_xlabel('day')

    # Remove the automatic legend
    axs.get_legend().remove()

    # Create custom legend handles and labels
    custom_handles = [
        Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', label='untrained'),
        Line2D([0], [0], marker='o', color='red', markerfacecolor='red', label='trained')
    ]

    # Add the custom legend to the plot
    axs.legend(handles=custom_handles, title='chord')


def calc_avg(X, columns=None, by=None):
    """
        Computes the average MD for each chordID in the given dataframe.

        Parameters:
        data (pd.DataFrame): The input dataframe containing 'chordID' and 'MD' columns.

        Returns:
        pd.DataFrame: A dataframe with 'chordID' and the corresponding average 'MD'.
        """
    # Group by 'chordID' and compute the mean of 'MD'
    if X is str:
        data = pd.read_csv(X)
    elif isinstance(X, pd.DataFrame):
        data = X
    else:
        data = None

    columns = {col: 'mean' for col in columns}
    avg = data.groupby(by).agg(columns).reset_index()
    # md.rename(columns={'MD': 'average_MD'}, inplace=True)

    return avg


if __name__ == "__main__":
    experiment = 'efc2'
    participants = [
        'subj100',
        'subj101',
        'subj102',
        'subj103',
        'subj104',
        # 'subj105',
        'subj106',
        'subj107'
    ]
    sessions = ['testing',
                'training']
    days = ['1', '2', '3', '4', '5']

    columns = ['MD', 'RT', 'angle', 'sine', 'jerk', 'PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'dist0', 'dist1', 'dist2',
               'dist3', 'dist4']

    results = pd.read_csv(os.path.join(gl.baseDir, experiment, 'results.csv'))
    df = calc_avg(results, by=['chord', 'day', 'participant_id'], columns=columns)

    plot_days(df, y='RT')
    plot_days(df, y='MD')
    plot_days(df, y='sine')
    plot_days(df, y='jerk')

    _, axs = plt.subplots()
    df_melted = df.melt(id_vars=['day', 'chord'],
                        value_vars=[f'PC{i}' for i in range(5)],
                        var_name='PCs', value_name='Explained')
    sns.pointplot(df_melted[df_melted['chord'] == 'untrained'], x='PCs', y='Explained', hue='day',
                  errorbar='se', palette='Blues')
    sns.pointplot(df_melted[df_melted['chord'] == 'trained'], x='PCs', y='Explained', hue='day',
                  errorbar='se', palette='Reds')

    df = calc_avg(results, by=['chord', 'day', 'repetition', 'participant_id'], columns=columns)
    offset = 5
    df['offset'] = df['repetition'] + df['day'] * offset

    plot_rep(df, 'MD')
    plot_rep(df, 'RT')
    plot_rep(df, 'jerk')
    plot_rep(df, 'sine')

    plt.show()
