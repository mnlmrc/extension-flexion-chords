import globals as gl
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np


def plot_days(df, y=None, title=None, ylabel=None, xlabel=None, save_path=None):
    if title == None:
        title = y

    fig, axs = plt.subplots()
    sns.pointplot(df, ax=axs, x='day', y=y, hue='chord', dodge=True, linestyle='none', errorbar='se',
                  palette=['red', 'blue'])
    axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)

    if save_path is not None:
        fig.savefig(save_path)


def plot_rep(df, y=None, title=None, ylabel=None, xlabel=None, save_path=None):

    if title == None:
        title = y

    fig, axs = plt.subplots()
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

    axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)

    if save_path is not None:
        fig.savefig(save_path)


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

def calc_success(X):

    success = X.groupby(['subNum', 'chordID', 'day', 'chord'])['trialPoint'].mean().reset_index()
    success.rename(columns={'trialPoint': 'success'}, inplace=True)

    success.sort_values(by='chord', inplace=True)

    return success


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

    results = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.csv'))
    df_day = calc_avg(results, by=['chord', 'day', 'participant_id'], columns=columns)

    save_path = os.path.join(gl.baseDir, experiment, 'figures')

    plot_days(df_day, y='RT', ylabel='RT (s)', xlabel='day', title='reaction time', save_path=os.path.join(save_path, 'rt.png'))
    plot_days(df_day, y='MD', ylabel='MD (a.u.)', xlabel='day', title='mean deviation', save_path=os.path.join(save_path, 'md.png'))
    plot_days(df_day, y='sine', ylabel='sine of first PC', xlabel='day', title='sine of first principal component', save_path=os.path.join(save_path, 'sine.png'))
    plot_days(df_day, y='jerk', ylabel='jerk (N/s\u00b3)', xlabel='day', title='jerk', save_path=os.path.join(save_path, 'jerk.png'))

    df_rep = calc_avg(results, by=['chord', 'day', 'repetition', 'participant_id'], columns=columns)
    offset = 5
    df_rep['offset'] = df_rep['repetition'] + df_rep['day'] * offset

    plot_rep(df_rep, 'MD', ylabel='MD (a.u.)', xlabel='day',
             title='mean deviation', save_path=os.path.join(save_path, 'md_rep.png'))
    plot_rep(df_rep, 'RT', ylabel='RT (s)', xlabel='day',
             title='reaction time', save_path=os.path.join(save_path, 'rt_rep.png'))
    # plot_rep(df_rep, 'jerk', ylabel='jerk (N/s\u00b3)', xlabel='day',
    #          title='jerk', save_path=os.path.join(save_path, 'jerk_rep.png'))
    # plot_rep(df_rep, 'sine', ylabel='sine of first PC', xlabel='day',
    #          title='sine of first principal component', save_path=os.path.join(save_path, 'sine_rep.png'))

    df_success = calc_success(results)
    plot_days(df_success, y='success', ylabel='%success', xlabel='day',
              title='success rate', save_path=os.path.join(save_path, 'success.png'))

    fig, axs = plt.subplots()
    sns.pointplot(df_success, ax=axs, x='day', y='success', hue='chordID', dodge=True, linestyle='none', errorbar='se',
                  palette='tab10')
    axs.set_title('success rate')
    axs.set_ylabel('%success')
    axs.set_xlabel('day')
    fig.savefig(os.path.join(save_path, 'success_chord.png'))

    plt.show()
