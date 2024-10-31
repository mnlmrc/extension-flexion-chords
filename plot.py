import argparse
import os
import pickle
import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from force import Force
from main import main
import globals as gl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.rcParams['svg.fonttype'] = 'none'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp


def savefig(path, fig, format='svg'):
    # Check if file exists
    if os.path.exists(path):
        response = input(f"The file {path} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() == 'y':
            fig.savefig(path, dpi=600)
            print(f"File {path} has been overwritten.")
        else:
            print("File not saved. Please choose a different name or path.")
    else:
        fig.savefig(path, dpi=600, format=format)
        print(f"File saved as {path}.")


def decor(axs=None, fontsize=None, xlim=(None, None), ylim=(None, None), xbounds=(None, None), ybounds=(None, None),
          spines_width=None):
    axs.set_ylim(ylim)
    axs.set_xlim(xlim)

    axs.spines['bottom'].set_bounds(xbounds[0], xbounds[1])
    axs.spines['left'].set_bounds(ybounds[0], ybounds[1])
    axs.spines[['right', 'top']].set_visible(False)
    axs.spines[['left', 'bottom']].set_linewidth(spines_width)
    axs.tick_params(axis='both', width=spines_width)

    axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
    axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)


def add_significance_brackets(ax, data, x=None, y=None, pairs=None, test_type='t-test_ind', text_format='star',
                              significance_level=0.05, y_pos=None, x1_pos=None, x2_pos=None):
    """
    Adds significance brackets with asterisks to an existing plot.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object where the data is already plotted.
        data (pd.DataFrame): The data for testing.
        x (str): The name of the column for the x-axis.
        y (str): The name of the column for the y-axis.
        hue (str): The name of the column for hue.
        pairs (list of tuples): List of tuples specifying the pairs to compare (e.g., [(1, 2), (3, 4)]).
        test_type (str): The type of t-test to run ('t-test_ind' for independent t-test).
        text_format (str): Format of the p-value display ('star' for asterisks).
        line_height (float): Height of the bracket above the data points, relative to the y-axis scale.
    """

    # Determine the y-axis limits to place the brackets properly
    if y_pos is None:
        all_y_data = []
        for line in ax.get_lines():
            y_data = line.get_ydata()
            all_y_data.extend(y_data)  # Collect all y-values from the plotted lines

            # Find the maximum y-value from the collected data
            y_pos = max(all_y_data) + 0.1 * max(all_y_data)

    brack_height = 0.01 * y_pos

    if x1_pos is None:
        x1_pos = 0

    if x2_pos is None:
        x2_pos = 1

    for pair in pairs:
        x1, x2 = pair

        data1 = data[(data[x] == x1)][y]
        data2 = data[(data[x] == x2)][y]

        # Perform t-test
        if test_type == 't-test_ind':
            stat, p_value = ttest_ind(data1, data2, nan_policy='omit')
        elif test_type == 't-test_rel':
            stat, p_value = ttest_rel(data1, data2, nan_policy='omit')
        else:
            raise ValueError(f"Test type '{test_type}' not supported.")

        # Determine significance level
        if text_format == 'star':
            if p_value < significance_level / 500:
                significance = '****'
            elif p_value < significance_level / 50:
                significance = '***'
            elif p_value < significance_level / 5:
                significance = '**'
            elif p_value < significance_level:
                significance = '*'
            else:
                significance = 'n.s.'
        else:
            significance = f"p = {p_value:.3e}"

        # Plot significance bracket
        ax.plot([x1_pos, x1_pos, x2_pos, x2_pos], [y_pos, y_pos + brack_height, y_pos + brack_height, y_pos], lw=1.5, color='k')
        ax.text((x1_pos + x2_pos) * 0.5, y_pos + brack_height, significance, ha='center', va='baseline', color='k', fontsize=12)

        return y_pos + 4 * brack_height
    # # Adjust the y-axis to accommodate the significance brackets
    # ax.set_ylim([ax.get_ylim()[0], y_max + 3 * line_height])


def add_significance_asterisks(ax, data, x=None, y=None, hue=None, x_point=None, popmean=None, test_type='t-test_ind',
                               significance_level=0.05, text_format='star', y_pos=None, color='k'):
    """
    Adds significance asterisks to an existing plot for comparisons between groups at the same x point.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object where the data is already plotted.
        data (pd.DataFrame): The data for testing.
        x (str): The name of the column for the x-axis.
        y (str): The name of the column for the y-axis.
        hue (str): The name of the column for hue (grouping variable).
        test_type (str): The type of t-test to run ('t-test_ind' for independent t-test, 't-test_rel' for paired t-test).
        significance_level (float): The significance level to consider a result significant (default: 0.05).
        text_format (str): Format of the p-value display ('star' for asterisks).
    """

    # Unique x points
    # x_points = data[x].unique()

    data_avg = data[data[x] == x_point].groupby(hue)[y].mean().reset_index()
    y_max = data_avg.max().values[1]

    # for x_point in x_points:
    # Get data for the two groups at the same x point
    group_data = data[data[x] == x_point]
    hue_levels = group_data[hue].unique()

    # if len(hue_levels) != 2:
    #     raise ValueError("There must be exactly two groups at each x point to perform the t-test.")

    # data1 = group_data[group_data[hue] == hue_levels[0]][y]
    # data2 = group_data[group_data[hue] == hue_levels[1]][y]

    # Perform t-test
    if test_type == 't-test_ind':
        data1 = group_data[group_data[hue] == hue_levels[0]][y]
        data2 = group_data[group_data[hue] == hue_levels[1]][y]
        stat, p_value = ttest_ind(data1, data2, nan_policy='omit')
    elif test_type == 't-test_rel':
        data1 = group_data[group_data[hue] == hue_levels[0]][y]
        data2 = group_data[group_data[hue] == hue_levels[1]][y]
        stat, p_value = ttest_rel(data1, data2, nan_policy='omit')
    elif test_type == 't-test_1samp':
        data_1samp = group_data[y].to_numpy()
        stat, p_value = ttest_1samp(data_1samp, popmean, nan_policy='omit')
    else:
        raise ValueError(f"Test type '{test_type}' not supported.")

    # Determine if the result is significant
    if p_value < significance_level:
        if text_format == 'star':
            if p_value < significance_level / 500:
                significance = '****'
            elif p_value < significance_level / 50:
                significance = '***'
            elif p_value < significance_level / 5:
                significance = '**'
            elif p_value < significance_level:
                significance = '*'
        else:
            significance = f"p = {p_value:.3e}"

        # Find the maximum y-value for the x point and add an asterisk above it
        if y_pos is None:
            y_pos = y_max + 0.2 * y_max  # Adjust height for the asterisk
        ax.text(x_point, y_pos, significance, ha='center', va='bottom', color=color, fontsize=12)

    # Optionally, adjust the y-axis to make sure the asterisks are visible
    # ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] * 1.1])


def plot(what, fontsize=12):
    match what:

        # region FORCE:average
        case 'FORCE:average':

            experiment = 'efc2'

            chord = 'trained'

            force = main('FORCE:average', experiment, day='1', chord=chord)

            fig, axs = plt.subplots(1, len(force), sharex=True, sharey='row', figsize=(15, 5))

            color = [
                (0.031, 0.188, 0.419),  # Dark Blue
                (0.129, 0.443, 0.710),  # Medium Blue
                (0.258, 0.573, 0.816),  # Sky Blue
                (0.454, 0.678, 0.819),  # Light Sky Blue
                (0.671, 0.851, 0.914)  # Pale Blue
            ]  # Pale Violet]

            for k, (key, f) in enumerate(force.items()):

                f = np.stack(f)
                f = np.abs(f)

                tAx = np.linspace(0, .8, f.shape[-1])

                f_avg = f.mean(axis=0).squeeze().T
                f_err = f.std(axis=0).squeeze().T / np.sqrt(f.shape[1])

                for i, char in enumerate(str(key)):
                    if char == '9':
                        axs[k,].plot(tAx, f_avg[:, i], color=color[i], lw=1, label=gl.channels['force'][i], ls='-')
                    elif char == '1':
                        axs[k,].plot(tAx, f_avg[:, i], color=color[i], lw=3, label=gl.channels['force'][i], ls='-')
                    else:
                        axs[k,].plot(tAx, f_avg[:, i], color=color[i], lw=3, label=gl.channels['force'][i], ls='-')
                    axs[k,].fill_between(tAx, f_avg[:, i] + f_err[:, i], f_avg[:, i] - f_err[:, i], lw=0,
                                         color=color[i],
                                         alpha=0.2)

                axs[k].set_title(f'chord:{key}', fontsize=fontsize)

                axs[k].axhspan(0, 1.2, color='grey', alpha=0.3, lw=0)

                axs[k].axhline(2, color='k', lw=2, ls='--')

                # axs[1].spines['bottom'].set_bounds(1, 5)
                axs[k,].set_xlim([0, .8])
                axs[k,].spines[['right', 'top', 'left']].set_visible(False)
                axs[k,].spines[['bottom']].set_linewidth(2)
                axs[k,].tick_params(axis='x', width=2)
                axs[k,].tick_params(axis='y', width=0)
                axs[k,].set_xticks([axs[0].get_xlim()[0], axs[0].get_xlim()[1] / 2, axs[0,].get_xlim()[1]])
                # axs[k].set_xticklabels(axs[1].get_xticklabels(), fontsize=fontsize)
                axs[k].set_xlabel('')

            custom_handles = [Line2D([0], [0], color=color, label=gl.channels['force'][c], lw=3) for c, color in
                              enumerate(color)]
            axs[0].legend(handles=custom_handles, frameon=False, loc='upper left')
            axs[0].spines[['left']].set_visible(True)
            axs[0].spines[['left']].set_linewidth(2)
            axs[0].spines[['left']].set_bounds(0, 3.5)
            axs[0].tick_params(axis='y', width=2)

            axs[0].text(0, 1.2, 'baseline', color='grey', ha='left', va='bottom')
            axs[0].text(0, 2, 'target', color='k', ha='left', va='bottom')

            fig.supxlabel('time (s)', fontsize=fontsize)
            fig.supylabel('force (N)', fontsize=fontsize)
            fig.suptitle('Force response in trained chords on day 5')
            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.force_avg4.svg'), fig)

        # endregion

# <<<<<<< Updated upstream
#         # region FORCE:finger_times
# =======
        # region FORCE:derivative
        case 'FORCE:derivative':

            experiment = 'efc2'

            chord = 'trained'

            dforce, dforce_norm = main('FORCE:derivative', experiment, day='1', chord=chord)

            fig, axs = plt.subplots(1, len(dforce), sharex=True, sharey=True, figsize=(15, 5))

            color = [(0.031, 0.188, 0.419),  # Dark Blue
                     (0.129, 0.443, 0.710),  # Medium Blue
                     (0.258, 0.573, 0.816),  # Sky Blue
                     (0.454, 0.678, 0.819),  # Light Sky Blue
                     (0.671, 0.851, 0.914)]  # Pale Blue

            for k, (key, f) in enumerate(dforce_norm.items()):

                f = np.stack(f)
                f = np.abs(f)

                tAx = np.linspace(0, .8, f.shape[-1])

                f_avg = f.mean(axis=0).squeeze().T
                f_err = f.std(axis=0).squeeze().T / np.sqrt(f.shape[1])

                for i, char in enumerate(str(key)):
                    if char == '9':
                        axs[k].plot(tAx, f_avg[:, i], color=color[i], lw=1, label=gl.channels['force'][i], ls='-')
                    elif char == '1':
                        axs[k].plot(tAx, f_avg[:, i], color=color[i], lw=3, label=gl.channels['force'][i], ls='-')
                    else:
                        axs[k].plot(tAx, f_avg[:, i], color=color[i], lw=3, label=gl.channels['force'][i], ls='-')
                    axs[k].fill_between(tAx, f_avg[:, i] + f_err[:, i], f_avg[:, i] - f_err[:, i], lw=0, color=color[i],
                                        alpha=0.2)

                axs[k].set_title(f'chord:{key}', fontsize=fontsize)

                # axs[1].spines['bottom'].set_bounds(1, 5)
                axs[k].set_xlim([.17, .33])
                axs[k].spines[['right', 'top', 'left']].set_visible(False)
                axs[k].spines[['bottom']].set_linewidth(2)
                axs[k].tick_params(axis='x', width=2)
                axs[k].tick_params(axis='y', width=0)
                # axs[k].set_xticks([axs[k].get_xlim()[0], axs[k].get_xlim()[1] / 2, axs[k].get_xlim()[1]])
                # axs[k].set_xticklabels(axs[1].get_xticklabels(), fontsize=fontsize)
                axs[k].set_xlabel('')

            custom_handles = [Line2D([0], [0], color=color, label=gl.channels['force'][c], lw=3) for c, color in
                              enumerate(color)]
            axs[0].legend(handles=custom_handles, frameon=False, loc='upper left')
            axs[0].spines[['left']].set_visible(True)
            axs[0].spines[['left']].set_linewidth(2)
            axs[0].spines[['left']].set_bounds(0, .5)
            axs[0].tick_params(axis='y', width=2)

            fig.supxlabel('time (s)', fontsize=fontsize)
            fig.supylabel('force (N)', fontsize=fontsize)
            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.force_der.svg'), fig)

        # endregion

        # region FORCE:finger_times

        case 'FORCE:finger_times':

            experiment = 'efc2'

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')
            metrics = metrics[metrics['trialPoint'] == 1]
            metrics_exit = pd.melt(metrics,
                                   id_vars=['day', 'chord', 'chordID', 'participant_id'],
                                   value_vars=['thumb_exit', 'index_exit', 'middle_exit', 'ring_exit', 'pinkie_exit'],
                                   var_name='finger',
                                   value_name='time')

            # Second group of value_vars (e.g., another measurement like 'offset' for each finger)
            metrics_onset = pd.melt(metrics,
                                    id_vars=['day', 'chord', 'chordID', 'participant_id'],
                                    value_vars=['thumb_onset', 'index_onset', 'middle_onset', 'ring_onset', 'pinkie_onset'],
                                    var_name='finger',
                                    value_name='time')
            metrics_onset['timepoint'] = 'onset'
            metrics_exit['timepoint'] = 'exit'

            metrics = pd.concat([metrics_onset, metrics_exit], axis=0)
            metrics['finger'] = metrics['finger'].str.replace('_exit', '', regex=False)
            metrics['finger'] = metrics['finger'].str.replace('_onset', '', regex=False)

            metrics = metrics.groupby(['chord', 'participant_id', 'day', 'finger', 'timepoint']).mean(
                numeric_only='True').reset_index()

            custom_order_exit = ['thumb', 'index', 'middle', 'ring', 'pinkie']
            metrics['finger'] = pd.Categorical(metrics['finger'], categories=custom_order_exit, ordered=True).codes
            metrics['chordID'] = metrics['chordID'].astype(str)
            jitter = 0.5 * (np.random.rand(len(metrics)) - 0.5)
            metrics['finger_jittered'] = metrics['finger'] + jitter

            offset = 5
            metrics['offset'] = metrics['finger'] + metrics['day'] * offset

            fig, axs = plt.subplots()

            sns.lineplot(data=metrics[(metrics['chord'] == 'trained') & (metrics['timepoint'] == 'exit')], ax=axs, x='offset',
                         y='time', hue='day', palette=['red'] * 5, err_kws={'linewidth': 0}, lw=3, legend=False)
            sns.lineplot(data=metrics[(metrics['chord'] == 'untrained') & (metrics['timepoint'] == 'exit')], ax=axs, x='offset',
                         y='time', hue='day', palette=['blue'] * 5, err_kws={'linewidth': 0}, lw=3, legend=False)
            sns.lineplot(data=metrics[(metrics['chord'] == 'trained') & (metrics['timepoint'] == 'onset')], ax=axs, x='offset',
                         y='time', hue='day', palette=[(1, .5, .5)] * 5, err_kws={'linewidth': 0}, lw=3, legend=False)
            sns.lineplot(data=metrics[(metrics['chord'] == 'untrained') & (metrics['timepoint'] == 'onset')], ax=axs,
                         x='offset', y='time', hue='day', palette=[(.5, .5, 1)] * 5, err_kws={'linewidth': 0}, lw=3,
                         legend=False)

            axs.set_xticks(np.linspace(7, 27, 5))
            axs.set_xticklabels(np.linspace(1, 5, 5, dtype=int), fontsize=fontsize)

            custom_handles = [
                Line2D([0], [0], color='blue', label='untrained, exit baseline', lw=3),
                Line2D([0], [0], color='red', label='trained, exit baseline', lw=3),
                Line2D([0], [0], color=(.5, .5, 1), label='untrained, onset', lw=3),
                Line2D([0], [0], color=(1, .5, .5), label='trained, onset', lw=3),
            ]

            axs.legend(handles=custom_handles, frameon=False, fontsize=fontsize, loc='upper right')

            axs.set_ylabel('time (s)', fontsize=fontsize)

            axs.set_xlabel('day', fontsize=fontsize)

            # axs.set_xticks([0, 1, 2, 3, 4])
            # axs.set_xticklabels(gl.channels['force'], rotation=45, ha='right', fontsize=fontsize)

            axs.set_title(f'Time of force onset and exit from baseline area', fontsize=fontsize)

            decor(axs=axs, fontsize=fontsize, ybounds=(.3, 1), xbounds=(7, 27), spines_width=2)

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.force_times2.svg'), fig)

        # endregion

        # region FORCE:asynchrony
        case 'ASYNCHRONY:day':

            experiment = 'efc2'

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            metrics['finger_asynch'] = metrics['finger_asynch']  # + 1

            metrics_grouped = metrics.groupby(['chord', 'participant_id', 'day']).mean(numeric_only='True').reset_index()

            fig, axs = plt.subplots(figsize=(3.5, 5))

            dodge = .2

            metrics_grouped['dday'] = metrics_grouped['day'] + metrics_grouped['chord'].map(
                {'trained': -dodge, 'untrained': dodge}) - 1

            # sns.boxplot(data=metrics, ax=axs, y='finger_asynch', x='day', hue='chord', palette=[(1, .8, .8), (.8, .8, 1)],
            #             showfliers=False, legend=False)
            sns.violinplot(data=metrics, ax=axs, y='finger_asynch', x='day', hue='chord', palette=['red', 'blue'], split=True,
                           inner=None, linewidth=0, dodge=True, alpha=.2, legend=False, cut=0)
            sns.lineplot(data=metrics_grouped, ax=axs, y='finger_asynch', x='dday', hue='chord', palette=['red', 'blue'],
                         marker='o', markeredgecolor='none',
                         lw=3, err_kws={'linewidth': 0, 'alpha': .2}, errorbar='se', zorder=100)

            # y_pos = add_significance_brackets(axs, metrics[metrics['chord'] == 'trained'], x='day', y='finger_asynch', pairs=[(1, 5)], test_type='t-test_rel', x1_pos=.9, x2_pos=4.9, significance_level=.05 / 3)
            # add_significance_brackets(axs, metrics[metrics['chord'] == 'untrained'], x='day', y='finger_asynch', pairs=[(1, 5)], test_type='t-test_rel', x1_pos=1.1, x2_pos=5.1, y_pos=y_pos, significance_level=.05 / 3)
            # add_significance_asterisks(axs, metrics, x='day', y='finger_asynch', hue='chord', x_point=5, test_type='t-test_rel',
            #                                     significance_level=.05 / 3, text_format='star', color='k', y_pos=.45)

            axs.set_title('Finger asynchrony\nat force onset', fontsize=fontsize)
            axs.set_ylabel(r"time (s)", fontsize=fontsize)
            axs.set_xlabel(r"day", fontsize=fontsize)

            axs.set_yscale('symlog', linthresh=.5)

            axs.set_yticks([0, .1, 1, 4])
            axs.set_yticklabels([0, .1, 1, 4])

            decor(axs=axs, fontsize=fontsize,  ybounds=(-0., 4), xbounds=(0, 4), spines_width=2)

            axs.legend(ncol=1, frameon=False, fontsize=fontsize, loc='best')

            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.asynchrony.svg'), fig)


        # endregion

        # region ORDER:rank_corr
        case 'ORDER:rank_corr':

            experiment = 'efc2'

            rank_corr = pd.read_csv(os.path.join(gl.baseDir, experiment, 'rank_corr.tsv'), sep='\t')

            rank_corr = rank_corr.groupby(['chord', 'participant_id', 'day', 'repetition']).mean(
                numeric_only=True).reset_index()

            fig, axs = plt.subplots(figsize=(5, 5))

            offset = 5
            rank_corr['offset'] = rank_corr['repetition'] + rank_corr['day'] * offset

            sns.lineplot(data=rank_corr[rank_corr['chord'] == 'trained'], ax=axs, x='offset', y='exit', hue='day',
                         errorbar='se', lw=3,
                         palette=['red'] * 5, marker='o', markeredgewidth=0, err_kws={'linewidth': 0}, legend=False)
            sns.lineplot(data=rank_corr[rank_corr['chord'] == 'untrained'], ax=axs, x='offset', y='exit', hue='day',
                         errorbar='se', lw=3, legend=False,
                         palette=['blue'] * 2, marker='o', markeredgewidth=0, err_kws={'linewidth': 0})
            sns.lineplot(data=rank_corr[rank_corr['chord'] == 'trained'], ax=axs, x='offset', y='onset', hue='day',
                         errorbar='se', lw=3, palette=[(1, .5, .5)] * 5, marker='o', markeredgewidth=0,
                         err_kws={'linewidth': 0}, legend=False)
            sns.lineplot(data=rank_corr[rank_corr['chord'] == 'untrained'], ax=axs, x='offset', y='onset', hue='day',
                         errorbar='se', lw=3, palette=[(.5, .5, 1)] * 2, marker='o', markeredgewidth=0,
                         err_kws={'linewidth': 0}, legend=False)
            axs.set_xticks(np.linspace(8, 28, 5))
            axs.set_xticklabels(np.linspace(1, 5, 5, dtype=int))

            custom_handles = [
                Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', label='untrained, exit baseline', lw=3),
                Line2D([0], [0], marker='o', color='red', markerfacecolor='red', label='trained, exit baseline', lw=3),
                Line2D([0], [0], marker='o', color=(.5, .5, 1), markerfacecolor=(.5, .5, 1), label='untrained, onset', lw=3),
                Line2D([0], [0], marker='o', color=(1, .5, .5), markerfacecolor=(1, .5, .5), label='trained, onset', lw=3)
            ]

            fig.legend(handles=custom_handles, ncol=1, frameon=False, fontsize=12, loc='upper left')

            axs.set_title("Consistency of finger order at force onset and baseline exit", fontsize=fontsize)
            axs.set_ylabel(r"correlation (Spearman's $\rho$)", fontsize=fontsize)
            axs.set_xlabel('day', fontsize=fontsize)

            decor(axs=axs, fontsize=fontsize, ybounds=(.2, .55), xbounds=(8, 28), spines_width=2)

            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.rank_corr.svg'), fig)

        # endregion

# <<<<<<< Updated upstream
        # region MEAN_DEVIATION:day
        case 'MEAN_DEVIATION:day':

            experiment = 'efc2'

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            metrics = metrics.groupby(['chord', 'participant_id', 'day']).mean(numeric_only='True').reset_index()

            fig, axs = plt.subplots(figsize=(3.5, 5))

            dodge = 0.1

            metrics['dday'] = metrics['day'] + metrics['chord'].map({'trained': -dodge, 'untrained': dodge})

            sns.lineplot(data=metrics, ax=axs, y='MD', x='day', hue='chord', palette=['red', 'blue'], marker='o',
                         markeredgecolor='none',
                         lw=3, err_kws={'linewidth': 0}, errorbar='se')

            # y_pos = add_significance_brackets(axs, metrics[metrics['chord'] == 'trained'], x='day', y='MD', pairs=[(1, 5)],
            #                                   test_type='t-test_rel', x1_pos=.9, x2_pos=4.9, significance_level=.05 / 3)
            # add_significance_brackets(axs, metrics[metrics['chord'] == 'untrained'], x='day', y='MD', pairs=[(1, 5)],
            #                           test_type='t-test_rel', x1_pos=1.1, x2_pos=5.1, y_pos=y_pos, significance_level=.05 / 3)
            # add_significance_asterisks(axs, metrics, x='day', y='MD', hue='chord', x_point=5, test_type='t-test_rel',
            #                            significance_level=.05 / 3, text_format='star', color='k', y_pos=1.55)

            decor(axs=axs, fontsize=fontsize, ybounds=(.8, 1.6), xbounds=(1, 5), spines_width=2)

            axs.set_title('Mean deviation', fontsize=fontsize)
            axs.set_ylabel('mean deviation (N)', fontsize=fontsize)

            axs.legend(ncol=1, frameon=False, fontsize=fontsize, loc='lower left')

            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.mean_deviation_day.svg'), fig)

        # endregion

        # region EXECUTION_TIME:day
        case 'EXECUTION_TIME:day':

            experiment = 'efc2'

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            metrics = metrics.groupby(['chord', 'participant_id', 'day']).mean(numeric_only='True').reset_index()

            fig, axs = plt.subplots(figsize=(3.5, 5))

            dodge = 0.1

            metrics['dday'] = metrics['day'] + metrics['chord'].map({'trained': -dodge, 'untrained': dodge})

            sns.lineplot(data=metrics, ax=axs, y='ET', x='day', hue='chord', palette=['red', 'blue'], marker='o',
                         markeredgecolor='none',
                         lw=3, err_kws={'linewidth': 0}, errorbar='se')

            # y_pos = add_significance_brackets(axs, metrics[metrics['chord'] == 'trained'], x='day', y='ET',
            #                                   pairs=[(1, 5)], test_type='t-test_rel', x1_pos=.9, x2_pos=4.9,
            #                                   significance_level=.05 / 3)
            # add_significance_brackets(axs, metrics[metrics['chord'] == 'untrained'], x='day', y='ET', pairs=[(1, 5)],
            #                           test_type='t-test_rel', x1_pos=1.1, x2_pos=5.1, y_pos=y_pos,
            #                           significance_level=.05 / 3)
            # add_significance_asterisks(axs, metrics, x='day', y='ET', hue='chord', x_point=5, test_type='t-test_rel',
            #                            significance_level=.05 / 3, text_format='star', color='k', y_pos=2)

            decor(axs=axs, fontsize=fontsize, ybounds=(.5, 2.5), xbounds=(1, 5), spines_width=2)

            axs.set_title('Execution time', fontsize=fontsize)
            axs.set_ylabel('execution time (s)', fontsize=fontsize)

            axs.legend(ncol=1, frameon=False, fontsize=fontsize, loc='lower left')

            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.execution_time_day.svg'), fig)

        # endregion

        # region CONSISTENCY:day
        case 'CONSISTENCY:day':

            experiment = 'efc2'

            rank_corr = pd.read_csv(os.path.join(gl.baseDir, experiment, 'rank_corr.tsv'), sep='\t')

            rank_corr = rank_corr.groupby(['chord', 'participant_id', 'day']).mean(numeric_only='True').reset_index()

            fig, axs = plt.subplots(figsize=(3.5, 5))

            dodge = 0.1

            rank_corr['dday'] = rank_corr['day'] + rank_corr['chord'].map({'trained': -dodge, 'untrained': dodge})

            sns.lineplot(data=rank_corr, ax=axs, y='onset', x='day', hue='chord', palette=['red', 'blue'], marker='o',
                         markeredgecolor='none',
                         lw=3, err_kws={'linewidth': 0}, errorbar='se')

            # y_pos = add_significance_brackets(axs, rank_corr[rank_corr['chord'] == 'trained'], x='day', y='onset',
            #                                   pairs=[(1, 5)], test_type='t-test_rel', x1_pos=.9, x2_pos=4.9,
            #                                   significance_level=.05 / 3)
            # # add_significance_brackets(axs, rank_corr[rank_corr['chord'] == 'untrained'], x='day', y='onset', pairs=[(1, 5)], test_type='t-test_rel', x1_pos=1.1, x2_pos=5.1, y_pos=y_pos, significance_level=.05 / 3)
            # add_significance_asterisks(axs, rank_corr, x='day', y='onset', hue='chord', x_point=5,
            #                            test_type='t-test_rel',
            #                            significance_level=.05 / 3, text_format='star', color='k', y_pos=.45)

            decor(axs=axs, fontsize=fontsize, ybounds=(.25, .45), xbounds=(1, 5), spines_width=2)

            axs.set_title('Finger order consistency', fontsize=fontsize)
            axs.set_ylabel(r"correlation (Spearman's $\rho$)", fontsize=fontsize)
            axs.set_xlabel("day", fontsize=fontsize)

            axs.legend(ncol=1, frameon=False, fontsize=fontsize, loc='best')

            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.consistency_day.svg'), fig)

        # endregion

        # region VAR_DEC:day
        case 'VAR_DEC:day':

            experiment = 'efc2'

            # Load and process the data
            var_dec_order = pd.read_csv(os.path.join(gl.baseDir, experiment, 'var_dec_order.tsv'), sep='\t')
            var_dec_order = var_dec_order.groupby(['chord', 'day', 'chordID']).mean(numeric_only=True).reset_index()

            # Apply a small offset to 'day' directly using np.where
            var_dec_order['dday'] = np.where(var_dec_order['chord'] == 'trained', var_dec_order['day'] - 0.1,
                                                   var_dec_order['day'] + 0.1)

            # Create subplots
            fig, axs = plt.subplots(1, 3, figsize=(6, 5), sharey=True)

            # Common plot settings
            lineplot_kwargs = {'palette': ['red', 'blue'], 'marker': 'o', 'markeredgecolor': 'none', 'lw': 3,
                               'err_kws': {'linewidth': 0}}

            # Plot each onset type with dodged x values
            sns.lineplot(data=var_dec_order, ax=axs[0], y='v_g_onset', x='dday', hue='chord', **lineplot_kwargs,
                         legend=False)
            sns.lineplot(data=var_dec_order, ax=axs[1], y='v_s_onset', x='dday', hue='chord', **lineplot_kwargs,
                         legend=False)
            sns.lineplot(data=var_dec_order, ax=axs[2], y='v_e_onset', x='dday', hue='chord', **lineplot_kwargs)

            # Custom handles for the legend
            custom_handles = [
                Line2D([0], [0], marker='o', color='red', markerfacecolor='red', label='trained', lw=3),
                Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', label='untrained', lw=3),
            ]

            # Apply decor and customizations to each subplot
            for ax, title in zip(axs, ['Chord', 'Subject', 'Error']):
                ax.spines[['right', 'top', 'left']].set_visible(False)
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['bottom'].set_bounds([1, 5])
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.set_xticklabels([1, 2, 3, 4, 5], fontsize=fontsize)
                ax.tick_params(axis='x', width=2)
                ax.set_title(title, fontsize=fontsize)
                ax.set_xlabel('')
                if title != 'Chord':
                    ax.tick_params(axis='y', width=0)

            # Additional customizations for the first subplot
            axs[0].set_ylabel('fraction of variance', fontsize=fontsize)
            axs[0].spines[['left']].set_visible(True)

            y_pos = add_significance_brackets(axs[0], var_dec_order[var_dec_order['chord'] == 'trained'], x='dday',
                                      y='v_g_onset', pairs=[(.9, 4.9)], significance_level=.05 / 3,x1_pos=.9, x2_pos=4.9)
            add_significance_brackets(axs[0], var_dec_order[var_dec_order['chord'] == 'untrained'], x='dday',
                                      y='v_g_onset', pairs=[(1.1, 5.1)], significance_level=.05 / 3,x1_pos=1.1, x2_pos=5.1, y_pos=y_pos)
            add_significance_asterisks(axs[0], var_dec_order, x='day', y='v_g_onset', hue='chord', x_point=5,
                                       test_type='t-test_rel',
                                       significance_level=.05 / 3, text_format='star', color='k', y_pos=.45)

            y_pos = add_significance_brackets(axs[1], var_dec_order[var_dec_order['chord'] == 'trained'], x='dday',
                                              y='v_s_onset', pairs=[(.9, 4.9)], significance_level=.05 / 3,x1_pos=.9, x2_pos=4.9)
            add_significance_brackets(axs[1], var_dec_order[var_dec_order['chord'] == 'untrained'], x='dday',
                                      y='v_s_onset', pairs=[(1.1, 5.1)], significance_level=.05 / 3,x1_pos=1.1, x2_pos=5.1, y_pos=y_pos)
            add_significance_asterisks(axs[1], var_dec_order, x='day', y='v_s_onset', hue='chord', x_point=5,
                                       test_type='t-test_rel',
                                       significance_level=.05 / 3, text_format='star', color='k', y_pos=.45)

            y_pos = add_significance_brackets(axs[2], var_dec_order[var_dec_order['chord'] == 'trained'], x='dday',
                                              y='v_e_onset', pairs=[(.9, 4.9)],significance_level=.05 / 3, x1_pos=.9, x2_pos=4.9)
            # add_significance_brackets(axs[2], var_dec_order[var_dec_order['chord'] == 'untrained'], x='dday',
            #                           y='v_e_onset', pairs=[(1.1, 5.1)],significance_level=.05 / 3, x1_pos=1.1, x2_pos=5.1, y_pos=y_pos)
            add_significance_asterisks(axs[2], var_dec_order, x='day', y='v_e_onset', hue='chord', x_point=5,
                                       test_type='t-test_rel',
                                       significance_level=.05 / 3, text_format='star', color='k', y_pos=.45)


            decor(axs=axs[0], fontsize=fontsize, ybounds=(.1, .8), xbounds=(1, 5), spines_width=2)

            # Add the legend and labels
            axs[2].legend(handles=custom_handles, frameon=False, fontsize=fontsize)
            fig.supxlabel('day', fontsize=fontsize)
            fig.suptitle('Variance decomposition of finger order', fontsize=fontsize)

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.var_dec_day.svg'), fig)

        # endregion

        # region EMG:correlation
        case 'EMG:correlation':

            experiment = 'efc3'
            participant_id = 'subj114'

            with open(os.path.join(gl.baseDir, experiment, 'correlations.pkl'), 'rb') as f:
                correlation = pickle.load(f)
            df_correlation = pd.read_csv(os.path.join(gl.baseDir, experiment, 'correlations.tsv'), sep='\t')

            pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

            trained = pinfo[pinfo['participant_id'] == participant_id].reset_index()['trained'].iloc[0].split('.')
            untrained = pinfo[pinfo['participant_id'] == participant_id].reset_index()['untrained'].iloc[0].split('.')

            # Use `+` to concatenate trained and untrained lists
            chords = trained + untrained

            # Red and Blue shades
            colors = [
                # Reds
                (1.0, 0.756, 0.756),  # Light Red
                (1.0, 0.451, 0.451),  # Soft Red
                (1.0, 0.2, 0.2),  # Medium Red
                (0.698, 0.0, 0.0),  # Dark Red
                # Blues
                (0.678, 0.847, 0.902),  # Light Blue
                (0.529, 0.808, 0.922),  # Sky Blue
                (0.275, 0.510, 0.706),  # Medium Blue
                (0.0, 0.0, 0.545)  # Dark Blue
            ]

            # Create a color map matching each chord to a color
            color_map = {chord: color for chord, color in zip(chords, colors)}

            fig, axs = plt.subplots(1, 3, sharey=True)

            for c, corr in enumerate(correlation['corr']):
                if correlation['participant_id'][c] == participant_id:
                    corr = sorted(corr, reverse=True)

                    df_corr_tmp = df_correlation[(df_correlation['chordID'] == correlation['chordID'][c]) & (
                                df_correlation['participant_id'] == participant_id)]

                    if correlation['day'][c] == '1':
                        ax = axs[0]
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.tick_params(axis='x', width=2, labelsize=fontsize)  # Set thickness and length of x-axis ticks
                        ax.tick_params(axis='y', width=2, labelsize=fontsize)  # Set thickness and length of y-axis ticks
                        ax.spines['bottom'].set_linewidth(2)
                        ax.spines['left'].set_linewidth(2)
                        ax.spines['bottom'].set_bounds(1, len(corr))
                        ax.spines['left'].set_bounds(-.6, .8)
                        ax.set_xticks([1, len(corr)])
                    else:
                        ax = axs[1]
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)

                        ax.tick_params(axis='x', width=2, labelsize=fontsize)  # Set thickness and length of x-axis ticks
                        ax.tick_params(axis='y', width=0, labelsize=fontsize)  # Set thickness and length of y-axis ticks
                        # Set spine thickness (only for bottom spine as left/right/top are hidden)
                        ax.spines['bottom'].set_linewidth(2)
                        ax.spines['bottom'].set_bounds(1, len(corr))
                        ax.set_xticks([1, len(corr)])

                    ax.plot(corr, color=color_map[str(correlation['chordID'][c])], lw=3, label=str(correlation['chordID'][c]))
                    ax.set_title(f"Day {correlation['day'][c]}", fontsize=fontsize)
                    ax.axhline(0, color='k')

                    # sns.lineplot(df_corr_tmp, ax=axs[2], y='corr', x='day', color=color_map[str(correlation['chordID'][c])], marker='o', markeredgecolor='none', alpha=.3, lw=3)

            axs[0].set_ylabel('correlation', fontsize=fontsize)
            axs[0].legend(frameon=False, loc='best')

            # axs[1].set_yticks([])

            df_correlation = df_correlation[df_correlation['participant_id'] == participant_id]

            sns.lineplot(df_correlation, ax=axs[2], y='corr', x='day', hue='chord', hue_order=['trained', 'untrained'],
                         palette=['red', 'blue'], marker='o', markeredgecolor='none', lw=3, err_kws={'lw': 0})
            axs[2].legend(frameon=False)
            axs[2].axhline(0, color='k', lw=2)

            # For the leftmost axis (axs[0]), keep only the left spine
            axs[0].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)

            axs[2].spines['top'].set_visible(False)
            axs[2].spines['right'].set_visible(False)
            axs[2].spines['left'].set_visible(False)

            axs[2].tick_params(axis='x', width=2, labelsize=fontsize)  # Set thickness and length of x-axis ticks
            axs[2].tick_params(axis='y', width=0, labelsize=fontsize)  # Set thickness and length of y-axis ticks
            # Set spine thickness (only for bottom spine as left/right/top are hidden)
            axs[2].spines['bottom'].set_linewidth(2)
            axs[2].spines['bottom'].set_bounds(1, 5)
            axs[2].set_xticks([1, 5])
            axs[2].set_xlabel('day', fontsize=fontsize)

            axs[0].set_xlabel('No. TMS stimuli', fontsize=fontsize)

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.emg_correlation2.svg'), fig)

        # endregion

        # region EXAMPLE:force_derivative
        case 'EXAMPLE:force_derivative':

            experiment = 'efc2'

            force = Force(experiment, 'subj100', 'testing', '5')
            force = force.load_pkl() #['force_filt10Hz'][4]

            chordID = int(force['chordID'][4])
            force = force['force_filt10Hz'][4]

            dforce = np.gradient(force, 1 / gl.fsample['force'], axis=0)

            scaler = MinMaxScaler()

            dforce = scaler.fit_transform(np.abs(dforce))

            fig, axs = plt.subplots(2, sharex=True, figsize=(3, 5))

            color = [
                (0.031, 0.188, 0.419),  # Dark Blue
                (0.129, 0.443, 0.710),  # Medium Blue
                (0.258, 0.573, 0.816),  # Sky Blue
                (0.454, 0.678, 0.819),  # Light Sky Blue
                (0.671, 0.851, 0.914)  # Pale Blue
            ]  # Pale Violet]

            tAx = np.linspace(0, dforce.shape[0] / gl.fsample['force'], force.shape[0])

            for i, char in enumerate(str(chordID)):
                if char == '9':
                    axs[0,].plot(tAx, np.abs(force[:, i]), color=color[i], lw=1, label=gl.channels['force'][i], ls='-')
                elif char == '1':
                    axs[0,].plot(tAx, np.abs(force[:, i]), color=color[i], lw=3, label=gl.channels['force'][i], ls='-')
                else:
                    axs[0,].plot(tAx, np.abs(force[:, i]), color=color[i], lw=3, label=gl.channels['force'][i], ls='-')

            for i, char in enumerate(str(chordID)):
                if char == '9':
                    axs[1,].plot(tAx, dforce[:, i], color=color[i], lw=1, label=gl.channels['force'][i], ls='-')
                elif char == '1':
                    axs[1,].plot(tAx, dforce[:, i], color=color[i], lw=3, label=gl.channels['force'][i], ls='-')
                else:
                    axs[1,].plot(tAx, dforce[:, i], color=color[i], lw=3, label=gl.channels['force'][i], ls='-')

            axs[0].legend(frameon=False)

            axs[0].spines[['right', 'top', 'bottom']].set_visible(False)
            axs[1].spines[['right', 'top',]].set_visible(False)
            axs[1].spines['bottom'].set_linewidth(2)
            axs[0].spines['left'].set_linewidth(2)
            axs[1].spines['left'].set_linewidth(2)
            axs[1].spines['bottom'].set_bounds([0, .8])
            axs[0].set_yticks([0, 1, 2, 3, 4])
            axs[1].set_yticks([.0, .2, .5, 1])
            axs[0].spines['left'].set_bounds([0, 4])
            axs[1].spines['left'].set_bounds([0, 1])
            axs[1].set_xticks([0, .4, .8])
            axs[1].set_xticklabels([0, 0.5, 1], fontsize=fontsize)
            axs[1].tick_params(axis='x', width=2)
            axs[0].tick_params(axis='x', width=0)
            axs[1].tick_params(axis='y', width=2, labelsize=fontsize)
            axs[0].tick_params(axis='y', width=2, labelsize=fontsize)
            # axs[0].set_title(title, fontsize=fontsize)
            axs[1].set_xlabel('time (s)', fontsize=fontsize)
            axs[0].set_ylabel('force (N)', fontsize=fontsize)
            axs[1].set_ylabel('normalized\nforce derivative (a.u.)', fontsize=fontsize)

            axs[1].axhline(.2, color='k', ls='--')

            axs[1].set_xlim([0, .8])

            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.example_derivative.svg'), fig)
        # endregion


# =======
        # region ORDER:correlation_between_days
        case 'ORDER:correlation_between_days':

            experiment = 'efc2'

            corr = np.load(os.path.join(gl.baseDir, experiment, 'order_day_corr.npy'))
            corr = np.nanmean(corr, axis=(0, 1))

            days = [f'day{day}' for day in np.linspace(1, 5, 5, dtype=int)]

            fig, axs = plt.subplots(figsize=(3, 4.5))

            cax = axs.imshow(corr, vmin=0, vmax=.25)

            # Add day labels to the axes
            axs.set_xticks(np.arange(len(days)))
            axs.set_yticks(np.arange(len(days)))
            axs.set_xticklabels(days, fontsize=fontsize, rotation=45, ha='right')
            axs.set_yticklabels(days, fontsize=fontsize, rotation=45, ha='right')

            # Add a color bar to represent the values
            cb = fig.colorbar(cax, orientation='horizontal', ticks=[0, .25])
            cb.set_label(r"correlation (Spearman's $\rho$)", fontsize=fontsize)

            cb.ax.tick_params(labelsize=fontsize)

            fig.suptitle('Finger order correlation\nbetween days', fontsize=fontsize)

            fig.tight_layout()

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.corr_between_days.svg'), fig)

        # endregion

        # region ORDER:variance_decomposition
        case 'ORDER:variance_decomposition':

            experiment = 'efc2'

            var_dec_order = pd.read_csv(os.path.join(gl.baseDir, experiment, 'var_dec_order.tsv'), sep='\t')

            var_dec_order = var_dec_order.groupby(['chord', 'day', 'chordID']).mean(numeric_only='True').reset_index()

            fig, axs = plt.subplots(1, 3, figsize=(6, 5), sharey=True)

            # sns.lineplot(data=var_dec_order, ax=axs[0], y='v_g_exit', x='day', hue='chord', palette=['red', 'blue'], marker='o',
            #              markeredgecolor='none',
            #              lw=3, err_kws={'linewidth': 0}, legend=False)
            # sns.lineplot(data=var_dec_order, ax=axs[1], y='v_s_exit', x='day', hue='chord', palette=['red', 'blue'], marker='o',
            #              markeredgecolor='none',
            #              lw=3, err_kws={'linewidth': 0}, legend=False)
            # sns.lineplot(data=var_dec_order, ax=axs[2], y='v_e_exit', x='day', hue='chord', palette=['red', 'blue'], marker='o',
            #              markeredgecolor='none',
            #              lw=3, err_kws={'linewidth': 0})

            sns.lineplot(data=var_dec_order, ax=axs[0], y='v_g_onset', x='day', hue='chord', palette=['red', 'blue'],
                         marker='o', markeredgecolor='none',
                         lw=3, err_kws={'linewidth': 0}, legend=False)
            sns.lineplot(data=var_dec_order, ax=axs[1], y='v_s_onset', x='day', hue='chord', palette=['red', 'blue'],
                         marker='o', markeredgecolor='none',
                         lw=3, err_kws={'linewidth': 0}, legend=False)
            sns.lineplot(data=var_dec_order, ax=axs[2], y='v_e_onset', x='day', hue='chord', palette=['red', 'blue'],
                         marker='o', markeredgecolor='none',
                         lw=3, err_kws={'linewidth': 0})

            custom_handles = [
                Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', label='untrained', lw=3),
                Line2D([0], [0], marker='o', color='red', markerfacecolor='red', label='trained', lw=3),
                # Line2D([0], [0], marker='o', color=(.5, .5, 1), markerfacecolor=(.5, .5, 1), label='untrained, onset', lw=3),
                # Line2D([0], [0], marker='o', color=(1, .5, .5), markerfacecolor=(1, .5, .5), label='trained, onset', lw=3)
            ]

            decor(axs=axs[0], fontsize=fontsize, ybounds=(.1, .7), xbounds=(1, 5), spines_width=2)

            axs[0].set_ylabel('% variance', fontsize=fontsize)
            axs[0].set_xlabel('')
            axs[0].set_title('Chord', fontsize=fontsize)
            axs[0].set_xticks([1, 2, 3, 4, 5])
            axs[0].set_xticklabels([1, 2, 3, 4, 5], fontsize=fontsize)

            axs[1].spines['bottom'].set_bounds(1, 5)
            axs[1].spines[['right', 'top', 'left']].set_visible(False)
            axs[1].spines[['bottom']].set_linewidth(2)
            axs[1].tick_params(axis='x', width=2)
            axs[1].tick_params(axis='y', width=0)
            axs[1].set_xticks([1, 2, 3, 4, 5])
            axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=fontsize)
            axs[1].set_title('Subject')
            axs[1].set_xlabel('')

            axs[2].spines['bottom'].set_bounds(1, 5)
            axs[2].spines[['right', 'top', 'left']].set_visible(False)
            axs[2].spines[['bottom']].set_linewidth(2)
            axs[2].tick_params(axis='x', width=2)
            axs[2].tick_params(axis='y', width=0)
            axs[2].set_xticks([1, 2, 3, 4, 5])
            axs[2].set_xticklabels(axs[2].get_xticklabels(), fontsize=fontsize)
            axs[2].set_title('Error')
            axs[2].set_xlabel('')

            axs[2].legend(handles=custom_handles, frameon=False, fontsize=fontsize)

            fig.supxlabel('day', fontsize=fontsize)
            fig.suptitle('Variance decomposition of finger order', fontsize=fontsize)

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc2.var_dec_order.svg'), fig)

        # endregion

        # region SUCCESS_RATE
        case 'SUCCESS_RATE':

            experiment = 'efc1'

            np.random.seed(10)

            data = pd.read_csv(os.path.join(gl.baseDir, experiment, 'efc1_all.tsv'), sep='\t')
            data = data.groupby(['chordID', 'sess'])['trialCorr'].mean().reset_index()
            data['sessj'] = data['sess'] + np.random.uniform(-1.2, -.8, size=data.shape[0])

            fig, axs = plt.subplots(figsize=(3.5, 5))

            sns.scatterplot(x='sessj', y='trialCorr', data=data, ax=axs, color='grey', legend=False, s=100,
                            edgecolor='none', alpha=0.3)
            sns.boxplot(x='sess', y='trialCorr', data=data, ax=axs, color='darkgrey', width=0.4, linewidth=2, linecolor='k',
                        showfliers=False)

            axs.set_ylim([0., 1.05])
            axs.spines[['right', 'top']].set_visible(False)
            axs.spines[['left', 'bottom']].set_linewidth(2)
            axs.tick_params(axis='both', width=2)

            axs.set_xticklabels(axs.get_xticklabels(), fontsize=12)
            axs.set_yticklabels(axs.get_yticklabels(), fontsize=12)

            axs.set_xlabel('day', fontsize=12)
            axs.set_ylabel('success rate', fontsize=12)

            fig.tight_layout()

            decor(axs=axs, fontsize=fontsize, ybounds=(.0, 1.), xbounds=(0, 3), spines_width=2)

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc1.success_rate.svg'), fig)

        # endregion

        # region MEAN_DEVIATION:day_exp1
        case 'MEAN_DEVIATION:day_exp1':

            experiment = 'efc1'

            np.random.seed(8)

            data = pd.read_csv(os.path.join(gl.baseDir, experiment, 'efc1_chord.tsv'), sep='\t')
            data['MD'] = pd.to_numeric(data['MD'], errors='coerce')

            fig, axs = plt.subplots(figsize=(3.5, 5))

            cmap = plt.get_cmap('Blues')
            colors = cmap(np.linspace(0.2, 0.7, len(data['num_fingers'].unique())))
            palette = sns.color_palette(colors)

            sns.lineplot(data=data, ax=axs, x='sess', y='MD', hue='num_fingers', marker='o', palette=palette,
                         linewidth=3,
                         markeredgewidth=0, err_kws={'linewidth': 0, 'alpha': .2})

            fontsize = 12

            axs.set_ylabel('mean deviation (N)', fontsize=fontsize)
            axs.set_xlabel('day', fontsize=fontsize)
            # axs.set_ylim([0, 2.5])
            axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
            axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
            axs.spines[['right', 'top']].set_visible(False)
            axs.spines[['left', 'bottom']].set_linewidth(2)

            axs.tick_params(axis='both', width=2)

            axs.legend(frameon=False, fontsize=fontsize, title='number\nof fingers', title_fontsize=fontsize,
                       loc='upper right')

            fig.tight_layout()

            decor(axs=axs, fontsize=fontsize, ybounds=(.75, 2.5), xbounds=(1, 4), spines_width=2)

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc1.md.svg'), fig)

        # endregion

        # region EXECUTION_TIME:day_exp1
        case 'EXECUTION_TIME:day_exp1':

            experiment = 'efc1'

            np.random.seed(8)

            data = pd.read_csv(os.path.join(gl.baseDir, experiment, 'efc1_chord.tsv'), sep='\t')
            data['MT'] = pd.to_numeric(data['MT'], errors='coerce') / 1000

            fig, axs = plt.subplots(figsize=(3.5, 5))

            cmap = plt.get_cmap('Blues')
            colors = cmap(np.linspace(0.2, 0.7, len(data['num_fingers'].unique())))
            palette = sns.color_palette(colors)

            sns.lineplot(data=data, ax=axs, x='sess', y='MT', hue='num_fingers', marker='o', palette=palette,
                         linewidth=3,
                         markeredgewidth=0, err_kws={'linewidth': 0, 'alpha': .2})

            fontsize = 12

            axs.set_ylabel('execution time (s)', fontsize=fontsize)
            axs.set_xlabel('day', fontsize=fontsize)
            axs.set_ylim([0, 2.7])
            axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
            axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
            axs.spines[['right', 'top']].set_visible(False)
            axs.spines[['left', 'bottom']].set_linewidth(2)

            axs.tick_params(axis='both', width=2)

            axs.legend(frameon=False, fontsize=fontsize, title='number\nof fingers', title_fontsize=fontsize,
                       loc='upper right')

            fig.tight_layout()

            decor(axs=axs, fontsize=fontsize, ybounds=(.0, 2.5), xbounds=(1, 4), spines_width=2)

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc1.et.svg'), fig)

        # endregion

        # region EXAMPLE:meps
        case 'EXAMPLE:meps':

            experiment = 'efc3'
            session = 'emgTMS'
            day = 'day1'
            participant_id = 'subj114'

            meps = np.load(os.path.join(gl.baseDir, experiment, session, day, participant_id, 'mep_segmented.npy'))

            meps = np.delete(meps, 9, axis=1)

            fig, axs = plt.subplots(meps.shape[1], sharex=True, sharey=True, figsize=(3, 8))

            tAx = np.linspace(0, .05, meps.shape[-1])

            for a, ax in enumerate(axs):

                ax.plot(tAx, meps[:, a, :].T, lw=.8, color='k', alpha=.3)

                ax.spines[['bottom', 'top', 'right']].set_visible(False)
                ax.spines['left'].set_linewidth(2)
                ax.spines['left'].set_bounds(-2, 2)
                ax.tick_params(axis='y', width=0)
                ax.tick_params(axis='x', width=0)

            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['bottom'].set_bounds(0., .05)
            ax.tick_params(axis='x', width=2)
            ax.set_xticks([0., .025, .05])

            ax.set_xlabel('time relative to TMS pulse (s)')

            savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc3.meps.svg'), fig)


        # endregion

# >>>>>>> Stashed changes
if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('what', nargs='?', default=None)
    args = parser.parse_args()
    what = args.what

    plot(what)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    plt.show()
