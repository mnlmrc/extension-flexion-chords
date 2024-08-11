import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def plot_rep(df, y=None, save_path=None):

    # if title == None:
    #     title = y

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

    # # Add the custom legend to the plot
    # axs.legend(handles=custom_handles, title='chord')
    #
    # axs.set_title(title)
    # axs.set_ylabel(ylabel)
    # axs.set_xlabel(xlabel)

    if save_path is not None:
        fig.savefig(save_path)


def plot_days(df, y=None, save_path=None):
    # if title == None:
    #     title = y

    sns.pointplot(df, ax=axs, x='day', y=y, hue='chord', dodge=True, linestyle='none', errorbar='se',
                  palette=['red', 'blue'])
    # axs.set_title(title)
    # axs.set_ylabel(ylabel)
    # axs.set_xlabel(xlabel)

    if save_path is not None:
        fig.savefig(save_path)
