import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind


def decor(axs=None, fontsize=None, xlim=(None, None), ylim=(None, None), xbounds=(None, None), ybounds=(None, None),
          spines_width=None):
    axs.set_ylim(ylim)

    axs.spines['bottom'].set_bounds(xbounds[0], xbounds[1])
    axs.spines['left'].set_bounds(ybounds[0], ybounds[1])
    axs.spines[['right', 'top']].set_visible(False)
    axs.spines[['left', 'bottom']].set_linewidth(spines_width)
    axs.tick_params(axis='both', width=spines_width)

    axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
    axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)



def add_significance_brackets(ax, data, x=None, y=None, pairs=None, test_type='t-test_ind', text_format='star',
                              y_max=None):
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
    if y_max is None:
        all_y_data = []
        for line in ax.get_lines():
            y_data = line.get_ydata()
            all_y_data.extend(y_data)  # Collect all y-values from the plotted lines

        # Find the maximum y-value from the collected data
        y_max = max(all_y_data) + 0.1 * max(all_y_data)
    brack_height = 0.01 * y_max

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
            if p_value < 0.0001:
                significance = '****'
            elif p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'n.s.'
        else:
            significance = f"p = {p_value:.3e}"

        # Plot significance bracket
        y_pos = y_max
        ax.plot([x1, x1, x2, x2], [y_pos, y_pos + brack_height, y_pos + brack_height, y_pos], lw=1.5, color='k')
        ax.text((x1 + x2) * 0.5, y_pos + brack_height, significance, ha='center', va='bottom', color='k', fontsize=12)

        return y_max + brack_height
    # # Adjust the y-axis to accommodate the significance brackets
    # ax.set_ylim([ax.get_ylim()[0], y_max + 3 * line_height])


def add_significance_asterisks(ax, data, x=None, y=None, hue=None, x_point=None, test_type='t-test_ind',
                               significance_level=0.05, text_format='star'):
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

    if len(hue_levels) != 2:
        raise ValueError("There must be exactly two groups at each x point to perform the t-test.")

    data1 = group_data[group_data[hue] == hue_levels[0]][y]
    data2 = group_data[group_data[hue] == hue_levels[1]][y]

    # Perform t-test
    if test_type == 't-test_ind':
        stat, p_value = ttest_ind(data1, data2, nan_policy='omit')
    elif test_type == 't-test_rel':
        stat, p_value = ttest_rel(data1, data2, nan_policy='omit')
    else:
        raise ValueError(f"Test type '{test_type}' not supported.")

    # Determine if the result is significant
    if p_value < significance_level:
        if text_format == 'star':
            if p_value < 0.0001:
                significance = '****'
            elif p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
        else:
            significance = f"p = {p_value:.3e}"

        # Find the maximum y-value for the x point and add an asterisk above it
        y_pos = y_max + 0.2 * y_max  # Adjust height for the asterisk
        ax.text(x_point, y_pos, significance, ha='center', va='bottom', color='k', fontsize=12)

    # Optionally, adjust the y-axis to make sure the asterisks are visible
    # ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] * 1.1])

# Example usage:
# Assuming 'ax' is your plot axis object, and you have data in a DataFrame 'df' with columns 'day', 'tau', 'chord'
# Example pairs: [(1, 2), (3, 4)]
# add_significance_brackets(ax, df, x='day', y='tau', hue='chord', pairs=[(1, 2), (3, 4)])
