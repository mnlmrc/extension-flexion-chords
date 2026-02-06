import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PcmPy as pcm
import os
import globals as gl
import seaborn as sb
from matplotlib.lines import Line2D
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def plot_behav(fig, ax, df, metric='ET', ylim=[0, 2.5], melt=False, id_vars=None, value_vars=None, var_name=None,
               ylabel=None, title=None):
    """
    Plot behavioural metrics assessed trial by trial
    Args:
        metric: 

    Returns: fig, ax

    """
    if melt and (id_vars is None or value_vars is None):
        pass # implement error

    max_bn = 0
    inset_list = []
    lines = []

    for day in df.day.unique():
        dat_tmp = df[df['day'] == day]
        if melt:
            dat_tmp = dat_tmp.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=metric)

        dat_tmp['BN'] = dat_tmp['BN'] + max_bn
        max_bn = dat_tmp.BN.max()
        min_bn = dat_tmp.BN.min()
        ax.text((max_bn + min_bn) / 2, ylim[0], f'{day}', ha='center', va='center', fontsize=8)
        ax.axvline(max_bn + .5, color='k', linestyle='-', lw=.8)
        ax.tick_params('x', bottom=False, labelbottom=False)
        ax.spines[['bottom', 'top', 'right']].set_visible(False)
        dat_bn = dat_tmp.groupby(['subNum', 'day', 'chord', 'BN']).mean(numeric_only=True).reset_index()

        dat_d = dat_tmp.groupby(['subNum', 'day', 'chord']).mean(numeric_only=True).reset_index()

        if len(dat_tmp.chord.unique()) == 1:
            fixed_width = 2.5
        else:
            fixed_width = 5

        center = (min_bn + max_bn) / 2
        x0 = center - fixed_width / 2
        y0 = ylim[1]
        height = (ylim[1] - ylim[0]) * .2
        inset = ax.inset_axes([x0, y0, fixed_width, height], transform=ax.transData)
        sb.barplot(data=dat_d, ax=inset, hue='chord', y=metric, legend=False, palette=['red', 'blue'],
                    hue_order=['trained', 'untrained'], errorbar='se')
        inset.spines[['top', 'right', 'bottom']].set_visible(False)
        inset.set_xticks([])

        # add sig bars
        if len(dat_tmp.chord.unique()) > 1:
            # do t-test
            a, b = dat_d[dat_d['chord'] == 'trained'][metric], dat_d[dat_d['chord'] == 'untrained'][metric]
            tval, pval = ttest_rel(a, b)
            lines.append(f'trained vs. untrained, day{day}: tval={tval:.3f}, pval={pval:.3f}')
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            else:
                stars = None
            ab = np.c_[a, b]
            bars = inset.patches
            # x1 = bars[0].get_x() + bars[0].get_width() / 2
            # x2 = bars[1].get_x() + bars[1].get_width() / 2
            if stars:
                # offset = .05 * inset.get_ylim()[1]
                y_max = ab.mean(axis=1).max()
                y_argmax = ab.mean(axis=1).argmax()
                se = ab[y_argmax].std() / np.sqrt(ab.shape[1])
                y_max += se

        if day == 1:
            sb.lineplot(data=dat_bn, ax=ax, x='BN', y=metric, hue='chord', errorbar='se', lw=1,
                         palette=['red', 'blue'], err_kws={'linewidth': 0}, legend=True)
            inset.set_ylabel(ylabel, fontsize=8)
        else:
            sb.lineplot(data=dat_bn, ax=ax, x='BN', y=metric, hue='chord', errorbar='se', lw=1,
                         palette=['red', 'blue'], err_kws={'linewidth': 0}, legend=False)
            inset.spines[['left']].set_visible(False)
            inset.set_yticks([])
            inset.set_ylabel('', fontsize=8)
        inset.set_ylim(ylim)
        inset_list.append(inset)

    ax.set_ylim(ylim)
    ax.set_xlim([-10, max_bn])
    ax.spines['left'].set_bounds(ylim)
    ax.text(max_bn / 2, ylim[0] - .05 * (ylim[1] - ylim[0]), '# session', ha='center', va='top', fontsize=10)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', bbox_to_anchor=(1, -.01), ncol=2, frameon=False)
    ax.set_title(title)

    print("\n".join(lines))

    return fig, ax, inset_list

def plot_rep(fig, ax, df, metric='ET', ylim=[0, 2.5], ylabel=None, title=None):
    offset_rep = 0
    for day in df.day.unique():
        dat_tmp = df[df['day'] == day]
        dat_tmp.Repetition = dat_tmp.Repetition + offset_rep
        max_rep = dat_tmp.Repetition.max()
        min_rep = dat_tmp.Repetition.min()
        offset_rep += 2
        ax.tick_params('x', bottom=False, labelbottom=False)
        ax.spines[['bottom', 'top', 'right']].set_visible(False)

        dat_d = dat_tmp.groupby(['subNum', 'day', 'chord', 'Repetition']).mean(numeric_only=True).reset_index()

        sb.lineplot(data=dat_d, ax=ax, x='Repetition', y=metric, hue='chord', errorbar='se', lw=1, marker='s',
                    markeredgecolor=None, ms=3, palette=['red', 'blue'], err_kws={'linewidth': 0},
                    legend=True if day == 1 else False)
        ax.text((max_rep + min_rep) / 2, ylim[0], f'{day}', ha='center', va='center', fontsize=8)
        ax.axvline(max_rep + .5, color='k', linestyle='-', lw=.8)

    ax.set_ylim(ylim)
    ax.spines['left'].set_bounds(ylim)
    ax.set_ylabel(ylabel)
    ax.text(max_rep / 2, ylim[0] - .05 * (ylim[1] - ylim[0]), '# day', ha='center', va='top', fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1, -.01), ncol=2, frameon=False)
    ax.set_title(title)

    return fig, ax

def lineplot_roi_avg(fig, axs, df, metric, hue=None, hue_order=None, color=None, label=None,
                     H='L', rois=['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']):
    if isinstance(color, list):
        palette=color
    else:
        palette = None

    sess_map = {
        3: 0,
        9: 1,
        23: 2
    }
    df['session'] = df['session'].map(sess_map)
    for r, roi in enumerate(rois):
        ax = axs[r]
        sb.lineplot(df[(df['roi'] == roi) & (df['Hem'] == H)],
                     ax=ax,
                     y=metric,
                     x='session',
                     hue=hue,
                     palette=None if hue is None else palette,
                     color=None if isinstance(color, list) else color,
                     hue_order=hue_order,
                     errorbar='se',
                     legend=False,
                     err_kws={'linewidth': 0}
                     )
        ax.axhline(0, ls='-', color='k', lw=.8)
        ax.set_title(roi)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_facecolor('lightgrey')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines[['bottom']].set_bounds(0, 2)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['3', '9', '23'])
        if r == 0:
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_position(('data', -1))
        else:
            ax.tick_params(axis=('y'), labelleft=False, length=0)
    # fig.supylabel('activation (a.u.)')
    # fig.suptitle(f'Average activity in ROIs, hemisphere:{H}, N={N}')
    if label is not None:
        legend_handles = [Line2D([0], [0], color=col, label=lab) for col, lab in zip(color, label)]
        fig.legend(handles=legend_handles,
                   loc='center left',
                   bbox_to_anchor=(1, .5),
                   frameon=False,
                   ncol=1,
                   fontsize=10)

    return fig, axs


def add_significance_bars(ax, tAx, sig, color='black', position='bottom', height=0.02, alpha=.5, spacing=0.005,
                          linestyle='-', linewidth=4):
    """
    Adds a horizontal significance line above or below the signal.
    Automatically stacks multiple bars to avoid overlap.

    Parameters:
    - ax: matplotlib axis
    - tAx: time axis (1D)
    - sig: boolean array (same shape as tAx) indicating significance
    - color: line color
    - position: 'top' or 'bottom'
    - height: line height as fraction of axis height (0.02 = 2%)
    - spacing: vertical spacing between stacked bars (axes coords)
    - linestyle: style of the line ('-', '--', ':', '-.')
    - linewidth: thickness of the line
    """
    from itertools import groupby
    from operator import itemgetter
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # Initialize storage for stacking info
    if not hasattr(ax, "_sig_bar_counts"):
        ax._sig_bar_counts = {"top": 0, "bottom": 0}

    offset_idx = ax._sig_bar_counts[position]

    transform = ax.get_xaxis_transform()  # x in data, y in axes coords

    if position == 'top':
        y = 1 - height - offset_idx * (height + spacing)
    else:
        y = offset_idx * (height + spacing)

    # Identify contiguous significant regions
    sig_regions = [(tAx[g[0][0]], tAx[g[-1][0]])
                   for k, g in groupby(enumerate(sig), key=itemgetter(1))
                   if k for g in [list(g)]]

    # Add lines for each significant region
    for start, end in sig_regions:
        line = mlines.Line2D([start, end], [y, y],
                             transform=transform,
                             color=color, alpha=alpha,
                             linestyle=linestyle,
                             linewidth=linewidth,
                             solid_capstyle='butt',
                             zorder=1e6)
        ax.add_line(line)

    # Increment stacking counter
    ax._sig_bar_counts[position] += 1


def plot_pcm_corr(fig, axs, panel, Mflex, theta, theta_g, r_bootstrap=None):
    ax = axs[panel]

    N = theta.shape[1]

    sigma2_1 = np.exp(theta[0])
    sigma2_2 = np.exp(theta[1])
    r_indiv = Mflex.get_correlation(theta)
    sigma2_e = np.exp(theta[3])
    SNR = np.sqrt(sigma2_1 * sigma2_2) / sigma2_e
    ax.scatter(SNR, r_indiv, color='k')

    theta_g, _ = pcm.group_to_individ_param(theta_g, Mflex, N)
    r_group = Mflex.get_correlation(theta_g)
    ax.axhline(r_group[0], color='r', linestyle='--')
    ax.axhline(0, color='k', linestyle='-', lw=.8)

    ax.set_ylim(-1.2, 1.2)
    # ax.set_xlim(-.05, .5)

    ax.spines[['top', 'right', 'left']].set_visible(False)

    if panel == 0:
        ax.spines[['left']].set_visible(True)
    else:
        ax.set_yticks([])

    if r_bootstrap is not None:
        (ci_lo, ci_hi), _, _ = bootstrap_summary(r_bootstrap, alpha=0.05)
        print(f"group estimate:{r_group[0]} central 90% CI for r: [{ci_lo:.3f}, {ci_hi:.3f}]")
        ax.axhspan(ci_lo, ci_hi, lw=0, color='lightgrey', zorder=0)

    return fig, axs


def add_grid_legend(fig, anchor=(0.05, 0.1, 0.1, 0.18),
                    col_labels=('day 1','day 2','day 3'),
                    row_labels=('trained','untrained'),
                    markers=('o','+','s'),
                    row_colors=('red','blue'),
                    markersize=30,
                    facecolor='lightgrey'):
    # anchor = (left, bottom, width, height) in figure fraction
    legax = fig.add_axes(anchor)
    legax.set_axis_off()

    nrows, ncols = len(row_labels), len(col_labels)
    legax.set_xlim(0, ncols + 1.2)   # +space for row labels
    legax.set_ylim(0, nrows + 1.2)
    legax.set_facecolor(facecolor)

    # column headers
    for j, lbl in enumerate(col_labels, start=1):
        legax.text(j, nrows + 0.9, lbl, ha='center', va='center', fontsize=10)

    # row labels + symbol cells
    for i, (r_lbl, c) in enumerate(zip(row_labels, row_colors), start=1):
        y = nrows + 1 - i
        legax.text(.075, y, r_lbl, ha='right', va='center', fontsize=10)
        for j, m in enumerate(markers, start=1):
            legax.scatter(j, y, s=markersize, marker=m, c=c)
