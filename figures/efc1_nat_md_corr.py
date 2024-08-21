import pandas as pd
import os
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt

import globals as gl

experiment = 'efc1'

fontsize = 12

data = pd.read_csv(os.path.join(gl.baseDir, experiment, 'natChord_analysis.tsv'), sep='\t')
data = data.groupby(['chordID', 'num_fingers'])[['MD', 'RT', 'log_slope', 'd']].mean().reset_index()

cmap = plt.get_cmap('Blues')
colors = cmap(np.linspace(0.7, 0.2, len(data['num_fingers'].unique())))
palette = sns.color_palette(colors)

fig, axs = plt.subplots(figsize=(3.5, 5))

plt.sca(axs)
for n, num_fingers in enumerate(data['num_fingers'].unique()):
    subset = data[data['num_fingers'] == num_fingers]

    # Plot scatter and regression line on the specified axes
    sns.regplot(data=subset, ax=axs, y='MD', x='log_slope', scatter=True,
                label=f'{num_fingers}', color=palette[n], line_kws={'linewidth': 3},
                scatter_kws={'s': 50, 'alpha': 0.8, 'edgecolor': 'none'})

axs.set_ylabel('MD (a.u.)', fontsize=fontsize)
axs.set_xlabel('log-likelihood that chord belongs\nto natural distribution', fontsize=fontsize)
axs.legend(title='number\nof fingers', loc='upper right', frameon=False)
axs.set_xlim([10, 20])
axs.set_ylim([.2, 2.7])
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.spines['bottom'].set_bounds(10, 20)  # Set x-axis spine to be shorter
axs.spines['left'].set_bounds(.5, 2.5)  # Set y-axis spine to be shorter
axs.tick_params(axis='both', width=2)

fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc1_nat_md_corr.svg'), dpi=300)

plt.show()
