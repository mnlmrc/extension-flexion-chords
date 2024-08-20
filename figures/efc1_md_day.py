import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from main import main
import globals as gl

experiment = 'efc1'

np.random.seed(8)

data = pd.read_csv(os.path.join(gl.baseDir, experiment, 'efc1_chord.tsv'), sep='\t')
data['MD'] = pd.to_numeric(data['MD'], errors='coerce')
# hues = np.random.permutation(data['num_fingers'].unique())
# offset = .05
# hue_mapping = {h: i * offset for i, h in enumerate(hues)}
# data['sess'] = data['sess'] + data['num_fingers'].map(hue_mapping)
# data['MD'] = data['MD']

fig, axs = plt.subplots(figsize=(3.5, 5))

cmap = plt.get_cmap('Blues')
colors = cmap(np.linspace(0.2, 0.7, len(data['num_fingers'].unique())))
palette = sns.color_palette(colors)

sns.lineplot(data=data, ax=axs, x='sess', y='MD', hue='num_fingers', marker='o', palette=palette, linewidth=3,
             markeredgewidth=0, err_kws={'linewidth': 0, 'alpha': .2})

fontsize = 12

axs.set_ylabel('MD (a.u.)', fontsize=fontsize)
axs.set_xlabel('day', fontsize=fontsize)
# axs.set_ylim([0, 2.5])
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.tick_params(axis='both', width=2)

axs.legend(frameon=False, fontsize=fontsize, title='number\nof fingers', title_fontsize=fontsize, loc='upper right')

fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'md_efc1_day.svg'), dpi=300)

plt.show()

