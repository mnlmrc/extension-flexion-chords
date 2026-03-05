import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import globals as gl

experiment = 'efc1'

np.random.seed(8)

data = pd.read_csv(os.path.join(gl.baseDir, experiment, 'efc1_chord.tsv'), sep='\t')
data['MT'] = pd.to_numeric(data['MT'], errors='coerce')
data = data.groupby(['chordID', 'sess', 'num_fingers'])['MT'].mean().reset_index()
data.rename(columns={'num_fingers': 'number\nof fingers'}, inplace=True)

# Add an offset to x based on the hue
hues = np.random.permutation(data['number\nof fingers'].unique())
offset = .05
hue_mapping = {h: i * offset for i, h in enumerate(hues)}
data['sess'] = data['sess'] + data['number\nof fingers'].map(hue_mapping)
data['MT'] = data['MT'] / 1000

fig, axs = plt.subplots(figsize=(3.5, 5))

cmap = plt.get_cmap('Blues')
colors = cmap(np.linspace(0.2, 0.7, len(data['number\nof fingers'].unique())))
palette = sns.color_palette(colors)

sns.lineplot(data=data, ax=axs, x='sess', y='MT', hue='number\nof fingers', marker='o', palette=palette, linewidth=3,
             markeredgewidth=0, err_kws={'linewidth': 0, 'alpha': .2})

fontsize = 12

axs.set_ylabel('ET (s)', fontsize=fontsize)
axs.set_xlabel('day', fontsize=fontsize)
axs.set_ylim([0, 3])
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.tick_params(axis='both', width=2)

axs.legend(frameon=False, fontsize=fontsize, title='number\nof fingers', title_fontsize=fontsize)

fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'et_efc1_day.svg'), dpi=300)

plt.show()

