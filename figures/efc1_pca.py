import os.path

import matplotlib.pyplot as plt
import numpy as np

import globals as gl
import seaborn as sns

import pandas as pd

experiment = 'efc1'

pca = pd.read_csv(os.path.join(gl.baseDir, experiment, 'natChord_pca.tsv'), sep='\t')

fig, axs = plt.subplots(figsize=(3.5, 5))

fontsize=12

sns.lineplot(data=pca, ax=axs, x='PC', y='nat_explained', label='natural', linewidth=3, marker='o',
             markeredgecolor='none', err_kws={'linewidth': 0}, color="#1b7e81")
sns.lineplot(data=pca, ax=axs, x='PC', y='chord_explained', label='chords', linewidth=3, marker='o',
             markeredgecolor='none', err_kws={'linewidth': 0}, color='#b9b9b9')

axs.set_xticks(np.linspace(1, 10, 10))
axs.set_yticks(np.linspace(0, 60, 7))
axs.set_yticklabels([f'{y}%' for y in axs.get_yticks().astype(int)])

axs.legend(loc='upper right', ncol=1, frameon=False, fontsize=12)

axs.spines['bottom'].set_bounds(1, 10)  # Set x-axis spine to be shorter
axs.spines['left'].set_bounds(0, 60)  # Set y-axis spine to be shorter
axs.set_ylabel('% variance', fontsize=fontsize)
axs.set_xlabel('day', fontsize=fontsize)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.tick_params(axis='both', width=2)

fig.tight_layout()

plt.show()

