import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from main import main
import globals as gl

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

axs.set_ylim([0, 1.1])
axs.spines[['right', 'top']].set_visible(False)

axs.set_xticklabels(axs.get_xticklabels(), fontsize=16)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=16)

axs.set_xlabel('day', fontsize=16)
axs.set_ylabel('success rate', fontsize=16)

fig.tight_layout()
# fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'success_rates.svg'), dpi=600)

plt.show()
