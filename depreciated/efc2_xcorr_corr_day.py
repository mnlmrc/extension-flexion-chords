import os

import pandas as pd
from matplotlib import pyplot as plt
from depreciated.main import main
import globals as gl
import seaborn as sns

experiment = 'efc2'

fontsize = 12

df = pd.read_csv(os.path.join(gl.baseDir, experiment, f'xcorr_corr_slope.tsv'), sep='\t')

fig, axs = plt.subplots(figsize=(4, 5))
sns.boxplot(data=df, ax=axs, x='day', y='corr', hue='chord', palette=['red', 'blue'], linewidth=2,
            linecolor='k', showfliers=False)
axs.set_ylabel("Pearson's r", fontsize=fontsize)
axs.set_xlabel("day", fontsize=fontsize)

# add noise ceiling
nc = main('NOISE_CEILING:tau', experiment)
axs.axhline(nc, color='grey', lw=2)
axs.text(axs.get_xlim()[1], nc, 'noise\nceiling', color='grey', va='center', ha='left', fontsize=fontsize)

# cosmetic
axs.spines[['right', 'top']].set_visible(False)
axs.legend(frameon=False, fontsize=fontsize)
axs.set_title('Correlation between\ninter-finger lags across trials', fontsize=fontsize)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.spines['bottom'].set_bounds(0, 4)  # Set x-axis spine to be shorter
axs.spines['left'].set_bounds(-.05, .3)  # Set y-axis spine to be shorter
axs.tick_params(axis='both', width=2)

axs.axhline(df['corr_shuff'].mean(), color='k', lw=2, ls='--')
axs.axhline(0, color='k', lw=2, ls='-')

axs.text(axs.get_xlim()[1], df['corr_shuff'].mean(), 'shuffled\ndata', va='center', ha='left', fontsize=fontsize)

fig.tight_layout()

# fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'xcorr_corr_day.svg'), dpi=600)

plt.show()
