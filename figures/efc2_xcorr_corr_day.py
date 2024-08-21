import os

import pandas as pd
from matplotlib import pyplot as plt
from main import main
import globals as gl

experiment = 'efc2'

fontsize = 12

df = pd.read_csv(os.path.join(gl.baseDir, experiment, f'xcorr_corr_slope.tsv'), sep='\t')

fig, axs = plt.subplots()
fig, ax = main('PLOT:xcorr_corr', experiment, fig=fig, axs=axs)
ax.set_ylabel("Pearson's r", fontsize=fontsize)
ax.set_xlabel("day", fontsize=fontsize)

# add noise ceiling
nc = main('NOISE_CEILING:tau', experiment)
ax.axhline(nc, color='grey', lw=2)
ax.text(ax.get_xlim()[1], nc, 'noise\nceiling', color='grey', va='center', ha='left', fontsize=fontsize)

# cosmetic
ax.spines[['right', 'top']].set_visible(False)
ax.legend(frameon=False, fontsize=fontsize)
ax.set_title('Correlation between\ninter-finger lags across trials', fontsize=fontsize)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(2)
ax.spines['bottom'].set_bounds(0, 4)  # Set x-axis spine to be shorter
ax.spines['left'].set_bounds(-.05, .3)  # Set y-axis spine to be shorter
ax.tick_params(axis='both', width=2)

axs.axhline(df['corr_shuff'].mean(), color='k', lw=2, ls='--')
axs.axhline(0, color='k', lw=2, ls='-')

axs.text(axs.get_xlim()[1], df['corr_shuff'].mean(), 'shuffled\ndata', va='center', ha='left', fontsize=fontsize)

fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'xcorr_corr_day.svg'), dpi=600)

plt.show()

