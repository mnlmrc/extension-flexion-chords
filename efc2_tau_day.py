import pandas as pd
from statannot import add_stat_annotation

from main import main
import matplotlib.pyplot as plt
import os
import globals as gl
import seaborn as sns

experiment = 'efc2'

tau = pd.read_csv(os.path.join(gl.baseDir, experiment, f'tau.tsv'), sep='\t')
tau.loc[tau['chord'] == 'trained', 'day'] = tau.loc[tau['chord'] == 'trained', 'day'].astype('float') + .05
tau.loc[tau['chord'] == 'untrained', 'day'] = tau.loc[tau['chord'] == 'untrained', 'day'].astype('float') - .05

fig, axs = plt.subplots(figsize=(3.5, 5))

sns.lineplot(data=tau, ax=axs, x='day', y='tau', hue='chord', errorbar='se', err_style='band',
                         palette=['red', 'blue'], marker='o', markeredgewidth=0, linewidth=3, err_kws={'linewidth': 0})

axs.legend(loc='upper right', ncol=1, frameon=False, fontsize=12)

fontsize = 12

x1, x2 = 1-.05, 5-.05
y, h, col = .95, .025, 'k'
axs.hlines(y, x1, x2, lw=2, color=col)
axs.vlines(x1, y - h, y, lw=2, color=col)
axs.vlines(x2, y - h, y, lw=2, color=col)
axs.text((x1 + x2) * .5, y, "*", ha='center', va='bottom', color=col, fontsize=fontsize)

x1, x2 = 1+.05, 5+.05
y, h, col = 1, .025, 'k'
axs.hlines(y, x1, x2, lw=2, color=col)
axs.vlines(x1, y - h, y, lw=2, color=col)
axs.vlines(x2, y - h, y, lw=2, color=col)
axs.text((x1 + x2) * .5, y, "*", ha='center', va='bottom', color=col, fontsize=fontsize)

axs.spines['bottom'].set_bounds(1, 5)  # Set x-axis spine to be shorter
axs.spines['left'].set_bounds(.2, .8)  # Set y-axis spine to be shorter
axs.set_ylabel('lag (s)', fontsize=fontsize)
axs.set_xlabel('day', fontsize=fontsize)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.tick_params(axis='both', width=2)


fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'tau_day.svg'), dpi=600)

plt.show()
