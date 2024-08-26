import pandas as pd
from scipy.stats import ttest_rel
from statannot import add_stat_annotation

import matplotlib.pyplot as plt
import os
import globals as gl
import seaborn as sns

from plot import add_significance_brackets, add_significance_asterisks
from util import savefig

experiment = 'efc2'

tau = pd.read_csv(os.path.join(gl.baseDir, experiment, f'tau.tsv'), sep='\t')
tau.loc[tau['chord'] == 'trained', 'dday'] = tau.loc[tau['chord'] == 'trained', 'day'].astype('float') + .05
tau.loc[tau['chord'] == 'untrained', 'dday'] = tau.loc[tau['chord'] == 'untrained', 'day'].astype('float') - .05

fig, axs = plt.subplots(figsize=(3.5, 5))

sns.lineplot(data=tau, ax=axs, x='dday', y='tau', hue='chord', errorbar='se', err_style='band',
                         palette=['red', 'blue'], marker='o', markeredgewidth=0, linewidth=3, err_kws={'linewidth': 0})

fontsize = 12

axs.set_ylim([.101, .999])
axs.spines['bottom'].set_bounds(1, 5)  # Set x-axis spine to be shorter
axs.spines['left'].set_bounds(.2, .9)  # Set y-axis spine to be shorter
axs.set_ylabel('lag (s)', fontsize=fontsize)
axs.set_xlabel('day', fontsize=fontsize)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.tick_params(axis='both', width=2)

y_max = add_significance_brackets(ax=axs, data=tau[tau['chord'] == 'untrained'], x='dday', y='tau', pairs=[(.95, 4.95)],
                          test_type='t-test_rel', text_format='star')
add_significance_brackets(ax=axs, data=tau[tau['chord'] == 'trained'], x='dday', y='tau', pairs=[(1.05, 5.05)],
                          test_type='t-test_rel', text_format='star', y_max=y_max + .05 * y_max)
add_significance_asterisks(ax=axs, data=tau, x='day', y='tau', x_point=5, hue='chord',
                          test_type='t-test_rel', text_format='star')

axs.legend(loc='best', ncol=1, frameon=False, fontsize=12)
fig.tight_layout()

# savefig(os.path.join(gl.baseDir, experiment, 'figures', 'tau_day.svg'), fig)

plt.show()
