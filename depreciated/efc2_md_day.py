import matplotlib.pyplot as plt
from depreciated.main import main
import pandas as pd
import os
import globals as gl
from util import calc_avg
from scipy.stats import ttest_rel

experiment = 'efc2'

metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.csv'))
df = calc_avg(metrics, by=['chord', 'day', 'participant_id'], columns='MD')

pval = {
    'trained-untrained_day1': None,
    'trained-untrained_day5': None,
    'day1-day5_trained': None,
    'day1-day5_untrained': None,
}
_, pval['trained-untrained_day1'] = ttest_rel(df.loc[(df['chord'] == 'trained') & (df['day'] == 1), 'MD'],
                                              df.loc[(df['chord'] == 'untrained') & (df['day'] == 1), 'MD'])
_, pval['trained-untrained_day5'] = ttest_rel(df.loc[(df['chord'] == 'trained') & (df['day'] == 5), 'MD'],
                                              df.loc[(df['chord'] == 'untrained') & (df['day'] == 5), 'MD'])
_, pval['day1-day5_trained'] = ttest_rel(df.loc[(df['chord'] == 'trained') & (df['day'] == 1), 'MD'],
                                         df.loc[(df['chord'] == 'trained') & (df['day'] == 5), 'MD'])
_, pval['day1-day5_untrained'] = ttest_rel(df.loc[(df['chord'] == 'untrained') & (df['day'] == 1), 'MD'],
                                           df.loc[(df['chord'] == 'untrained') & (df['day'] == 5), 'MD'])

print(pval)

fig, axs = plt.subplots(figsize=(3.5, 5))
fig, axs, custom_handles = main('PLOT:metric_day', experiment, metric='MD', fig=fig, axs=axs,
                                palette=['red', 'blue'], linewidth=3, err_kw={'linewidth': 0})

fig.legend(handles=custom_handles, loc='upper right', ncol=1, frameon=False, fontsize=12)

fontsize = 12

axs.spines['bottom'].set_bounds(1, 5)  # Set x-axis spine to be shorter
axs.spines['left'].set_bounds(.6, 1.6)  # Set y-axis spine to be shorter
axs.set_ylabel('MD (a.u.)', fontsize=fontsize)
axs.set_xlabel('day', fontsize=fontsize)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.tick_params(axis='both', width=2)

fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'md_day.svg'), dpi=600)

plt.show()
