import pandas as pd
import os

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from util import calc_success
import seaborn as sns
import globals as gl
import matplotlib.pyplot as plt

experiment = 'efc2'

metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.csv'))
df_success = calc_success(metrics)

fig, axs = plt.subplots(figsize=(3.5, 5))

fontsize = 12

sns.boxplot(df_success, ax=axs, x='day', y='success', hue='chord', palette=['red', 'blue'], linewidth=2,
            linecolor='k', showfliers=False)

axs.legend( loc='lower right', ncol=1, frameon=False, fontsize=12)

axs.spines['bottom'].set_bounds(0, 4)  # Set x-axis spine to be shorter
axs.spines['left'].set_bounds(.65, 1)  # Set y-axis spine to be shorter
axs.set_ylabel('success rate', fontsize=fontsize)
axs.set_xlabel('day', fontsize=fontsize)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_xticklabels(axs.get_xticklabels(), fontsize=fontsize)
axs.spines[['right', 'top']].set_visible(False)
axs.spines[['left', 'bottom']].set_linewidth(2)
axs.tick_params(axis='both', width=2)

fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'success_rate.svg'), dpi=300)

plt.show()

