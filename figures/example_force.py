import matplotlib.pyplot as plt
from main import main
import globals as gl
import os

experiment = 'efc2'
participant_id = 'subj100'
session = 'testing'
day = '1'
ntrial = 1

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6, 3))
_, ax, chordID, day, chord = main('PLOT:force_in_trial', experiment, participant_id, session, day, ntrial=ntrial,
                                  fig=fig, axs=axs[0], linewidth=3)

ax.set_ylabel('force (N)', fontsize=12)
ax.set_title(f'day{day}', fontsize=12)
ax.spines[['left', 'bottom']].set_linewidth(2)
ax.tick_params(axis='both', width=2)  # Change y-tick width

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

print(f'day{day}, chordID: {chordID}')

ax.spines[['right', 'top']].set_visible(False)

session = 'training'
day = '4'
ntrial = 54
_, ax, chordID, day, chord = main('PLOT:force_in_trial', experiment, participant_id, session, day, ntrial=ntrial,
                                  fig=fig, axs=axs[1], linewidth=3)
ax.set_title(f'day{day}', fontsize=12)

print(f'day{day}, chordID: {chordID}')

ax.spines[['right', 'top', 'left']].set_visible(False)
ax.tick_params(axis='y', which='both', left=False, labelleft=False)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params(axis='x', width=2)  # Change y-tick width
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)

ax.set_ylim([-6, 6])

fig.legend(gl.channels['force'], loc='upper right', ncol=1, frameon=False, fontsize=12)

fig.supxlabel('time (s)', fontsize=12)

fig.tight_layout()

# fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'force_example.svg'))

plt.show()
