import os

import matplotlib.pyplot as plt
import globals as gl
from depreciated.main import main

experiment = 'efc2'


participant_id = gl.participants[experiment]
chordID = 91211
chord = 'trained'

fontsize = 12

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)

day = '1'
fig, ax, _ = main('PLOT:xcorr_chord', experiment, participant_id=participant_id, chordID=chordID, chord=chord, day=day, fig=fig, axs=axs[0])
ax.set_title('day 1', fontsize=fontsize)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)

day = '4'
fig, ax, cax = main('PLOT:xcorr_chord', experiment, participant_id=participant_id, chordID=chordID, chord=chord, day=day, fig=fig, axs=axs[1])
ax.set_title('day 5', fontsize=fontsize)

cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', location='bottom', fraction=0.046)
cbar.set_label('lag (s)', fontsize=fontsize)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)

ax.set_xlim([.5, 4.5])
ax.set_ylim([.5, 4.5])

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'lag_example.svg'), dpi=300)

plt.show()

# fig.suptitle(f'Average lag at maximum cross-correlation\nbetween finger pairs (chord: {chordID}, {chord}, N={len(participant_id)})')