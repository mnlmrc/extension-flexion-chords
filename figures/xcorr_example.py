import numpy as np
from matplotlib import pyplot as plt
import globals as gl
import os

from force import Force

experiment = 'efc2'
participant_id = 'subj100'
session = 'testing'
day = '1'
ntrial = 1

force = Force(experiment, participant_id, session, day)
xcorr, _, _ = force.crosscorrelation()
xcorr = xcorr[ntrial]

chordID = force.dat.loc[ntrial, 'chordID']
chordID_str = str(chordID)
fingers = np.array([f != '9' for f in chordID_str])

for i in range(len(fingers)):
    for j in range(len(fingers)):
        xcorr[i, j] = np.nan if (fingers[i] == False) or (fingers[j] == False) else xcorr[i, j]

fig, axs = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8, 8))

timeL = xcorr.shape[-1] / gl.fsample / 2

tAx = np.linspace(-timeL, timeL, xcorr.shape[-1])
for i in range(len(fingers)):
    for j in range(len(fingers)):
        axs[i, j].plot(tAx, xcorr[i, j], color='grey')
        axs[i, j].axvline(0, color='k', ls='--', lw=.8)
        axs[i, j].set_xlim([-2.5, 2.5])
        axs[i, j].set_ylim([-1, 1])
        if (i == 0) or (j == 0):
            axs[i, j].set_ylabel(gl.channels['force'][i], fontsize=12) if j == 0 else None
            axs[i, j].set_title(gl.channels['force'][j], fontsize=12) if i == 0 else None

fig.supxlabel('lag (s)')
fig.supylabel('correlation')

fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'xcorr_example.svg'), dpi=600)

plt.show()

