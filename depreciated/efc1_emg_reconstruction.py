import matplotlib.pyplot as plt
import numpy as np

from depreciated.main import main
import pandas as pd
import globals as gl
import os
import pickle

experiment = 'efc1'

width = .5
fontsize = 12

fig, axs = plt.subplots(figsize=(5, 5))
fig, axs, pval = main('PLOT:recon_emg', experiment, fig=fig, axs=axs, width=width, linewidth=2,
                            linecolor='k', showfliers=False, palette=['darkgrey', 'teal'])

with open(os.path.join(gl.baseDir, experiment, 'recon_emg.pkl'), 'rb') as file:
    recon_dict = pickle.load(file)

df = pd.DataFrame({key: recon_dict[key] for key in ['participant_id', 'r2_chord2nat_shuffle',  'r2_nat2chord_shuffle']})
df_shuffle = df.groupby('participant_id')[['r2_chord2nat_shuffle', 'r2_nat2chord_shuffle']].mean().reset_index()

pos = axs.get_xticks()

shuffle_chord = df_shuffle['r2_chord2nat_shuffle']
axs.hlines(y=shuffle_chord.mean(), xmin=pos[0] - width / 2, xmax=pos[0] + width / 2, color='k', ls=':', lw=2)
axs.hlines(y=np.array(recon_dict['nc_natural']).mean(), xmin=pos[0] - width / 2, xmax=pos[0] + width / 2,
           color='dimgrey', ls='-', lw=2)
shuffle_nat = df_shuffle['r2_nat2chord_shuffle']
axs.hlines(y=shuffle_nat.mean(), xmin=pos[1] - width / 2, xmax=pos[1] + width / 2, color='k', ls=':', lw=2)
axs.hlines(y=np.array(recon_dict['nc_chords']).mean(), xmin=pos[1] - width / 2, xmax=pos[1] + width / 2,
           color='dimgrey', ls='-', lw=2)

axs.text(axs.get_xlim()[1], shuffle_nat.mean(), 'shuffled data', va='center', ha='left', color='k', fontsize=fontsize)
axs.text(axs.get_xlim()[1], np.array(recon_dict['nc_chords']).mean(), 'noise ceiling', va='center', ha='left',
         color='dimgrey', fontsize=fontsize)
axs.text(axs.get_xlim()[0], shuffle_chord.mean(), 'shuffled data', va='center', ha='right', color='k', fontsize=fontsize)
axs.text(axs.get_xlim()[0], np.array(recon_dict['nc_natural']).mean(), 'noise ceiling', va='center', ha='right',
         color='dimgrey', fontsize=fontsize)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
axs.set_ylabel('R²', fontsize=fontsize)

axs.set_xticks([])
axs.set_xlabel('')

axs.set_xlim([-1.6, 2.6])

print(pval)
axs.text(0, .85, '*', color='dimgrey', va='center', ha='right', fontsize=fontsize)
axs.text(0, .85, '*', color='k', va='center', ha='left', fontsize=fontsize)

axs.spines[['right', 'bottom', 'top']].set_visible(False)
axs.spines['left'].set_linewidth(2)
axs.tick_params(axis='y', width=2)  # Change y-tick width

axs.text(0, .55, 'natural from chords', va='top', ha='center', color='k', fontsize=fontsize, rotation=90)
axs.text(1, .28, 'chords from natural', va='top', ha='center', color='k', fontsize=fontsize, rotation=90)

fig.tight_layout()

fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'emg_reconstruction.svg'), dpi=600)

plt.show()

