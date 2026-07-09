import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import load_glm_onset, get_trained_and_untrained
import nitools.spm as spm
import numpy as np
import pandas as pd
import os

if __name__=='__main__':

    tAx = np.arange(-3, 17)

    sns = [101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113]

    glm = 3

    df = pd.DataFrame()
    for H in ['L']: #gl.Hem:
        for r, roi in enumerate(gl.rois['ROI']):
            for sn in sns:
                print(f'doing participant {sn}, {H}, {roi}, glm {glm}, fitted')

                # load bold
                bold = np.load(os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', f'BOLD.adj.{H}.{roi}.npy'))

                # retrieve onsets
                onset, events = load_glm_onset(sn, glm, output_events=True)

                bold_cut = spm.cut(bold, pre=3, at=onset, post=16)  # (n_trials, 20, n_voxels)

                # average over voxels -> one timecourse (20 samples) per trial
                tc = bold_cut.mean(axis=2)

                trained_untrained = get_trained_and_untrained(sn)

                # tag each trial's timecourse with its chord and session
                trials = pd.DataFrame(tc, columns=tAx)
                trials['chordID'] = np.asarray(events.chordID)
                trials['sess'] = np.asarray(events.day)
                trials['chord'] = 'untrained'
                trained_ids = [int(c) for c in trained_untrained[:4]]
                trials.loc[trials.chordID.isin(trained_ids), 'chord'] = 'trained'

                # reshape to long/tidy form and add the descriptor columns
                sub = trials.melt(id_vars=['chordID', 'sess', 'chord'], var_name='time', value_name='signal')
                sub['sn'] = sn
                sub['roi'] = roi
                sub['Hem'] = H

                df = pd.concat([df, sub], ignore_index=True)
            
    df.to_csv(os.path.join(gl.baseDir, 'bold', 'bold_segmented.tsv'), sep='\t', index=False)
