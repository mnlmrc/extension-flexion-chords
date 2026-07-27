import EFC_learningfMRI.globals as gl
import pandas as pd
import numpy as np
import os
import PcmPy as pcm
from EFC_learningfMRI.util import get_trained_and_untrained
from EFC_learningfMRI.geometry import G_sorted_mean

def split_trained(M):
    """Mean of the trained (first 4) and untrained (last 4) condition blocks."""
    trained   = M[:4, :4].mean(axis=(0, 1))
    untrained = M[4:, 4:].mean(axis=(0, 1))
    return np.r_[trained, untrained]

def calc_metrics(G, suffix=''):
    """Trained/untrained mean of every geometry metric for one second-moment matrix G.

    Each value is a length-2 array, ordered [trained, untrained].
    """
    euc = pcm.G_to_dist(G)
    cos = pcm.G_to_cosine(G)
    return {f'G{suffix}'    : split_trained(G),
            f'dist{suffix}' : split_trained(euc),
            f'cos{suffix}'  : split_trained(cos),
            f'angle{suffix}': split_trained(np.arccos(cos))}


if __name__=='__main__':

    glm = 3
    atlas_name = 'ROI'
    rois = gl.rois[atlas_name]
    sns = [101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113]

    rows = []

    for H in gl.Hem:
        for roi in rois:
            for sn in sns:

                G_obs = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs.{H}.{roi}.npy'))

                tr_untr = get_trained_and_untrained(sn)
                G_hat   = G_sorted_mean(sns, H, roi, order=tr_untr, exclude=sn)
                G_hat  *= np.trace(G_obs[0]) / np.trace(G_hat[0])

                for s, sess in enumerate(gl.sessions):
                    metrics = {**calc_metrics(G_obs[s]), **calc_metrics(G_hat[s], '_hat')}
                    for i, chord in enumerate(('trained', 'untrained')):
                        rows.append({**{k: v[i] for k, v in metrics.items()},
                                     'chord': chord, 'session': sess,
                                     'sn': sn, 'roi': roi, 'Hem': H})

    dist_obs = pd.DataFrame(rows)
    dist_obs.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_obs.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)