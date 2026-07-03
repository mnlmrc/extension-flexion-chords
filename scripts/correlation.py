import EFC_learningfMRI.globals as gl
import pandas as pd
import numpy as np
import os
import PcmPy as pcm

if __name__=='__main__':

    corrs = ['3-9', '9-23', '3-23']
    chords = ['trained', 'untrained']
    atlas = 'ROI'
    rois = gl.rois[atlas]

    glm = 3

    r_xval = {'R': [], 'corr': [], 'sn': [], 'roi': [], 'Hem': [], 'chord': []}

    for H in gl.Hem:
        for roi in rois:
            for corr in corrs:
                for chord in chords:
                    cov = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.corr_across_sess.glm{glm}.{corr}.{chord}.{H}.{roi}.npy'))
                    std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))  # shape: (N, k)
                    r = cov / (std[:, :, None] * std[:, None, :])
                    r_avg = np.diagonal(r[:, :4, 4:], axis1=1, axis2=2).mean(axis=1)
                    N = r_avg.size
                    r_xval['R'].extend(r_avg)
                    r_xval['corr'].extend([corr] * N)
                    r_xval['sn'].extend(np.linspace(1, N, N, dtype=int))
                    r_xval['chord'].extend([chord] * N)
                    r_xval['Hem'].extend([H] * N)
                    r_xval['roi'].extend([roi] * N)
    r_xval = pd.DataFrame(r_xval)
    r_xval.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'xval_correlation.{atlas}.glm{glm}.tsv'), sep='\t', index=False)