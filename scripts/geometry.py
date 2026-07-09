import EFC_learningfMRI.globals as gl
import pandas as pd
import numpy as np
import os
import PcmPy as pcm

if __name__=='__main__':

    glm = 3
    atlas_name = 'ROI'
    rois = gl.rois[atlas_name]
    sns = [101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113]

    dist = {'dist': [], 'cos': [], 'angle': [], 'chord': [], 'session': [], 'sn': [], 'roi': [], 'Hem': []}
    for sn in sns:
        for H in gl.Hem:
            for r, roi in enumerate(rois):
                for s, sess in enumerate(gl.sessions):
                    
                    G             = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs.{H}.{roi}.npy'))

                    C             = pcm.G_to_cosine(G[s] )
                    cos_trained   = C[:4, :4].mean(axis=(0, 1))
                    cos_untrained = C[4:, 4:].mean(axis=(0, 1))
                    cos_tmp       = np.r_[cos_trained, cos_untrained]

                    A             = np.arccos(C)
                    ang_trained   = A[:4, :4].mean(axis=(0, 1))
                    ang_untrained = A[4:, 4:].mean(axis=(0, 1))
                    ang_tmp       = np.r_[ang_trained, ang_untrained]

                    D              = pcm.G_to_dist(G[s] )
                    dist_trained   = D[:4, :4].mean(axis=(0, 1))
                    dist_untrained = D[4:, 4:].mean(axis=(0, 1))
                    dist_tmp       = np.r_[dist_trained, dist_untrained]

                    sns            = np.r_[np.arange(dist_trained.size), np.arange(dist_trained.size)]

                    dist['dist'].extend(dist_tmp)
                    dist['cos'].extend(cos_tmp)
                    dist['angle'].extend(ang_tmp)
                    dist['chord'].extend(['trained'] * cos_trained.size + ['untrained'] * cos_trained.size)
                    dist['session'].extend([sess] * dist_tmp.size)
                    dist['sn'].extend([sn, sn])
                    dist['roi'].extend([roi] * dist_tmp.size)
                    dist['Hem'].extend([H] * dist_tmp.size)

    dist = pd.DataFrame(dist)
    dist.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)