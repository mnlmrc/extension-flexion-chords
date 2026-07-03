import EFC_learningfMRI.globals as gl
import pandas as pd
import numpy as np
import os
import PcmPy as pcm

if __name__=='__main__':

    glm = 3
    rois = gl.rois['ROI']

    for H in ['L']:

        # trained-untrained
        dist = {'dist': [], 'cos': [], 'angle': [], 'session': [], 'sn': [], 'roi': [], 'Hem': []}
        for r, roi in enumerate(rois):
            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.trained-untrained.glm{glm}.{H}.{roi}.npy'))
            D = pcm.G_to_dist(G)
            C = pcm.G_to_cosine(G)
            A = np.arccos(C)
            for I, i in enumerate(np.arange(1, 6, 2)):
                dist_tmp = D[:, i-1, i]
                cos_tmp = C[:, i-1, i]
                ang_tmp = A[:, i-1, i]
                dist['dist'].extend(dist_tmp)
                dist['cos'].extend(cos_tmp)
                dist['angle'].extend(ang_tmp)
                dist['session'].extend(np.repeat(gl.sessions[I], dist_tmp.size))
                dist['sn'].extend(np.arange(dist_tmp.size))
                dist['roi'].extend([roi] * dist_tmp.size)
                dist['Hem'].extend([H] * dist_tmp.size)
        dist = pd.DataFrame(dist)
        dist.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dist.trained-untrained.glm{glm}.{H}.tsv'), sep='\t', index=False)

        # chord-session
        dist = {'dist': [], 'cos': [], 'angle': [], 'chord': [], 'session': [], 'sn': [], 'roi': [], 'Hem': []}
        for r, roi in enumerate(rois):
            for s, sess in enumerate(gl.sessions):
                G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.chord-session.glm{glm}.{H}.{roi}.npy'))

                C = pcm.G_to_cosine(G[:, s] )
                cos_trained = C[:, :4, :4].mean(axis=(1, 2))
                cos_untrained = C[:, 4:, 4:].mean(axis=(1, 2))
                cos_tmp = np.r_[cos_trained, cos_untrained]

                A = np.arccos(C)
                ang_trained = A[:, :4, :4].mean(axis=(1, 2))
                ang_untrained = A[:, 4:, 4:].mean(axis=(1, 2))
                ang_tmp = np.r_[ang_trained, ang_untrained]

                D = pcm.G_to_dist(G[:, s] )
                dist_trained = D[:, :4, :4].mean(axis=(1, 2))
                dist_untrained = D[:, 4:, 4:].mean(axis=(1, 2))
                dist_tmp = np.r_[dist_trained, dist_untrained]

                sns = np.r_[np.arange(dist_trained.size), np.arange(dist_trained.size)]

                dist['dist'].extend(dist_tmp)
                dist['cos'].extend(cos_tmp)
                dist['angle'].extend(ang_tmp)
                dist['chord'].extend(['trained'] * cos_trained.size + ['untrained'] * cos_trained.size)
                dist['session'].extend([sess] * dist_tmp.size)
                dist['sn'].extend(sns)
                dist['roi'].extend([roi] * dist_tmp.size)
                dist['Hem'].extend([H] * dist_tmp.size)

        dist = pd.DataFrame(dist)
        dist.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dist.chord-session.glm{glm}.{H}.tsv'), sep='\t', index=False)