import globals.path as pth
import globals.imaging as im
import globals.design as dn
import pandas as pd
import numpy as np
import os
import PcmPy as pcm

if __name__=='__main__':

    rois = im.rois['ROI']

    for H in im.Hem:
        for glm in [1, ]:

            # trained-untrained
            dist = {'dist': [], 'cos': [], 'session': [], 'sn': [], 'roi': [], 'Hem': []}
            for r, roi in enumerate(rois):
                G = np.load(os.path.join(pth.baseDir, pth.pcmDir, f'G_obs.trained-untrained.glm{glm}.{H}.{roi}.npy'))
                D = pcm.G_to_dist(G)
                C = pcm.G_to_cosine(G)
                for I, i in enumerate(np.arange(1, 6, 2)):
                    dist_tmp = D[:, i-1, i]
                    cos_tmp = C[:, i-1, i]
                    dist['dist'].extend(dist_tmp)
                    dist['cos'].extend(cos_tmp)
                    dist['session'].extend(np.repeat(dn.sessions[I], dist_tmp.size))
                    dist['sn'].extend(np.arange(dist_tmp.size))
                    dist['roi'].extend([roi] * dist_tmp.size)
                    dist['Hem'].extend([H] * dist_tmp.size)
            dist = pd.DataFrame(dist)
            dist.to_csv(os.path.join(pth.baseDir, pth.pcmDir, f'dist.trained-untrained.glm{glm}.{H}.tsv'), sep='\t', index=False)

            # chord-session
            dist = {'dist': [], 'cos': [], 'chord': [], 'session': [], 'sn': [], 'roi': [], 'Hem': []}
            for r, roi in enumerate(rois):
                for s, sess in enumerate(dn.sessions):
                    G = np.load(os.path.join(pth.baseDir, pth.pcmDir, f'G_obs.chord.glm{glm}.{H}.{roi}.npy'))

                    C = pcm.G_to_cosine(G[:, s] )
                    cos_trained = C[:, :4, :4].mean(axis=(1, 2))
                    cos_untrained = C[:, 4:, 4:].mean(axis=(1, 2))
                    cos_tmp = np.r_[cos_trained, cos_untrained]

                    D = pcm.G_to_dist(G[:, s] )
                    dist_trained = D[:, :4, :4].mean(axis=(1, 2))
                    dist_untrained = D[:, 4:, 4:].mean(axis=(1, 2))
                    dist_tmp = np.r_[dist_trained, dist_untrained]

                    dist['dist'].extend(dist_tmp)
                    dist['cos'].extend(cos_tmp)
                    dist['chord'].extend(['trained'] * cos_trained.size + ['untrained'] * cos_trained.size)
                    dist['session'].extend([sess] * dist_tmp.size)
                    dist['sn'].extend(np.arange(dist_tmp.size))
                    dist['roi'].extend([roi] * dist_tmp.size)
                    dist['Hem'].extend([H] * dist_tmp.size)

            dist = pd.DataFrame(dist)
            dist.to_csv(os.path.join(pth.baseDir, pth.pcmDir, f'dist.chord-session.glm{glm}.{H}.tsv'), sep='\t', index=False)