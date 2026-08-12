import os
from collections import defaultdict
import numpy as np
import PcmPy as pcm
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.G_matrix import calc_G
from EFC_learningfMRI.betas import BetasPrewithenedLoader


def calc_G_rois(loader, sessions=('all', *gl.sessions), repetitions=('all', 1, 2)):
    """G of every ROI and session, with the subjects on the first dimension.

    Returns {(hemisphere, roi, session): (n_subj, n_cond, n_cond) array}, where
    n_cond is 8 within a session and 24 across sessions. Subjects come out in
    the order of ``loader.sns``.
    """
    glm = loader.glm
    
    for data in loader:
        for session in sessions:
            for repetition in repetitions:

                G        = calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=False)
                cov      = calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=True)
                G_raw    = calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=True, fixed_effect=False)
                G_noxval = calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=True, fixed_effect=False, crossval=False)

                save_path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
                os.makedirs(save_path, exist_ok=True)

                fname = 'across_session' if session == 'all' else 'within_session'
                fname = fname + '.' + str(session) if session != 'all' else fname
                fname = fname + '.' + str(repetition) if repetition != 'all' else fname

                np.save(os.path.join(save_path, f'G_obs.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G)
                np.save(os.path.join(save_path, f'cov.{fname}.glm{glm}.{data.Hem}.{data.roi}'), cov)
                np.save(os.path.join(save_path, f'G_obs_raw.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G_raw)
                np.save(os.path.join(save_path, f'G_obs_noxval.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G_noxval)


if __name__=='__main__':
    sns         = gl.participants
    glm         = 3
    sessions    = ('all', *gl.sessions)
    repetitions = ['all'] #, 1, 2]
    loader      = BetasPrewithenedLoader(sns=sns, glm=glm)

    calc_G_rois(loader, sessions, repetitions)
