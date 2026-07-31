import PcmPy as pcm
import os
import numpy as np
from EFC_learningfMRI.util import  get_trained_and_untrained, runs_to_keep
import EFC_learningfMRI.globals as gl
from imaging_pipelines.model import calc_prewhitened_betas
import nitools as nt
import nibabel as nb
import pandas as pd
import pickle

C = pcm.centering(8)

FINGER = {
     21911: np.array([1, 1, 0, 1, 1]),
     92122: np.array([0, 1, 1, 1, 1]),
     91211: np.array([0, 1, 1, 1, 1]),
     22911: np.array([1, 1, 0, 1, 1]),
     21291: np.array([1, 1, 1, 0, 1]),
     12129: np.array([1, 1, 1, 1, 0]),
     12291: np.array([1, 1, 1, 0, 1]),
     11911: np.array([1, 1, 0, 1, 1])
}

PATTERN = {
     21911: np.array([-1,  1,  0,  1,  1]),
     92122: np.array([ 0, -1,  1, -1, -1]),
     91211: np.array([ 0,  1, -1,  1,  1]),
     22911: np.array([-1, -1,  0,  1,  1]),
     21291: np.array([-1,  1, -1,  0,  1]),
     12129: np.array([ 1, -1,  1, -1,  0]),
     12291: np.array([ 1, -1, -1,  0,  1]),
     11911: np.array([ 1,  1,  0,  1,  1])
}

FLEXION = {
    21911: 1,
    92122: 3,
    91211: 1,
    22911: 2,
    21291: 2,
    12129: 2,
    12291: 2,
    11911: 0
}

def fixed_models():
    
    # trained untrained
    v_tr_untr = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    G_tr_untr = C @ np.outer(v_tr_untr, v_tr_untr)

    # trained
    tr = np.zeros(8)
    tr[:4] = 1
    G_tr = C @ np.diag(tr)

    # untrained
    untr = np.zeros(8)
    untr[4:] = 1
    G_untr = C @ np.diag(untr)

    return G_tr_untr, G_tr, G_untr

def subj_spec_models(order=None, glm=3):

    finger  = np.zeros((8, 5))
    pattern = np.zeros_like(finger)
    flexion = np.zeros(8)
    for i, ch in enumerate(order):
        flexion[i] = FLEXION[ch]
        finger[i]  = FINGER[ch]
        pattern[i] = PATTERN[ch]

    G_finger  = C @ (finger @ finger.T)
    G_pattern = C @ (pattern @ pattern.T)
    G_flexion = C @ np.outer(flexion, flexion)

    return G_finger, G_pattern, G_flexion

def make_models(sn):

    comp_names = ['type', 'trained', 'untrained', 'finger', 'pattern', 'flexion']

    G_tr_untr, G_tr, G_untr = fixed_models()
    order = np.array(get_trained_and_untrained(sn)).astype(int)
    G_finger, G_pattern, G_flexion = subj_spec_models(order=order)

    M = []
    M.append(pcm.FixedModel('null',      np.zeros((8, 8))))
    M.append(pcm.FixedModel('type',      G_tr_untr)) 
    M.append(pcm.FixedModel('trained',   G_tr)) 
    M.append(pcm.FixedModel('untrained', G_untr))
    M.append(pcm.FixedModel('finger',    G_finger))
    M.append(pcm.FixedModel('pattern',   G_pattern))
    M.append(pcm.FixedModel('flexion',   G_flexion))
    M.append(pcm.ComponentModel('component', np.array([G_tr_untr / np.trace(G_tr_untr),
                                                       G_tr      / np.trace(G_tr),
                                                       G_untr    / np.trace(G_untr),
                                                       G_finger  / np.trace(G_finger),
                                                       G_pattern / np.trace(G_pattern),
                                                       G_flexion / np.trace(G_flexion)
                                                      ]))) 
    M.append(pcm.FreeModel('ceil', 8)) 

    return M, comp_names


def fit_component_model(loader):

    glm   = loader.glm
    atlas = loader.atlas_name

    df     = pd.DataFrame()
    thetas = {}                       # (sn, Hem, roi, session) -> theta_in, pickled below
    for data in loader:
        for session in gl.sessions:
            keep = runs_to_keep(data.part_vec.size, session=session)

            betas    = data.betas[keep]
            cond_vec = data.cond_vec[keep]
            part_vec = data.part_vec[keep]

            # make dataset
            obs_des = {'cond_vec': cond_vec, 'part_vec': part_vec}
            Y       = pcm.dataset.Dataset(betas, obs_descriptors=obs_des)

            model, comp_names = make_models(data.sn)
            model = model[-2] # select component model

            G_session3 = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.within_session.3.glm{glm}.{data.Hem}.{data.roi}.npy'))
            G_session3 = G_session3.mean(axis=0)
            comp_names.append('base')
            model.Gc      = np.r_[model.Gc, G_session3[None, :, :]]
            model.n_param = model.Gc.shape[0]     # sync param count with the extended Gc

            _, theta_in = pcm.fit_model_individ(Y, model, fit_scale=False, verbose=True, fixed_effect='block')

            path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f'component_model.theta_in.{atlas}.glm{glm}.{session}.{data.Hem}.{data.roi}.p'), 'wb') as f:
                pickle.dump(theta_in, f)




    



    