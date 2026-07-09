import PcmPy as pcm
import os
import numpy as np
from EFC_learningfMRI.util import get_cond_part, make_chord_mapping, get_trained_and_untrained
import EFC_learningfMRI.globals as gl
from imaging_pipelines.model import calc_prewhitened_betas
import nitools as nt
import nibabel as nb
import pandas as pd

C = pcm.centering(8)

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

def subj_spec_models(sn):

    trained_untrained = np.array(get_trained_and_untrained(sn)).astype(int)

    chord_finger = np.zeros((8, 5))
    chord_pattern = np.zeros_like(chord_finger)
    for i, ch in enumerate(trained_untrained):
        chord_finger[i] = gl.chord_finger[ch]
        chord_pattern[i] = gl.chord_pattern[ch]

    G_finger = C @ (chord_finger @ chord_finger.T)

    G_pattern = C @ (chord_pattern @ chord_pattern.T)

    return G_finger, G_pattern

def make_models(sn):

    comp_names = ['type', 'trained', 'untrained', 'finger', 'pattern']

    G_tr_untr, G_tr, G_untr = fixed_models()
    G_finger, G_pattern = subj_spec_models(sn)

    M = []
    M.append(pcm.FixedModel('null',      np.zeros((8, 8))))
    M.append(pcm.FixedModel('type',      G_tr_untr)) 
    M.append(pcm.FixedModel('trained',   G_tr)) 
    M.append(pcm.FixedModel('untrained', G_untr))
    M.append(pcm.FixedModel('finger',    G_finger))
    M.append(pcm.FixedModel('pattern',   G_pattern))
    M.append(pcm.ComponentModel('component', np.array([G_tr_untr / np.trace(G_tr_untr),
                                                       G_tr      / np.trace(G_tr),
                                                       G_untr    / np.trace(G_untr),
                                                       G_finger  / np.trace(G_finger),
                                                       G_pattern / np.trace(G_pattern),
                                                      ])))  # 4
    M.append(pcm.FreeModel('ceil', 8))  # 5

    return M, comp_names


def fit_component_model(glm, atlas):
    path_glm = os.path.join(gl.baseDir, f'glm{glm}')
    path_rois = os.path.join(gl.baseDir, gl.roiDir)
    path_pcm = os.path.join(gl.baseDir, gl.pcmDir)

    df = pd.DataFrame()
    for sn in gl.participants:
        print(f'starting participant {sn}...')

          # load betas and residuals
        betas     = nb.load(os.path.join(path_glm, f'subj{sn}', 'beta.dscalar.nii'))
        betas     = nt.volume_from_cifti(betas)
        residuals = nb.load(os.path.join(path_glm, f'subj{sn}', 'residual.dtseries.nii'))

          # make cond and part vec
        part_vec, cond_vec = get_cond_part(sn, glm, type='chord-session')
        
          # make pcm model
        M, comp_names = make_models(sn)
        Mc            = M[-2]

          # define save dir for subj G matrix
        saveDir_G = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
        os.makedirs(saveDir_G, exist_ok=True)
        
        for H in gl.Hem:
            for roi in gl.rois[atlas]:
                
                  # prewhiten betas in ROI
                mask              = nb.load(os.path.join(path_rois, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
                betas_prewhitened = calc_prewhitened_betas(betas, residuals, mask)

                nRuns = betas_prewhitened.shape[0] // 3

                G = np.zeros((3, 8, 8))
                for n_sess, sess in enumerate(gl.sessions): 

                        # select session rows
                    rows      = np.arange(nRuns) + nRuns * n_sess
                    beta_sess = betas_prewhitened[rows]
                    cond_sess = cond_vec[rows]
                    part_sess = part_vec[rows]

                        # make dataset
                    obs_des = {
                        'cond_vec': cond_sess,
                        'part_vec': part_sess
                    }
                    Y        = pcm.dataset.Dataset(beta_sess, obs_descriptors=obs_des)
                    G_tmp, _ = pcm.est_G_crossval(Y.measurements, 
                                           Y.obs_descriptors['cond_vec'], 
                                           Y.obs_descriptors['part_vec'], 
                                           X=pcm.indicator(Y.obs_descriptors['part_vec']))
                    G[n_sess] = G_tmp

                        # fit_model
                    _, theta_in = pcm.fit_model_individ(Y, Mc, fit_scale=False, verbose=True, fixed_effect='block')

                    weight = np.asarray(np.exp(theta_in)).ravel()

                        # component order matches make_models(); a trailing param is the noise scale
                    if len(weight) > len(comp_names): 
                        labels = comp_names + ['noise'] * (len(weight) - len(comp_names))
                    else: 
                        labels = comp_names[:len(weight)]

                    df_tmp = pd.DataFrame({
                        'weight'   : weight,
                        'component': labels,
                    })
                    df_tmp['sn']   = sn
                    df_tmp['Hem']  = H
                    df_tmp['roi']  = roi
                    df_tmp['sess'] = sess
                    df             = pd.concat([df, df_tmp], ignore_index=True)

                np.save(os.path.join(saveDir_G, f'G_obs.glm{glm}.{H}.{roi}.npy'), G)

    df.to_csv(os.path.join(path_pcm, f'component_model.{atlas}.glm{glm}.tsv'), sep='\t', index=False)


if __name__=='__main__':
    glm = 2
    atlas = 'ROI'
    fit_component_model(glm, atlas)


    