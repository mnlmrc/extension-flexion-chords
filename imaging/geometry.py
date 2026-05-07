import PcmPy as pcm
import os
import argparse
import pandas as pd
import numpy as np
import imaging_pipelines.model as md
from imaging_pipelines.model import calc_prewhitened_betas
import nibabel as nb
import nitools as nt
import time
from util.util import get_trained_and_untrained
import AnatSearchlight.searchlight as sl
import globals.path as pth
import globals.imaging as im
import globals.design as dn


def searchlight_encoding(sns, glm):

    def _calc_D_searchlight(data, cond_vec=None, part_vec=None):
        data = data[:, ~np.isnan(data).any(axis=0)]
        G, _ = pcm.est_G_crossval(data, cond_vec, part_vec, X=pcm.indicator(part_vec))
        D = pcm.G_to_dist(G)
        D_trained = D[:4, :4]
        D_untrained = D[4:, 4:]
        return D.mean(), D_trained.mean(), D_untrained.mean()

    surf_path = os.path.join(pth.baseDir, pth.surfDir)
    n_session = 3
    for H in im.Hem:
        for n_sess in range(n_session):
            D, D_trained, D_untrained = [], [], []
            for sn in sns:
                print(f'starting participant {sn}, session {n_sess + 1}/{n_session}...')
                glm_path = os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}')
                roi_path = os.path.join(pth.baseDir, pth.roiDir, f'subj{sn}')

                # load searchlight
                print('loading searchlight...')
                SL = sl.load(os.path.join(roi_path, f'searchlight.{H}.h5'))
                    
                # load betas residuals
                print('loading and prewhitening betas...')
                beta_cifti = nb.load(os.path.join(glm_path, 'beta.dscalar.nii'))
                beta_vol = nt.volume_from_cifti(beta_cifti)
                res_vol = nb.load(os.path.join(glm_path, 'ResMS.nii'))
                beta_pw = beta_vol.get_fdata() / np.sqrt(res_vol.get_fdata()[:, :, :, None])

                # map regressors to integer for consistent order
                trained_untrained = get_trained_and_untrained(sn)
                regressor_mapping = {
                    f"{chordID},sess{sess:02d}": i
                    for i, (chordID, sess) in enumerate(
                        ((c, s) for s in dn.sessions for c in trained_untrained))}

                # load reginfo and select regressors for session
                reginfo = pd.read_csv(os.path.join(glm_path, 'reginfo.tsv'), sep='\t')
                cond_vec = reginfo.name.map(regressor_mapping).to_numpy()
                part_vec = reginfo.run.to_numpy()
                regr_interest = np.arange(n_sess * part_vec.size // 3, (n_sess + 1) * part_vec.size // 3)
                obs_des = {'cond_vec': cond_vec[regr_interest], 'part_vec': part_vec[regr_interest]}

                # run searchlight in parallel
                beta_pw_vol = nb.Nifti2Image(beta_pw[:, :, :, regr_interest], affine=beta_vol.affine, header=beta_vol.header)
                results = SL.run_parallel(beta_pw_vol, _calc_D_searchlight, obs_des, nargout=3)

                # store results
                D_tmp, D_trained_tmp, D_untrained_tmp = results[:, 0], results[:, 1], results[:, 2]
                D.append(D_tmp)
                D_trained.append(D_trained_tmp)
                D_untrained.append(D_untrained_tmp)

            # distance trained to gifti
            gifti = nt.make_func_gifti(np.array(D).T, anatomical_struct=SL.structure, column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding.session{dn.sessions[n_sess]}.{H}.func.gii'))

            # distance trained to gifti
            gifti = nt.make_func_gifti(np.array(D_trained).T, anatomical_struct=SL.structure, column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding-trained.session{dn.sessions[n_sess]}.{H}.func.gii'))

            # distance untrained to gifti
            gifti = nt.make_func_gifti(np.array(D_untrained).T, anatomical_struct=SL.structure, column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding-untrained.session{dn.sessions[n_sess]}.{H}.func.gii'))



def calc_G(sns, glm, rois, type='chord-session', sessions=None):
    path_glm = os.path.join(pth.baseDir, f'glm{glm}')
    path_rois = os.path.join(pth.baseDir, pth.roiDir)
    path_pcm = os.path.join(pth.baseDir, pth.pcmDir)
    for h, H in enumerate(im.Hem):
        for r, roi in enumerate(rois):
            G = []
            for s, sn in enumerate(sns):
                print(f'doing participant {sn}, {H}, {roi}...')
                reginfo = pd.read_csv(os.path.join(path_glm, f'subj{sn}', 'reginfo.tsv'), sep='\t')
                betas = nb.load(os.path.join(path_glm, f'subj{sn}', 'beta.dscalar.nii'))
                betas = nt.volume_from_cifti(betas)
                residuals = nb.load(os.path.join(path_glm, f'subj{sn}', f'ResMS.nii'))
                mask = nb.load(os.path.join(path_rois, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
                G_tmp = _calc_G_participant(betas, residuals, mask, reginfo, type=type, sessions=sessions)
                G.append(G_tmp)
            G = np.array(G)
            np.save(os.path.join(path_pcm, f'G_obs.{type}.glm{glm}.{H}.{roi}.npy'), G)


def _calc_G_participant(betas, residuals, mask, reginfo, type='set', sessions=None):

    # get trained chords
    sn = reginfo.sn.unique()[0]
    trained_untrained = np.array(get_trained_and_untrained(sn)).astype(int)
    label = [1, 1, 1, 1, 2, 2, 2, 2,] #['trained'] * 4 + ['untrained'] * 4
    if type == 'trained-untrained':
        chordID_mapping = dict(zip(trained_untrained, label))
    elif type == 'chord-session':
        chordID_mapping = dict(zip(trained_untrained, np.arange(8)))
    else:
        raise ValueError("Wrong type. Use 'trained-untrained' for trained vs. untrained and 'chord-session' for "
                         "individual chords in each session.")

    # make cond and part
    sess = reginfo.name.str.split(',', n=1, expand=True).loc[:, 1]
    sess = sess.map(dn.sess_mapping)
    chordID = reginfo.name.str.split(',', n=1, expand=True).loc[:, 0]
    chord = chordID.astype(int).map(chordID_mapping)
    part_vec = (reginfo.run % 10).to_numpy()
    cond_vec = sess.astype(str) + ',' + chord.astype(str)

    betas_prewhitened = calc_prewhitened_betas(betas, residuals, mask)
    if sessions is not None:
        G = np.zeros((len(sessions), 8, 8))
        for s, _ in enumerate(sessions):
            #sess = sess.to_numpy().astype(int)
            betas_prewhitened_s = betas_prewhitened[sess==s]
            part_vec_s = part_vec[sess==s]
            cond_vec_s = cond_vec[sess==s]
            G[s], _ = pcm.est_G_crossval(betas_prewhitened_s, cond_vec_s, part_vec_s, X=pcm.indicator(part_vec_s))
    else:
        G, _ = pcm.est_G_crossval(betas_prewhitened, cond_vec, part_vec, X=pcm.indicator(part_vec))

    return G

