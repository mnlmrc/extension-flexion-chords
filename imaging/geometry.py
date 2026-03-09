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
import globals.path as pth
import globals.imaging as im
import globals.design as dn

def searchlight_encoding(args):
    Hem = ['L', 'R']
    structnames = ['CortexLeft', 'CortexRight']
    glm_path = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}')
    cifti_img_name = 'beta.dscalar.nii'
    res_img_name = 'ResMS.nii'
    searchlight_path = os.path.join(gl.baseDir, gl.roiDir)
    surf_path = os.path.join(gl.baseDir, gl.surfDir)
    regressor_mapping = {
        f"sess{sess:02d},{chordID}": i
        for i, (sess, chordID) in enumerate(
            ((s, c) for s in gl.sessions for c in gl.chordID)
        )
    }
    for h, H in enumerate(Hem):
        SL = md.PcmSearchlight(
            cifti_img=[os.path.join(glm_path, f'subj{sn}', cifti_img_name) for sn in args.sns],
            res_img=[os.path.join(glm_path, f'subj{sn}', res_img_name) for sn in args.sns],
            searchlight_list=[os.path.join(searchlight_path, f'subj{sn}', f'searchlight.{H}.h5') for sn in args.sns],
            structnames=structnames[h],
            regressor_mapping=regressor_mapping,
            regr_interest=[0, 1, 2, 3, 4, 5, 6, 7],
            #n_jobs=args.n_jobs
        )
        #SL.n_centre = 2
        n_centre = SL.n_centre
        distance = np.full((n_centre, SL.N), np.nan)
        #SL._run_searchlight(0)
        G_obs = SL.run_seachlight_parallel()
        for c in range(SL.n_centre):
            G = G_obs[c]
            distance[c] = np.array([pcm.G_to_dist(G[s]).mean() for s in range(SL.N)])

        # distance to gifti
        data = distance
        gifti = nt.make_func_gifti(data, anatomical_struct=structnames[h], column_names=args.sns)
        nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding.session3.{H}.func.gii'))


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

