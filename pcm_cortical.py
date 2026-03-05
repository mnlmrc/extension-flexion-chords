import argparse
import os
import pickle
import warnings

import PcmPy as pcm
from analysis import globals as gl
import nibabel as nb
import nitools as nt
import numpy as np
import time
from util import get_trained_and_untrained
from imaging_pipelines import model as md


warnings.filterwarnings("ignore")

def pcm_rois(M, epoch, args):
    Hem = ['L', 'R']
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    cifti_img = 'beta.dscalar.nii'
    roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    subj_ids = [f'subj{sn}' for sn in args.sns]
    PCM = md.PcmRois(subj_ids, M, glm_path, cifti_img,
             roi_path=roi_path,
             roi_imgs=roi_imgs,
             regressor_mapping=gl.regressor_mapping,
             regr_interest=[0, 1, 2, 3, 4] if epoch == 'plan' else [5, 6, 7, 8, 9, 10, 11, 12,],
             res_img='residual.dtseries.nii',
             n_jobs=args.n_jobs)
    res = PCM.run_parallel_pcm_across_rois()

    for H in Hem:
        for roi in rois:
            r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')

            path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
            os.makedirs(path, exist_ok=True)

            res['T_in'][r].to_pickle(os.path.join(path, f'T_in.{epoch}.glm{args.glm}.{H}.{roi}.p'))
            res['T_cv'][r].to_pickle(os.path.join(path, f'T_cv.{epoch}.glm{args.glm}.{H}.{roi}.p'))
            res['T_gr'][r].to_pickle(os.path.join(path, f'T_gr.{epoch}.glm{args.glm}.{H}.{roi}.p'))

            np.save(os.path.join(path, f'G_obs.{epoch}.glm{args.glm}.{H}.{roi}.npy'), res['G_obs'][r])

            f = open(os.path.join(path, f'theta_in.{epoch}.glm{args.glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res['theta_in'][r], f)
            f = open(os.path.join(path, f'theta_cv.{epoch}.glm{args.glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res['theta_cv'][r], f)
            f = open(os.path.join(path, f'theta_gr.{epoch}.glm{args.glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res['theta_gr'][r], f)

def pcm_searchlight_sess(M, n_sess, args):
    Hem = ['L',]
    structnames = ['CortexLeft', 'CortexRight']
    glm_path = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}')
    cifti_img_name = 'beta.dscalar.nii'
    res_img_name = 'ResMS.nii'
    searchlight_path = os.path.join(gl.baseDir, gl.roiDir)
    surf_path = os.path.join(gl.baseDir, gl.surfDir)
    regr_interest = np.arange(8) + 8 * n_sess
    session = list(gl.sess_mapping.values())[n_sess]
    regressor_mapping = []
    for sn in args.sns:
        trained_untrained = get_trained_and_untrained(sn)
        regressor_mapping.append({
            f"{chordID},sess{sess:02d}": i
            for i, (chordID, sess) in enumerate(
                ((c, s) for s in gl.sessions for c in trained_untrained))})
    print(f'Using {args.n_jobs} CPUs')
    for h, H in enumerate(Hem):
        SL = md.PcmSearchlight(M=M,
            cifti_img=[os.path.join(glm_path, f'subj{sn}', cifti_img_name) for sn in args.sns],
            res_img=[os.path.join(glm_path, f'subj{sn}', res_img_name) for sn in args.sns],
            searchlight_list=[os.path.join(searchlight_path, f'subj{sn}', f'searchlight.{H}.h5') for sn in args.sns],
            structnames=structnames[h],
            regressor_mapping=regressor_mapping,
            regr_interest=regr_interest,
            n_jobs=args.n_jobs)
        n_centre = SL.n_centre
        Mc = M[0]
        distance = np.full((n_centre, SL.N), np.nan)
        param_c = np.full((n_centre, Mc.n_param), np.nan)
        var_tot = np.full((n_centre, SL.N), np.nan)
        #SL._run_searchlight(0)
        #SL.n_centre = 100
        G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr, good = SL.run_seachlight_parallel()
        for c in range(SL.n_centre):
            G = G_obs[c]
            var_tot[c] = np.trace(G, axis1=1, axis2=2)
            distance[c] = np.array([pcm.G_to_dist(G[s]).mean() for s in range(SL.N)])
            if good[c]:
                param_c[c] = theta_gr[c][0][:Mc.n_param]

        var_expl = np.exp(param_c)

        # trace to gifti
        data = var_tot
        gifti = nt.make_func_gifti(data, anatomical_struct=structnames[h], column_names=args.sns)
        nb.save(gifti, os.path.join(surf_path, f'searchlight.var_tot.session{session}.{H}.func.gii'))

        # distance to gifti
        data = distance
        gifti = nt.make_func_gifti(data, anatomical_struct=structnames[h], column_names=args.sns)
        nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding.session{session}.{H}.func.gii'))

        # var_expl to gifti
        data = var_expl
        column_names = ['trained_untrained', 'chordID']
        gifti = nt.make_func_gifti(data, anatomical_struct=structnames[h], column_names=column_names)
        nb.save(gifti, os.path.join(surf_path, f'searchlight.var_expl.session{session}.{H}.func.gii'))

def main(args):
    if args.what == 'searchlight_trained_untrained':
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.trained_untrained.p'), "rb")
        M = pickle.load(f)
        for n_sess in [1]:
            pcm_searchlight_sess(M, n_sess, args)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105, 106, 107,])
    parser.add_argument('--atlas', type=str, default='ROI')
    # parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=3)
    parser.add_argument('--n_jobs', type=int, default=10)

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')