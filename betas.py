import argparse

import pandas as pd
import numpy as np
import os
import scipy

import globals as gl

import nibabel as nb
import nitools as nt


def get_roi(experiment=None, sn=None, Hem=None, roi=None, day=None):
    mat = scipy.io.loadmat(os.path.join(gl.baseDir, experiment, gl.roiDir, f'day{day}',f'subj{sn}',
                                        f'subj{sn}_ROI_region.mat'))
    R_cell = mat['R'][0]
    R = list()
    for r in R_cell:
        R.append({field: r[field].item() for field in r.dtype.names})

    # find roi
    R = R[[True if (r['name'].size > 0) and (r['name'] == roi) and (r['hem'] == Hem)
           else False for r in R].index(True)]

    return R


def get_roi_betas(experiment=None, sn=None, Hem=None, roi=None, glm=None, day=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'day{day}', f'subj{sn}',
                                       f'reginfo.tsv'), sep="\t")

    R = get_roi(experiment, sn, Hem, roi, day)

    betas = list()
    for n_regr in np.arange(0, reginfo.shape[0]):
        print(f'ROI.{Hem}.{roi} - loading regressor #{n_regr + 1}')

        vol = nb.load(
            os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'day{day}', f'subj{sn}', f'beta_{n_regr + 1:04d}.nii'))
        beta = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
        betas.append(beta)

    betas = np.array(betas)

    return betas


def get_roi_ResMS(experiment=None, sn=None, Hem=None, roi=None, glm=None, day=None):
    R = get_roi(experiment, sn, Hem, roi, day)

    ResMS = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'day{day}', f'subj{sn}', 'ResMS.nii'))
    res = nt.sample_image(ResMS, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)

    return res


def get_roi_contrasts(experiment=None, sn=None, Hem=None, roi=None, glm=None, day=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'day{day}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    regressors = reginfo['name'].unique()

    R = get_roi(experiment, sn, Hem, roi)

    contrasts = list()
    for regr, regressor in enumerate(regressors):
        vol = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'day{day}', f'subj{sn}',
                                   f'con_{regressor.replace(" ", "")}.nii'))
        con = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
        contrasts.append(con)

    contrasts = np.array(contrasts)

    return contrasts


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--day', type=int, default=None)
    parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    if args.what == 'save_roi_contrasts':
        contrasts = get_roi_contrasts(
            experiment=args.experiment,
            sn=args.sn,
            Hem=args.Hem,
            roi=args.roi,
            glm=args.glm,
            day=args.day
        )
        np.save(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}',
                             f'ROI.{args.Hem}.{args.roi}.con.npy'), contrasts)
    elif args.what == 'save_rois_contrasts':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:
                print(f'Hemisphere: {H}, region:{roi}')
                contrasts = get_roi_contrasts(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    roi=roi,
                    glm=args.glm,
                    day=args.day
                )
                np.save(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}',
                                     f'ROI.{H}.{roi}.con.npy'), contrasts)
    elif args.what == 'save_roi_betas':
        betas = get_roi_betas(
            experiment=args.experiment,
            sn=args.sn,
            Hem=args.Hem,
            roi=args.roi,
            glm=args.glm,
            day=args.day
        )
        np.save(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}',
                             f'ROI.{args.Hem}.{args.roi}.beta.npy'), betas)
    elif args.what == 'save_rois_betas':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:
                print(f'Hemisphere: {H}, region:{roi}')
                betas = get_roi_betas(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    roi=roi,
                    glm=args.glm,
                    day=args.day
                )
                np.save(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}',
                                     f'ROI.{H}.{roi}.beta.npy'), betas)
    elif args.what == 'save_rois_ResMS':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:
                print(f'Hemisphere: {H}, region:{roi}')
                res = get_roi_ResMS(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    roi=roi,
                    glm=args.glm,
                    day=args.day
                )
                np.save(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}',
                                     f'ROI.{H}.{roi}.res.npy'), res)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

