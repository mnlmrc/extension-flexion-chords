import nibabel as nb
import os

import pandas as pd

import globals as gl
import numpy as np
import pickle

import argparse

import nitools as nt
from nitools import spm

import time
import Functional_Fusion.atlas_map as am


def main(args):

    Hem = ['L', 'R']
    struct = ['CortexLeft', 'CortexRight']

    if args.what == 'save_timeseries_cifti':
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']

        SPM = spm.SpmGlm(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}'))  #
        SPM.get_info_from_spm_mat()

        for i, (s, H) in enumerate(zip(struct, Hem)):
            mask = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'day{args.day}',f'subj{args.sn}', f'Hem.{H}.nii')
            atlas = am.AtlasVolumetric(H, mask, structure=s)

            if i == 0:
                brain_axis = atlas.get_brain_model_axis()
            else:
                brain_axis += brain_axis

        coords = nt.affine_transform_mat(brain_axis.voxel.T, brain_axis.affine)

        # get raw time series in roi
        y_raw = nt.sample_images(SPM.rawdata_files, coords)

        # rerun glm
        _, info, y_filt, y_hat, y_adj, _ = SPM.rerun_glm(y_raw)

        row_axis = nb.cifti2.SeriesAxis(1, 1, y_filt.shape[0], 'second')

        save_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}',f'day{args.day}', f'subj{args.sn}')

        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))

        # save y_raw
        cifti = nb.Cifti2Image(
            dataobj=y_raw,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
        nb.save(cifti, save_path + '/' + 'y_raw.dtseries.nii')

        # save y_filt
        cifti = nb.Cifti2Image(
            dataobj=y_filt,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
        nb.save(cifti, save_path + '/' + 'y_filt.dtseries.nii')

        # save y_hat
        cifti = nb.Cifti2Image(
            dataobj=y_hat,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
        nb.save(cifti, save_path + '/' + 'y_hat.dtseries.nii')

        # save y_adj
        cifti = nb.Cifti2Image(
            dataobj=y_adj,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
        nb.save(cifti, save_path + '/' + 'y_adj.dtseries.nii')

    if args.what == 'save_betas_cifti_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_timeseries_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)



if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='efc4')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--day', type=int, default=1)
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108])
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()

    main(args)
    end = time.time()
    print(f'Finished in {end - start} seconds')


# # load residuals for prewhitening
# res_img = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', 'ResMS.nii'))
# ResMS = nt.sample_image(res_img, coords[0], coords[1], coords[2], 0)
#
# if stats == 'mean':
#     y_raw = data.mean(axis=1)
# elif stats == 'whiten':
#     y_raw = (data / np.sqrt(ResMS)).mean(axis=1)
# elif stats == 'pca':
#     pass
#
# fdata = SPM.spm_filter(SPM.weight @ data)
# beta = SPM.pinvX @ fdata
# pdata = SPM.design_matrix @ beta
#
#
#
#
#
