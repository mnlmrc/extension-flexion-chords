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
import Functional_Fusion.dataset as ds


def main(args):

    Hem = ['L', 'R']
    struct = ['CortexLeft', 'CortexRight']

    if args.what == 'save_timeseries_cifti':

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

    if args.what == 'save_timeseries_parcel':

        ydata = ['y_hat', 'y_adj', 'y_filt']

        for y in ydata:
            for i, (s, H) in enumerate(zip(struct, Hem)):
                atlas = am.AtlasVolumetric(args.atlas, os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'day{args.day}',
                                                                    f'subj{args.sn}', f'Hem.{H}.nii'), structure=s)
                mask = os.path.join(gl.baseDir, args.experiment, gl.roiDir,f'day{args.day}',
                                    f'subj{args.sn}', f'{args.atlas}.{H}.nii')
                if i==0:
                    label_vec, _ = atlas.get_parcel(mask)
                    parcel_axis = atlas.get_parcel_axis()
                else:
                    label_vec = np.concatenate((label_vec, atlas.get_parcel(mask)[0] + label_vec.max()), axis=0)
                    parcel_axis += atlas.get_parcel_axis()

            cifti = nb.load(os.path.join(gl.baseDir, args.experiment,
                                         f'{gl.glmDir}{args.glm}',f'day{args.day}', f'subj{args.sn}', f'{y}.dtseries.nii'))
            data = cifti.get_fdata()
            parcel_data, label = ds.agg_parcels(data, label_vec)

            row_axis = nb.cifti2.SeriesAxis(1, 1, parcel_data.shape[0], 'second')

            header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
            cifti_parcel = nb.Cifti2Image(parcel_data, header=header)

            print(f'saving {y} parcels...')
            nb.save(cifti_parcel, os.path.join(gl.baseDir, args.experiment,
                                        f'{gl.glmDir}{args.glm}', f'day{args.day}',f'subj{args.sn}',
                                        f'{args.atlas}.{y}.ptseries.nii'))

    if args.what == 'save_timeseries_cut':

        ydata = ['y_hat', 'y_adj', 'y_filt']

        TR = 1000
        nVols = 336
        dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'day{args.day}',f'subj{args.sn}',
                                       f'{args.experiment}_{args.sn}.dat'), sep='\t')
        pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
        runs = pinfo[pinfo['sn'] == args.sn].FuncRuns.reset_index(drop=True)[0].split('.')
        for i, BN in enumerate(dat['BN'].unique()):
            if str(BN) in runs:
                if i == 0:
                    at = (dat[dat['BN']==BN].startTRReal).tolist()
                else:
                    at.extend((dat[dat['BN']==BN].startTRReal + int(nVols * i)).tolist())
            else:
                print(f'excluding block {BN}')

        for y in ydata:
            for i, (s, H) in enumerate(zip(struct, Hem)):
                atlas = am.AtlasVolumetric(args.atlas, os.path.join(gl.baseDir, args.experiment, gl.roiDir,f'day{args.day}',
                                                                    f'subj{args.sn}', f'Hem.{H}.nii'), structure=s)
                mask = os.path.join(gl.baseDir, args.experiment, gl.roiDir,
                                    f'subj{args.sn}', f'{args.atlas}.{H}.nii')
                if i == 0:
                    label_vec, _ = atlas.get_parcel(mask)
                    parcel_axis = atlas.get_parcel_axis()
                else:
                    label_vec = np.concatenate((label_vec, atlas.get_parcel(mask)[0] + label_vec.max()), axis=0)
                    parcel_axis += atlas.get_parcel_axis()

            cifti = nb.load(os.path.join(gl.baseDir, args.experiment,
                                         f'{gl.glmDir}{args.glm}', f'day{args.day}',f'subj{args.sn}', f'{y}.dtseries.nii'))
            data = cifti.get_fdata()

            y_cut = spm.avg_cut(data, 10, at, 10)

            parcel_data, label = ds.agg_parcels(y_cut, label_vec)

            row_axis = nb.cifti2.SeriesAxis(-10, 10, parcel_data.shape[0], 'second')

            header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
            cifti_parcel = nb.Cifti2Image(parcel_data, header=header)

            print(f'saving {y} parcels...')
            nb.save(cifti_parcel, os.path.join(gl.baseDir, args.experiment,
                                        f'{gl.glmDir}{args.glm}',f'day{args.day}', f'subj{args.sn}',
                                        f'{args.atlas}.{y}.cut.ptseries.nii'))

    if args.what == 'save_timeseries_cifti_all':
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
