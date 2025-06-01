import argparse

import globals as gl

import os
import pandas as pd
import numpy as np

import rsatoolbox as rsa
from imaging_pipelines import rdms


def main(args):

    if args.what == 'save_rois_rdms':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:
                print(f'Hemisphere: {H}, region:{roi}')
                mask = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}',f'{args.atlas}.{H}.{roi}.nii')
                beta_img = os.path.join(gl.baseDir, args.experiment,  f'glm{args.glm}', f'subj{args.sn}','beta.dscalar.nii')
                res_img = os.path.join(gl.baseDir, args.experiment,  f'glm{args.glm}', f'subj{args.sn}','ResMS.nii')
                rdm = rdms.calc_rdm_roi(mask, beta_img, res_img, Hem=H, roi=roi, sn=args.sn, unbalanced=False)
                save_dir = os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'subj{args.sn}')
                os.makedirs(save_dir, exist_ok=True)
                rdm.save(os.path.join(save_dir, f'glm{args.glm}.{H}.{roi}.hdf5'), overwrite=True, file_type='hdf5')

    if args.what == 'save_force_rdms':
        days = [3, 9, 23]
        data = pd.DataFrame()
        for day in days:
            data_tmp = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'day{day}',
                                            f'{args.experiment}_{args.sn}_single_trial.tsv'), sep='\t')
            data_tmp['chordID'] = data_tmp['chordID'].astype(str)
            data_tmp['day'] = data_tmp['day'].astype(str)
            data_tmp = data_tmp.groupby(['chordID', 'day', 'BN']).mean(numeric_only=True).reset_index()
            data = pd.concat([data, data_tmp], ignore_index=True)

        conds = data['day'] + ',' +  data['chordID']
        run = data['BN']
        data = data[['thumb', 'index', 'middle', 'ring', 'pinkie']].to_numpy()

        dataset = rsa.data.Dataset(
            data,
            channel_descriptors={
                'channel': ['thumb', 'index', 'middle', 'ring', 'pinkie']},
            obs_descriptors={'conds': conds,
                             'run': run},
        )

        rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run', remove_mean=False)
        save_dir = os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'subj{args.sn}')
        os.makedirs(save_dir, exist_ok=True)
        rdm.save(os.path.join(save_dir, f'force.hdf5'), overwrite=True, file_type='hdf5')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--day', type=int, default=None)
    # parser.add_argument('--Hem', type=str, default=None)
    # parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    main(args)
