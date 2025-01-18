import argparse

import globals as gl

import os
import pandas as pd
import numpy as np

import rsatoolbox as rsa


def calc_rdm_roi(experiment=None, sn=None, Hem=None, roi=None, glm=None, day=None):
    print(f'region:{roi}, hemisphere:{Hem}')

    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'day{day}', f'subj{sn}',
                                       f'day{day}_subj{sn}_reginfo.tsv'), sep="\t")

    betas = np.load(
        os.path.join(gl.baseDir, experiment, gl.glmDir + glm, f'day{day}', f'subj{sn}', f'ROI.{Hem}.{roi}.beta.npy'))
    res = np.load(
        os.path.join(gl.baseDir, experiment, gl.glmDir + glm, f'day{day}', f'subj{sn}', f'ROI.{Hem}.{roi}.res.npy'))
    betas_prewhitened = betas / np.sqrt(res)

    betas_prewhitened = np.array(betas_prewhitened)
    dataset = rsa.data.Dataset(
        betas_prewhitened,
        channel_descriptors={
            'channel': np.array(['vox_' + str(x) for x in range(betas_prewhitened.shape[-1])])},
        obs_descriptors={'conds': reginfo.name,
                         'run': reginfo.run})
    rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run')
    rdm.rdm_descriptors = {'roi': [roi], 'hem': [Hem], 'index': [0]}

    return rdm


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--day', type=int, default=None)
    # parser.add_argument('--Hem', type=str, default=None)
    # parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    if args.what == 'save_rois_rdms':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:
                print(f'Hemisphere: {H}, region:{roi}')
                rdm = calc_rdm_roi(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    roi=roi,
                    glm=args.glm,
                    day=args.day
                )
                rdm.save(os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'day{args.day}', f'subj{args.sn}',
                                      f'glm{args.glm}.{Hem}.{roi}.hdf5'), overwrite=True, file_type='hdf5')
