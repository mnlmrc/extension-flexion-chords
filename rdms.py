import argparse

import globals as gl

import os
import pandas as pd
import numpy as np

import rsatoolbox as rsa
from imaging_pipelines import rdms

def calc_rdms_in_rois(sn, glm):
    Hem = ['L', 'R']
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    for H in Hem:
        for roi in rois:
            print(f'Hemisphere: {H}, region:{roi}')
            mask = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'{args.atlas}.{H}.{roi}.nii')
            beta_img = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'beta.dscalar.nii')
            res_img = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'ResMS.nii')
            rdm = rdms.calc_rdm_roi(mask, beta_img, res_img, Hem=H, roi=roi, sn=sn, unbalanced=False,
                                    runs_of_interest=args.runs_of_interest)
            save_dir = os.path.join(gl.baseDir, gl.rdmDir, f'subj{sn}')
            os.makedirs(save_dir, exist_ok=True)
            rdm.save(os.path.join(save_dir, f'glm{glm}.{H}.{roi}.hdf5'), overwrite=True, file_type='hdf5')


def main(args):

    if args.what == 'calc_rdms_in_rois':
        calc_rdms_in_rois(args.sn, args.glm)
    if args.what == 'calc_rdms_in_rois_all':
        for sn in args.sns:
            print(f'doing subj{sn}')
            args = argparse.Namespace(
                what='calc_rdms_in_rois',
                sn=sn,
                glm=args.glm,
            )
            main(args)
    if args.what == 'calc_rdms_force':
        days = [3, 9, 23]
        data = pd.DataFrame()
        for day in days:
            data_tmp = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, f'day{day}',
                                            f'efc4_{args.sn}_single_trial.tsv'), sep='\t')
            data_tmp['chordID'] = data_tmp['chordID'].astype(str)
            data_tmp['day'] = data_tmp['day'].astype(str)
            data_tmp = data_tmp.groupby(['chordID', 'day', 'BN']).mean(numeric_only=True).reset_index()
            data = pd.concat([data, data_tmp], ignore_index=True)

        conds = (data['day'] + ',' +  data['chordID']).to_numpy()
        run = data['BN'].to_numpy()
        data = data[['thumb', 'index', 'middle', 'ring', 'pinkie']].to_numpy()

        data = data[np.isin(run, args.runs_of_interest)]
        conds = conds[np.isin(run, args.runs_of_interest)]
        run = run[np.isin(run, args.runs_of_interest)]

        dataset = rsa.data.Dataset(
            data,
            channel_descriptors={
                'channel': ['thumb', 'index', 'middle', 'ring', 'pinkie']},
            obs_descriptors={'conds': conds,
                             'run': run},
        )

        rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run', remove_mean=False)
        save_dir = os.path.join(gl.baseDir, gl.rdmDir, f'subj{args.sn}')
        os.makedirs(save_dir, exist_ok=True)
        rdm.save(os.path.join(save_dir, f'force.hdf5'), overwrite=True, file_type='hdf5')

    if args.what == 'calc_dissimilarity':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        days = [3, 9, 23]

        pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')

        dissimilarity = {
            'day': [],
            'chord': [],
            'roi': [],
            'Hem': [],
            'dissimilarity': [],
            'sn': []
        }

        for sn in args.sns:
            chord = {
                'trained': pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.'),
                'untrained': pinfo[pinfo.sn == sn].reset_index()['untrained'][0].split('.')
            }
            for H in Hem:
                for roi in rois:
                    rdms = rsa.rdm.load_rdm(os.path.join(gl.baseDir, gl.rdmDir, f'subj{sn}',
                                                         f'glm{args.glm}.{H}.{roi}.hdf5'))
                    for day in days:
                        for ch in list(chord.keys()):
                            diss = rdms.subset_pattern('conds',
                                                       [f'{day:02d},{tr}' for tr in chord[ch]]).dissimilarities.mean()
                            dissimilarity['dissimilarity'].append(diss)
                            dissimilarity['chord'].append(ch)
                            dissimilarity['day'].append(day)
                            dissimilarity['roi'].append(roi)
                            dissimilarity['sn'].append(sn)
                            dissimilarity['Hem'].append(H)

        dissimilarity = pd.DataFrame(dissimilarity)

        dissimilarity.to_csv(os.path.join(gl.baseDir, gl.rdmDir, f'dissimilarity.tsv'), sep='\t', index=False)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument('--day', type=int, default=None)
    parser.add_argument('--runs_of_interest', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # parser.add_argument('--Hem', type=str, default=None)
    # parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()

    main(args)
