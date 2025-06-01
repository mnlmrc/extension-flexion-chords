import argparse

import pandas as pd
import numpy as np
import os
import scipy

import globals as gl

import nibabel as nb
import nitools as nt
import imaging_pipelines.betas as bt
import time


def main(args):

    Hem = ['L', 'R']
    struct = ['CortexLeft', 'CortexRight']

    if args.what == 'save_betas_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        path_rois = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}')
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in Hem]
        reginfo = pd.read_csv(os.path.join(path_glm, 'reginfo.tsv'), sep='\t')
        row_axis = reginfo['name'] + '.' + reginfo['run'].astype(str)
        cifti = bt.make_cifti_betas(path_glm, masks, struct, row_axis=row_axis,)
        nb.save(cifti, path_glm + '/' + 'beta.dscalar.nii')
    if args.what == 'save_contrasts_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        path_rois = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}')
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in Hem]
        days = [3, 9, 23]
        regressors = [f'{day},{chordID}' for day in days for chordID in gl.chordID]
        cifti = bt.make_cifti_contrasts(path_glm, masks, struct, regressors)
        nb.save(cifti, path_glm + '/' + 'contrast.dscalar.nii')
    if args.what == 'save_contrasts_cifti_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_contrasts_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'save_contrasts_roi':
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        con_dict = {
            'con': [],
            'chordID': [],
            'day': [],
            'chord': [],
            'sn': [],
            'roi': [],
            'Hem': []
        }
        for sn in args.snS:
            path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{sn}')
            path_rois = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}')
            cifti = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
            regr = cifti.header.get_axis(0).name
            vol = nt.volume_from_cifti(cifti)
            pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
            for H in Hem:
                for roi in rois:
                    mask = os.path.join(path_rois, f'ROI.{H}.{roi}.nii')
                    coords = nt.get_mask_coords(mask)
                    con = nt.sample_image(vol, coords[0], coords[1], coords[2], 0)
                    con = np.nanmean(con, axis=0)
                    for i, c in enumerate(con):
                        day, chordID = regr[i].split(',')
                        if chordID in pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.'):
                            chord = 'trained'
                        else:
                            chord = 'untrained'
                        con_dict['con'].append(c)
                        con_dict['chordID'].append(chordID)
                        con_dict['chord'].append(chord)
                        con_dict['day'].append(day)
                        con_dict['sn'].append(sn)
                        con_dict['roi'].append(roi)
                        con_dict['Hem'].append(H)

        con_df = pd.DataFrame(con_dict)
        con_df.to_csv(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', 'ROI.con.avg.tsv'),
                      sep='\t',index=False)


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[101])
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()
    main(args)
    finish = time.time()

    print(f'Time elapsed: {finish - start} seconds')
