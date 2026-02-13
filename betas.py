import argparse

import pandas as pd
import numpy as np
import os
import scipy
import shutil
import subprocess

import globals as gl

import nibabel as nb
import nitools as nt
import imaging_pipelines.betas as bt
import time


def save_spm_as_mat7(glm, sn):
    # Define the path to the SPM.mat file
    spm_path = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'SPM.mat') #"/cifs/diedrichsen/data/Chord_exp/EFC_learningfMRI/glm1/subj101/SPM.mat"
    backup_path = spm_path + ".backup"

    # Step 1: Backup the original file
    shutil.copy(spm_path, backup_path)
    print(f"Backed up {spm_path} to {backup_path}")

    # Step 2: Run MATLAB command
    matlab_cmd = (
        f"matlab -nodesktop -nosplash -r "
        f"\"load('{spm_path}'); save('{spm_path}', '-struct', 'SPM', '-v7'); exit\""
    )

    # Execute the command
    subprocess.run(matlab_cmd, shell=True, check=True)
    print(f"Processed {spm_path} with MATLAB")

def main(args):
    sessions = [3, 9, 23]
    nSess = len(sessions)
    Hem = ['L', 'R']
    struct = ['CortexLeft', 'CortexRight']
    path_glm = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
    path_rois = os.path.join(gl.baseDir, gl.roiDir, f'subj{args.sn}')
    if args.what=='spm2mat7':
        save_spm_as_mat7(args.glm, args.sn)
    if args.what == 'spm2mat7_all':
        for sn in args.sns:
            print(f'doing participant {sn}')
            args = argparse.Namespace(
                what='spm2mat7',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'make_betas_cifti':
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in Hem]
        reginfo = pd.read_csv(os.path.join(path_glm, 'reginfo.tsv'), sep='\t')
        row_axis = nb.cifti2.ScalarAxis(reginfo['name'] + '.' + reginfo['run'].astype(str))
        cifti = bt.make_cifti_betas(masks, struct, path_glm=path_glm, row_axis=row_axis,)
        nb.save(cifti, path_glm + '/' + 'beta.dscalar.nii')
    if args.what == 'make_betas_cifti_all':
        for sn in args.sns:
            print(f'doing participant {sn}')
            args = argparse.Namespace(
                what='make_betas_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'make_residuals_cifti':
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in Hem]
        residuals = bt.make_cifti_residuals(path_glm=path_glm, masks=masks, struct=struct)
        nb.save(residuals, path_glm + '/' + 'residual.dtseries.nii')
    if args.what == 'make_residuals_cifti_all':
        for sn in args.sns:
            print(f'Processing subj{sn}...')
            arg = argparse.Namespace(
                what='make_residuals_cifti',
                sn=sn,
                glm=args.glm
            )
            main(arg)
    if args.what == 'make_intercept_cifti':
        reginfo = pd.read_csv(os.path.join(path_glm, 'reginfo.tsv'), sep='\t')
        day = reginfo.name.str.split(',', n=1, expand=True)[0]
        nRuns = [reginfo[day == d].run.nunique() for d in day.unique()]
        nRegressors = reginfo.shape[0]
        intercept = []
        for sess in range(nSess):
            for run in range(nRuns[sess]):
                intercept.append(os.path.join(path_glm, f'beta_0{nRegressors + run + 1 + sess * nRuns[0]}.nii'))
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in Hem]
        cond_vec = np.sort(np.array([f'{sess},{run}' for run in range(nRuns[sess]) for sess in range(nSess)]) )
        row_axis = nb.cifti2.ScalarAxis(cond_vec)
        cifti = bt.make_cifti_betas(masks, struct, intercept, row_axis=row_axis, )
        nb.save(cifti, path_glm + '/' + 'intercept.dscalar.nii')
    if args.what == 'make_intercept_cifti_sml':
        nSess = 4
        nRun = 8
        sns = [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        path = os.path.join(gl.baseDir, 'EFC_learningfMRI', 'SML',)
        for sn in sns:
            intercept = []
            for sess in range(nSess):
                for run in range(nRun):
                    intercept.append(os.path.join(path, f'glmSess{sess+1}', f's{sn:02d}', f'beta_{run+97:04d}.nii'))
            masks = [os.path.join(path, f'glmSess{sess+1}', f's{sn:02d}', f'Hem.{H}.nii') for H in Hem]
            cond_vec = np.sort(np.array([f'{sess},{run}' for run in range(nRun) for sess in range(nSess)]) )
            row_axis = nb.cifti2.ScalarAxis(cond_vec)
            cifti = bt.make_cifti_betas(masks, struct, intercept, row_axis=row_axis, )
            nb.save(cifti, os.path.join(path, f'glmSess1', f's{sn:02d}', 'intercept.dscalar.nii'))
    if args.what == 'make_intercept_cifti_all':
        for sn in args.sns:
            print(f'doing participant {sn}')
            args = argparse.Namespace(
                what='make_intercept_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'make_contrasts_cifti':
        path_glm = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        path_rois = os.path.join(gl.baseDir, gl.roiDir, f'subj{args.sn}')
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in Hem]
        regressors = pd.Series([f'{chordID},sess{sess:02d}' for sess in sessions for chordID in gl.chordID])
        cifti = bt.make_cifti_contrasts(path_glm, masks, struct, regressors)
        nb.save(cifti, path_glm + '/' + 'contrast.dscalar.nii')
    if args.what == 'make_contrasts_cifti_all':
        for sn in args.sns:
            args = argparse.Namespace(
                what='make_contrasts_cifti',
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'roi_contrasts':
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        con_dict = {
            'con': [],
            'chordID': [],
            'session': [],
            'chord': [],
            'sn': [],
            'roi': [],
            'Hem': []
        }
        for sn in args.sns:
            path_glm = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}', f'subj{sn}')
            path_rois = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
            cifti = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
            regr = cifti.header.get_axis(0).name
            vol = nt.volume_from_cifti(cifti)
            pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
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
                        con_dict['session'].append(day)
                        con_dict['sn'].append(sn)
                        con_dict['roi'].append(roi)
                        con_dict['Hem'].append(H)

        con_df = pd.DataFrame(con_dict)
        con_df.to_csv(os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}', 'ROI.con.avg.tsv'),
                      sep='\t',index=False)


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='EFC_learningfMRI')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105, 106, 107])
    parser.add_argument('--glm', type=int, default=3)

    args = parser.parse_args()
    main(args)
    finish = time.time()

    print(f'Time elapsed: {finish - start} seconds')
