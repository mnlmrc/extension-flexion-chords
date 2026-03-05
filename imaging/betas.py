import argparse

import pandas as pd
import numpy as np
import os
import shutil
import subprocess

import globals.path as pth
import globals.imaging as im
import globals.design as dn

import nibabel as nb
import nitools as nt
import imaging_pipelines.betas as bt
import nitools.spm as spm
import time


def roi_contrasts(sns, atlas_name='ROI', glm=3):
    rois = im.rois[atlas_name]
    con_dict = {
        'con': [],
        #'psc': [],
        'chordID': [],
        'session': [],
        'chord': [],
        'sn': [],
        'roi': [],
        'Hem': [],
        'rep': [] if glm == 2 else None
    }
    for sn in sns:
        path_glm = os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}')
        path_rois = os.path.join(pth.baseDir, pth.roiDir, f'subj{sn}')
        con_cifti = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
        #psc_cifti = nb.load(path_glm + '/' + 'psc.dscalar.nii')
        con_vol = nt.volume_from_cifti(con_cifti)
        #psc_vol = nt.volume_from_cifti(psc_cifti)
        regr = con_cifti.header.get_axis(0).name
        pinfo = pd.read_csv(os.path.join(pth.baseDir, 'participants.tsv'), sep='\t')
        for H in im.Hem:
            for roi in rois:
                print(f'doing participant {sn}, {H}, {roi}...')
                mask = os.path.join(path_rois, f'{atlas_name}.{H}.{roi}.nii')
                coords = nt.get_mask_coords(mask)
                con = nt.sample_image(con_vol, coords[0], coords[1], coords[2], 0)
                con = np.nanmean(con, axis=0)
                # psc = nt.sample_image(psc_vol, coords[0], coords[1], coords[2], 0)
                # psc = np.nanmean(psc, axis=0)
                for i, c in enumerate(con):
                    if glm == 2:
                        chordID, sess, rep = regr[i].split(',')
                    elif (glm == 1) | (glm == 3):
                        chordID, sess = regr[i].split(',')
                    sess = int(sess[-2:])
                    if chordID in pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.'):
                        chord = 'trained'
                    else:
                        chord = 'untrained'
                    con_dict['con'].append(c)
                    #con_dict['psc'].append(p)
                    con_dict['chordID'].append(chordID)
                    con_dict['chord'].append(chord)
                    con_dict['session'].append(sess)
                    con_dict['sn'].append(sn)
                    con_dict['roi'].append(roi)
                    con_dict['Hem'].append(H)
                    con_dict['rep'].append(rep) if glm == 2 else None

    con_df = pd.DataFrame(con_dict)
    con_df.to_csv(os.path.join(pth.baseDir, f'glm{glm}', f'{atlas_name}.con.avg.tsv'), sep='\t', index=False)


def save_spm_as_mat7(sn, glm):
    path_glm = os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}')
    spm_path = os.path.join(path_glm, 'SPM.mat') #"/cifs/diedrichsen/data/Chord_exp/EFC_learningfMRI/glm1/subj101/SPM.mat"
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


def make_cifti(sn, glm=None, type='beta'):
    print(f'doing participant {sn}, {type}...')
    path_glm = os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}')
    path_rois = os.path.join(pth.baseDir, pth.roiDir, f'subj{sn}')
    masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in im.Hem]
    reginfo = pd.read_csv(os.path.join(path_glm, 'reginfo.tsv'), sep='\t')
    row_axis = nb.cifti2.ScalarAxis(reginfo['name'] + '.' + reginfo['run'].astype(str))
    if type == 'beta':
        cifti = bt.make_cifti_betas(masks, im.struct, path_glm=path_glm, row_axis=row_axis, )
        nb.save(cifti, path_glm + '/' + 'beta.dscalar.nii')
    elif type == 'residual':
        residuals = bt.make_cifti_residuals(path_glm=path_glm, masks=masks, struct=im.struct)
        nb.save(residuals, path_glm + '/' + 'residual.dtseries.nii')
    elif type == 'contrast':
        cifti = bt.make_cifti_contrasts(path_glm, masks, im.struct, reginfo.name)
        nb.save(cifti, path_glm + '/' + 'contrast.dscalar.nii')
    elif type =='psc':
        contrast = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
        intercept = nb.load(path_glm + '/' + 'intercept.dscalar.nii')
        SPM = spm.SpmGlm(path_glm)
        SPM.get_info_from_spm_mat()
        cifti = bt.make_cifti_psc(contrast=contrast, intercept=intercept, SPM=SPM, masks=masks, struct=im.struct)
        nb.save(cifti, path_glm + '/' + 'psc.dscalar.nii')
    elif type == 'intercept':
        session = reginfo.name.str.split(',', n=1, expand=True)[1]
        nRuns = [reginfo[session == sess].run.nunique() for sess in session.unique()]
        nRegressors = reginfo.shape[0]
        intercept = []
        for sess in range(dn.nSess):
            for run in range(nRuns[sess]):
                intercept.append(os.path.join(path_glm, f'beta_0{nRegressors + run + 1 + sess * nRuns[0]}.nii'))
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in im.Hem]
        cond_vec = np.sort(np.array([f'{sess},{run}' for run in range(nRuns[sess]) for sess in range(dn.nSess)]))
        row_axis = nb.cifti2.ScalarAxis(cond_vec)
        cifti = bt.make_cifti_betas(masks, im.struct, intercept, row_axis=row_axis, )
        nb.save(cifti, path_glm + '/' + 'intercept.dscalar.nii')
    else:
        raise Exception(f'Unknown type {type}. Must be beta, residual, contrast or intercept.')
