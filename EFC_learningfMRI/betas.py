import argparse

import pandas as pd
import numpy as np
import os
import shutil
import subprocess
from SUITPy.atlas import summarize_data

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import add_chord_column

import nibabel as nb
import nitools as nt
import imaging_pipelines.betas as bt
import nitools.spm as spm


def _parse_regressor_name(name, sep=','):
    """Split a contrast regressor name into its component condition labels.

    Regressor names are ``sep``-delimited condition labels, e.g.
    ``'chordID,sess'`` or ``'chordID,sess,rep'`` with the default ``sep=','``.

    Parameters
    ----------
    name : pandas.Series of str
        The regressor names to parse.
    sep : str, optional
        Field separator (default ``','``).

    Returns
    -------
    tuple of pandas.Series
        One ``Series`` per delimited component, in order. The tuple has as many
        elements as the number of parts found, so it can be unpacked directly,
        e.g. ``chordID, sess = _parse_regressor_name(regressor)``.
    """
    parts = name.str.split(sep, expand=True)
    return tuple(parts[col] for col in parts.columns)
    

def roi_avg(sns=None, glm=None, atlas_name='ROI', cond_names=['chordID', 'session'], fname='contrast.dscalar.nii'):
    df = pd.DataFrame()
    for sn in sns:
        for H in gl.Hem:
            print(f'doing participant {sn}, {H}...')

            # load contrasts
            cifti = nb.load(os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', fname))
            vol = nt.volume_from_cifti(cifti)

            # load rois
            masks = nb.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'{atlas_name}.{H}.nii'))

            # do summarize
            tmp = summarize_data(vol, label_image=masks, 
                                 region_names=['S1', 'M1', 'PMd', 'PMv', 'SMA', 'V1', 'SPLa', 'SPLp'])

            # add condition descriptors
            name = cifti.header.get_axis(0).name
            regressor = pd.Series(np.repeat(name, tmp.region.nunique()))
            cond = _parse_regressor_name(regressor)
            for c, c_name in enumerate(cond_names):tmp[c_name] = cond[c]

            # add subject, Hem and trained/untrained descriptors
            tmp['sn'] = sn
            tmp['Hem'] = H
            tmp = add_chord_column(tmp)

            tmp['session'] = tmp.session.map({'sess03': 3, 'sess09': 9, 'sess23': 23})

            df = pd.concat([df, tmp])

    df.to_csv(os.path.join(gl.baseDir, f'glm{glm}', f'{atlas_name}.activation.tsv'), sep='\t', index=False)


def save_spm_as_mat7(sn, glm):
    path_glm = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}')
    spm_path = os.path.join(path_glm, 'SPM.mat')
    backup_path = spm_path + ".backup"

    if os.path.exists(backup_path):
        pass
    else:
        # Step 1: Backup the original file
        shutil.copy(spm_path, backup_path)
        print(f"Backed up {spm_path} to {backup_path}")

    # Step 2: Run MATLAB command
    matlab_cmd = (
        f"matlab -nodesktop -nosplash -r "
        f"\"load('{spm_path}'); save('{spm_path}', '-struct', 'SPM', '-v7'); exit\""
    )

    subprocess.run(matlab_cmd, shell=True, check=True)
    print(f"Processed {spm_path} with MATLAB")


def make_cifti_cortex(sn, glm=None, type='beta', session=None):
    print(f'doing participant {sn}, {type}...')
    path_glm = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}',
                                f'day{session}') if session is not None else os.path.join(
                                gl.baseDir, f'glm{glm}', f'subj{sn}')
    path_rois = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in gl.Hem]
    reginfo = pd.read_csv(os.path.join(path_glm, 'reginfo.tsv'), sep='\t')
    row_axis = nb.cifti2.ScalarAxis(reginfo['name'] + '.' + reginfo['run'].astype(str))
    if type == 'beta':
        cifti = bt.make_cifti_betas(masks, gl.struct_cortex, path_glm=path_glm, row_axis=row_axis, )
        nb.save(cifti, path_glm + '/' + 'beta.dscalar.nii')
    elif type == 'repetition_suppression':
        cifti          = bt.make_cifti_contrasts(path_glm, masks, gl.struct_cortex, reginfo.name)
        brain_axis     = cifti.header.get_axis(1)
        regr           = pd.Series(cifti.header.get_axis(0).name)[::2]
        chord_sess_rep = regr.str.split(',', expand=True)
        row_axis       = chord_sess_rep.astype(str)[0] + ',' + chord_sess_rep[1]
        row_axis       = nb.cifti2.ScalarAxis(row_axis)
        data           = cifti.get_fdata()
        rep1           = data[::2]
        rep2           = data[1::2]
        suppr          = rep2 / rep1
        header         = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti_suppr    = nb.Cifti2Image(dataobj=suppr,  header=header)
        nb.save(cifti_suppr, path_glm + '/' + 'rep_suppr.dscalar.nii')
    elif type == 'residual':
        residuals = bt.make_cifti_residuals(path_glm=path_glm, masks=masks, struct=gl.struct_cortex)
        nb.save(residuals, path_glm + '/' + 'residual.dtseries.nii')
    elif type == 'contrast':
        cifti = bt.make_cifti_contrasts(path_glm, masks, gl.struct_cortex, reginfo.name)
        nb.save(cifti, path_glm + '/' + 'contrast.dscalar.nii')
    elif type =='psc':
        contrast = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
        intercept = nb.load(path_glm + '/' + 'intercept.dscalar.nii')
        SPM = spm.SpmGlm(path_glm)
        SPM.get_info_from_spm_mat()
        cifti = bt.make_cifti_psc(contrast=contrast, intercept=intercept, SPM=SPM, masks=masks, struct=gl.struct_cortex)
        nb.save(cifti, path_glm + '/' + 'psc.dscalar.nii')
    elif type == 'intercept':
        session = reginfo.name.str.split(',', n=1, expand=True)[1]
        nRuns = [reginfo[session == sess].run.nunique() for sess in session.unique()]
        nRegressors = reginfo.shape[0]
        intercept = []
        for sess in range(gl.nSess):
            for run in range(nRuns[sess]):
                intercept.append(os.path.join(path_glm, f'beta_0{nRegressors + run + 1 + sess * nRuns[0]}.nii'))
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in gl.Hem]
        cond_vec = np.sort(np.array([f'{sess},{run}' for run in range(nRuns[sess]) for sess in range(gl.nSess)]))
        row_axis = nb.cifti2.ScalarAxis(cond_vec)
        cifti = bt.make_cifti_betas(masks, gl.struct_cortex, intercept, row_axis=row_axis, )
        nb.save(cifti, path_glm + '/' + 'intercept.dscalar.nii')
    else:
        raise Exception(f'Unknown type {type}. Must be beta, residual, contrast or intercept.')
