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

            

def load_contrasts(sn, glm=None, fname='contrast.dscalar.nii'):
    """Load a subject's contrast CIFTI image for a given GLM.

    Parameters
    ----------
    sn : int or str
        Subject number, used to build the ``subj{sn}`` path.
    glm : int or str
        GLM number, used to build the ``glm{glm}`` path.
    fname : str, optional
        CIFTI file name within the subject's GLM directory
        (default ``'contrast.dscalar.nii'``).

    Returns
    -------
    nibabel.Cifti2Image
        The loaded contrast image. Rows are contrast regressors (see
        ``_parse_regressor_name``); columns are cortical grayordinates.
    """
    return nb.load(os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', fname))


def fit_origin(x, y, axis=-1):
    """Least-squares slope of ``y = beta * x`` forced through the origin.

    Computes the closed-form no-intercept OLS estimate
    ``beta = sum(x * y) / sum(x * x)`` along ``axis``. Any element where either
    ``x`` or ``y`` is not finite is dropped from both sums, so pairs with
    partial coverage still contribute their valid voxels.

    Parameters
    ----------
    x, y : array_like
        Predictor and response, broadcastable to a common shape. The fit is
        taken across ``axis`` (e.g. voxels), independently for every other
        index.
    axis : int, optional
        Axis to reduce over (default ``-1``).

    Returns
    -------
    numpy.ndarray
        The through-origin slope(s), with ``axis`` removed. Entries whose
        denominator is zero (or which have no finite data) are ``NaN``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    num = np.sum(xv * yv, axis=axis)
    den = np.sum(xv * xv, axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        beta = np.where(den > 0, num / den, np.nan)
    return beta


def fit_origin_consecutive(data):
    """Through-origin slope across voxels for each consecutive row pair.

    Rows of ``data`` are grouped into non-overlapping pairs ``(0, 1)``,
    ``(2, 3)``, ... — matching the rep1/rep2 layout used elsewhere (see
    ``make_cifti_cortex(..., type='repetition_suppression')``). Within each
    pair the even row is the predictor ``x`` (e.g. first presentation) and the
    odd row the response ``y`` (e.g. second presentation), and a single slope
    ``y = beta * x`` is fit across voxels.

    Parameters
    ----------
    data : array_like, shape (nRows, nVoxels)
        Contrast estimates. ``nRows`` should be even; a trailing unpaired row
        is ignored.

    Returns
    -------
    numpy.ndarray, shape (nRows // 2,)
        One through-origin slope per pair. ``beta < 1`` indicates repetition
        suppression, ``beta > 1`` enhancement.
    """
    data = np.asarray(data, dtype=float)
    x, y = data[::2], data[1::2]
    n = min(len(x), len(y))
    return fit_origin(x[:n], y[:n], axis=1)


def _pair_row_labels(cifti, sep=','):
    """Labels for consecutive row pairs, dropping the last (rep) field.

    Takes the even-indexed row names and strips the trailing ``sep``-delimited
    field (the repetition index), so e.g. ``'chordID,sess,rep'`` collapses to
    ``'chordID,sess'`` — one label per pair produced by
    ``fit_origin_consecutive``.
    """
    names = pd.Series(cifti.header.get_axis(0).name)[::2].reset_index(drop=True)
    parts = names.str.split(sep, expand=True)
    return parts.iloc[:, :-1].astype(str).agg(sep.join, axis=1)


def slopes_consecutive(sn, glm=None, fname='contrast.dscalar.nii'):
    """Load a subject's contrasts and fit through-origin slopes per row pair.

    Convenience wrapper that composes ``load_contrasts`` and
    ``fit_origin_consecutive``: for every consecutive (e.g. rep1 vs. rep2) row
    pair it returns the slope of ``rep2 = beta * rep1`` fit across all cortical
    voxels.

    Parameters
    ----------
    sn : int or str
        Subject number.
    glm : int or str
        GLM number.
    fname : str, optional
        Contrast CIFTI file name (default ``'contrast.dscalar.nii'``).

    Returns
    -------
    pandas.DataFrame
        Columns ``['sn', 'name', 'slope']``, one row per consecutive pair.
    """
    cifti = load_contrasts(sn, glm=glm, fname=fname)
    slope = fit_origin_consecutive(cifti.get_fdata())
    return pd.DataFrame({'sn': sn, 'name': _pair_row_labels(cifti).values, 'slope': slope})


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
