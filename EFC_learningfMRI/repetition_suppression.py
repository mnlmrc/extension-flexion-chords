import pandas as pd
import nibabel as nb
import os
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import add_chord_column
import nitools as nt
import numpy as np
import pandas as pd


def fit_through_origin(data, axis=-1):
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

    data = np.asarray(data, dtype=float)
    x, y = data[::2], data[1::2]
    # n = min(len(x), len(y))
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


def _get_condition(cifti, sep=','):
    """Labels for consecutive row pairs, dropping the last (rep) field.

    Takes the even-indexed row names and strips the trailing ``sep``-delimited
    field (the repetition index), so e.g. ``'chordID,sess,rep'`` collapses to
    ``'chordID,sess'`` — one label per pair produced by
    ``fit_origin_consecutive``.
    """
    names = pd.Series(cifti.header.get_axis(0).name)[::2].reset_index(drop=True)
    parts = names.str.split(sep, expand=True)
    return parts.iloc[:, :-1].astype(str).agg(sep.join, axis=1)


def r2_v_r1_slope(sns, glm=None, fname='contrast.dscalar.nii', atlas_name='ROI'):
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
        Columns ``['sn', 'Hem', 'roi', 'name', 'slope']``, one row per
        consecutive pair for every subject / hemisphere / ROI combination.
    """

    df_list = []
    for sn in sns:
        cifti = nb.load(os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', fname))
        vol   = nt.volume_from_cifti(cifti)
        parts   = _get_condition(cifti).str.split(',', expand=True)
        chordID = parts[0].values
        session = parts[1].values
        for H in gl.Hem:
            for roi in gl.rois[atlas_name]:
                mask    = nb.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'{atlas_name}.{H}.{roi}.nii'))
                coords  = nt.get_mask_coords(mask)
                vol_roi = nt.sample_image(vol, coords[0], coords[1], coords[2], interpolation=0)
                slope   = fit_through_origin(vol_roi.T)
                tmp = pd.DataFrame({
                    'sn': sn,
                    'Hem': H,
                    'roi': roi,
                    'chordID': chordID,
                    'session': session,
                    'slope': slope,
                })
                tmp = add_chord_column(tmp)
                df_list.append(tmp)

    df = pd.concat(df_list)
    df.session = df.session.map({f'sess{s:02d}': s for s in gl.sessions})
    df.to_csv(os.path.join(gl.baseDir, f'glm{glm}', f'{atlas_name}.rep_suppr_slope.tsv'), index=False, sep='\t')









