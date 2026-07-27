import pandas as pd
import nibabel as nb
import os
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import get_trained_and_untrained, get_cond_part, Roi
from EFC_learningfMRI.betas import load_prewhitened_betas
import nitools as nt
from imaging_pipelines.model import calc_prewhitened_betas
import numpy as np
import pandas as pd
import PcmPy as pcm


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


def fit_pc1(data):

    C, V = data.shape

    theta = []
    for c in range(0, C, 2):
        r1       = data[c]
        r2       = data[c+1]
        R        = np.c_[r1, r2].T
        R        = R[:, ~np.isnan(R).any(axis=0)]
        G        = R @ R.T
        theta.append(0.5 * np.atan2(2*G[0, 1], G[0, 0] - G[1, 1]))

    return np.array(theta)


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


def _subject_covariance(sn, glm, H, roi, atlas_name='ROI'):

    path_glm = os.path.join(gl.baseDir, f'glm{glm}')

    betas_prewhitened = load_prewhitened_betas(sn, glm, Roi(roi, H, atlas_name), residual_fname='ResMS.nii')
    betas_prewhitened -= betas_prewhitened.mean(axis=1, keepdims=True)

    cond_vec, part_vec = get_cond_part(sn, glm, type='chord-session')

    cov, _ = pcm.est_G(betas_prewhitened, cond_vec, part_vec)

    return cov
    
def _parse_subject_variance(cov, sn):
    """Tidy the covariance diagonal into one row per (session, chord, rep).

    The ``cov`` conditions are ordered ``'sess,chord,rep'`` with ``rep`` the
    fastest-varying field, then ``chord``, then ``sess`` (slowest). Each diagonal
    entry is the variance of one condition. ``make_chord_mapping(..., 'chord-
    session')`` maps the trained chords to the first half of the chord indices and
    untrained to the second half, so the mapped index also indexes
    ``get_trained_and_untrained(sn)`` to recover the real chordID.
    """
    var     = np.diag(cov)
    n_cond  = var.size
    n_sess  = len(gl.sess_mapping)                 # 3 sessions
    n_rep   = 2
    n_chord = n_cond // (n_sess * n_rep)           # 8 chords per session

    sess, chord, rep = np.unravel_index(np.arange(n_cond), (n_sess, n_chord, n_rep))

    chordID  = np.array(get_trained_and_untrained(sn)).astype(int)   # indexed by mapped chord

    return pd.DataFrame({
        'session': [gl.sessions[s] for s in sess],
        'chordID': chordID[chord],
        'chord'  : np.where(chord < n_chord // 2, 'trained', 'untrained'),
        'rep'    : rep + 1,
        'var'    : var,
    })


def _parse_subject_covariance(cov, sn):
    """Tidy the rep1-rep2 covariance (first off-diagonal) into one row per (session, chord).

    Conditions are ordered ``'sess,chord,rep'`` with ``rep`` the fastest-varying
    field, so the two reps of each ``(sess, chord)`` sit in consecutive rows/cols.
    Their covariance is therefore the first super-diagonal at even offsets,
    ``cov[2k, 2k+1]``. This collapses the rep dimension, leaving one value per
    ``(session, chord)``. Chord indices map to trained/untrained and to the real
    chordID exactly as in ``_parse_subject_variance``.
    """
    covar   = np.diag(cov, 1)[::2]                  # cov[2k, 2k+1] = rep1-rep2 covariance
    n_pair  = covar.size
    n_sess  = len(gl.sess_mapping)                 # 3 sessions
    n_chord = n_pair // n_sess                      # 8 chords per session

    sess, chord = np.unravel_index(np.arange(n_pair), (n_sess, n_chord))

    chordID = np.array(get_trained_and_untrained(sn)).astype(int)   # indexed by mapped chord

    return pd.DataFrame({
        'session': [gl.sessions[s] for s in sess],
        'chordID': chordID[chord],
        'chord'  : np.where(chord < n_chord // 2, 'trained', 'untrained'),
        'cov'    : covar,
    })


def covariance(sns, glm, atlas_name='ROI'):

    rois = gl.rois[atlas_name]

    var_list, cov_list = [], []
    for H in gl.Hem:
        for roi in rois:
            for sn in sns:
                print(f'doing participant {sn}, {H}, {roi}...')
                cov           = _subject_covariance(sn, glm, H, roi, atlas_name)

                df_var        = _parse_subject_variance(cov, sn)
                df_var['sn']  = sn
                df_var['Hem'] = H
                df_var['roi'] = roi

                df_cov        = _parse_subject_covariance(cov, sn)
                df_cov['sn']  = sn
                df_cov['Hem'] = H
                df_cov['roi'] = roi
                var_list.append(df_var)
                cov_list.append(df_cov)

    df_var = pd.concat(var_list, ignore_index=True)
    df_cov = pd.concat(cov_list, ignore_index=True)

    df_var.to_csv(os.path.join(gl.baseDir, f'glm{glm}', f'{atlas_name}.variance.tsv'), index=False, sep='\t')
    df_cov.to_csv(os.path.join(gl.baseDir, f'glm{glm}', f'{atlas_name}.covariance.tsv'), index=False, sep='\t')


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

    path_glm = os.path.join(gl.baseDir, f'glm{glm}')

    df_list = []
    for sn in sns:
        cifti     = nb.load(os.path.join(path_glm, f'subj{sn}', fname))
        contrasts = nt.volume_from_cifti(cifti)
        residuals = nb.load(os.path.join(path_glm, f'subj{sn}', 'ResMS.nii'))
        parts     = _get_condition(cifti).str.split(',', expand=True)
        chordID   = parts[0].astype(int).values
        session   = parts[1].values
        trained   = np.array(get_trained_and_untrained(sn)[:4]).astype(int)
        chord     = np.where(np.isin(chordID, trained), 'trained', 'untrained')
        for H in gl.Hem:
            for roi in gl.rois[atlas_name]:
                print(f'doing participant {sn}, {H}, {roi}...')
                mask          = nb.load(Roi(roi, H, atlas_name).mask_path(sn))
                coords        = nt.get_mask_coords(mask)
                # contrasts_pw  = calc_prewhitened_betas(contrasts, residuals, mask)
                contrasts_roi = nt.sample_image(contrasts, coords[0], coords[1], coords[2], interpolation=0).T
                rep1          = contrasts_roi[::2]  # (n_pairs, n_voxels)
                rep2          = contrasts_roi[1::2]
                for sess in np.unique(session):
                    for grp in ('trained', 'untrained'):
                        sel       = (session == sess) & (chord == grp)
                        data      = np.c_[rep1[sel].mean(axis=0), rep2[sel].mean(axis=0)].T  # average across chords
                        theta     = fit_pc1(data)
                        slope_lin = fit_through_origin(data)
                        df_list.append(pd.DataFrame({
                            'sn'     : sn,
                            'Hem'    : H,
                            'roi'    : roi,
                            'chord'  : grp,
                            'session': sess,
                            'theta'  : theta,
                            'slope'  : slope_lin
                        }))

    df = pd.concat(df_list)
    df.session = df.session.map({f'sess{s:02d}': s for s in gl.sessions})
    df.to_csv(os.path.join(gl.baseDir, f'glm{glm}', f'{atlas_name}.rep_suppr_slope.tsv'), index=False, sep='\t')


if __name__=='__main__':
    sns = gl.participants
    glm = 2
    r2_v_r1_slope(sns, glm)
    #covariance(sns, glm)

    