import PcmPy as pcm
import os
import argparse
import itertools
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import imaging_pipelines.model as md
from imaging_pipelines.model import calc_prewhitened_betas
import nibabel as nb
import nitools as nt
import time
from EFC_learningfMRI.util import get_trained_and_untrained, get_cond_part
import AnatSearchlight.searchlight as sl
import EFC_learningfMRI.globals as gl


def searchlight_encoding(sns, glm):

    def _calc_D_searchlight(data, cond_vec=None, part_vec=None):
        data = data[:, ~np.isnan(data).any(axis=0)]
        G, _ = pcm.est_G_crossval(data, cond_vec, part_vec, X=pcm.indicator(part_vec))
        D = pcm.G_to_dist(G)
        D_trained = D[:4, :4]
        D_untrained = D[4:, 4:]
        return D.mean(), D_trained.mean(), D_untrained.mean()

    surf_path = os.path.join(gl.baseDir, gl.surfDir)
    n_session = 3
    for H in gl.Hem:
        for n_sess in range(n_session):
            D, D_trained, D_untrained = [], [], []
            for sn in sns:
                print(f'starting participant {sn}, session {n_sess + 1}/{n_session}...')
                glm_path = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}')
                roi_path = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')

                # load searchlight
                print('loading searchlight...')
                SL = sl.load(os.path.join(roi_path, f'searchlight.{H}.h5'))
                    
                # load betas residuals
                print('loading and prewhitening betas...')
                beta_cifti = nb.load(os.path.join(glm_path, 'beta.dscalar.nii'))
                beta_vol = nt.volume_from_cifti(beta_cifti)
                res_vol = nb.load(os.path.join(glm_path, 'ResMS.nii'))
                beta_pw = beta_vol.get_fdata() / np.sqrt(res_vol.get_fdata()[:, :, :, None])

                # map regressors to integer for consistent order
                trained_untrained = get_trained_and_untrained(sn)
                regressor_mapping = {
                    f"{chordID},sess{sess:02d}": i
                    for i, (chordID, sess) in enumerate(
                        ((c, s) for s in gl.sessions for c in trained_untrained))}

                # load reginfo and select regressors for session
                reginfo = pd.read_csv(os.path.join(glm_path, 'reginfo.tsv'), sep='\t')
                cond_vec = reginfo.name.map(regressor_mapping).to_numpy()
                part_vec = reginfo.run.to_numpy()
                regr_interest = np.arange(n_sess * part_vec.size // 3, (n_sess + 1) * part_vec.size // 3)
                obs_des = {'cond_vec': cond_vec[regr_interest], 'part_vec': part_vec[regr_interest]}

                # run searchlight in parallel
                beta_pw_vol = nb.Nifti2Image(beta_pw[:, :, :, regr_interest], affine=beta_vol.affine, header=beta_vol.header)
                results = SL.run_parallel(beta_pw_vol, _calc_D_searchlight, obs_des, nargout=3)

                # store results
                D_tmp, D_trained_tmp, D_untrained_tmp = results[:, 0], results[:, 1], results[:, 2]
                D.append(D_tmp)
                D_trained.append(D_trained_tmp)
                D_untrained.append(D_untrained_tmp)

            # distance trained to gifti
            gifti = nt.make_func_gifti(np.array(D).T, anatomical_struct=SL.structure, column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding.glm{glm}.session{gl.sessions[n_sess]}.{H}.func.gii'))

            # distance trained to gifti
            gifti = nt.make_func_gifti(np.array(D_trained).T, anatomical_struct=SL.structure, column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding-trained.glm{glm}.session{gl.sessions[n_sess]}.{H}.func.gii'))

            # distance untrained to gifti
            gifti = nt.make_func_gifti(np.array(D_untrained).T, anatomical_struct=SL.structure, column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding-untrained.glm{glm}.session{gl.sessions[n_sess]}.{H}.func.gii'))


def rdm_vector(D):
    """Vectorise the upper triangle (off-diagonal) of one or more RDMs.

    Args:
        D (ndarray): a dissimilarity matrix of shape (n, n) or a stack of them
            with shape (..., n, n).

    Returns:
        ndarray: the flattened upper triangle. Shape ``(n_pairs,)`` for a single
            RDM or ``(..., n_pairs)`` for a stack, with ``n_pairs = n*(n-1)/2``.
    """
    n = D.shape[-1]
    iu = np.triu_indices(n, k=1)
    return D[..., iu[0], iu[1]]


def sqrt_mean_diag(G, n_trained=4):
    """Root-mean pattern magnitude for trained and untrained chords.

    The diagonal of a second-moment matrix ``G`` holds each condition's pattern
    variance (squared activation magnitude). This returns the square root of the
    mean diagonal separately for the trained chords (upper-left
    ``n_trained`` x ``n_trained`` block) and the untrained chords (the remaining
    lower-right block).

    Args:
        G (ndarray): second-moment matrix, shape (n, n) or a stack (..., n, n).
        n_trained (int): number of trained conditions (default 4).

    Returns:
        tuple[ndarray, ndarray]: ``(sqrt_trained, sqrt_untrained)``, each a scalar
            for a single ``G`` or shape ``(...)`` for a stack.
    """
    diag = np.diagonal(G, axis1=-2, axis2=-1)  # (..., n)
    return (np.sqrt(diag[..., :n_trained].mean(axis=-1)),
            np.sqrt(diag[..., n_trained:].mean(axis=-1)))


def avg_crossnobis(G, n_trained=4):
    """Mean crossnobis dissimilarity for trained and untrained chords.

    The pairwise crossnobis distances are derived from ``G`` and averaged over the
    off-diagonal (unique) pairs within the trained block (upper-left
    ``n_trained`` x ``n_trained``) and within the untrained block (lower-right).

    Args:
        G (ndarray): second-moment matrix, shape (n, n) or a stack (..., n, n).
        n_trained (int): number of trained conditions (default 4).

    Returns:
        tuple[ndarray, ndarray]: ``(d_trained, d_untrained)``, each a scalar for a
            single ``G`` or shape ``(...)`` for a stack.
    """
    D = pcm.G_to_dist(G)
    return (rdm_vector(D[..., :n_trained, :n_trained]).mean(axis=-1), # trained
            rdm_vector(D[..., n_trained:, n_trained:]).mean(axis=-1)) # untrained


def rescaled_crossnobis(G, G_ref, n_trained=4):
    """Rescale a new G's mean crossnobis dissimilarity by its pattern magnitude
    relative to a reference G.

    For the trained and untrained blocks separately, the scaling factor is the
    ratio of the root-mean diagonal (pattern magnitude) of the new ``G`` to that
    of the reference ``G_ref``. The new ``G``'s mean crossnobis dissimilarity is
    multiplied by this factor::

        est = (sqrt_mean_diag(G) / sqrt_mean_diag(G_ref)) * avg_crossnobis(G_ref)

    Args:
        G (ndarray): new second-moment matrix, shape (n, n) or a stack (..., n, n).
        G_ref (ndarray): reference second-moment matrix, broadcastable to ``G``.
        n_trained (int): number of trained conditions (default 4).

    Returns:
        tuple[ndarray, ndarray]: ``(est_trained, est_untrained)``, each a scalar
            for a single ``G`` or shape ``(...)`` for a stack.
    """
    sqrt_new_tr, sqrt_new_untr = sqrt_mean_diag(G, n_trained)
    sqrt_ref_tr, sqrt_ref_untr = sqrt_mean_diag(G_ref, n_trained)
    d_tr, d_untr = avg_crossnobis(G_ref, n_trained)
    return ((sqrt_new_tr / sqrt_ref_tr) * d_tr,
            (sqrt_new_untr / sqrt_ref_untr) * d_untr)


def _corr_sess(D):
    """Correlate the RDM between every pair of sessions for one participant.

    Args:
        D (ndarray): per-session dissimilarity matrices, shape (n_sess, n, n).
        sessions (list, optional): session labels; defaults to ``gl.sessions``.
        method (str): 'pearson' (default) or 'spearman'.

    Returns:
        list[dict]: one entry per session pair with keys ``session_x``,
            ``session_y`` and ``corr``.
    """
    sessions = gl.sessions
    vec = rdm_vector(D)  # (n_sess, n_pairs)
    rows = []
    for i, j in itertools.combinations(range(D.shape[0]), 2):
        rows.append({
            'session_x': sessions[i],
            'session_y': sessions[j],
            'corr': np.corrcoef(vec[i], vec[j])[0, 1]
        })
    return rows


def noise_ceiling(D_group, method='pearson'):
    """Lower and upper RSA noise ceiling for a group of RDMs (Nili et al. 2014).

    Each participant's RDM is correlated against the group-mean RDM. The upper
    bound uses the mean over all participants (including the participant itself,
    an overestimate); the lower bound uses a leave-one-out mean (an
    underestimate). Both are averaged across participants.

    Args:
        D_group (ndarray): RDMs across participants, shape (n_subj, n, n).
        method (str): 'pearson' (default) or 'spearman'.

    Returns:
        tuple[float, float]: ``(nc_low, nc_high)``.
    """
    vec = rdm_vector(D_group)  # (n_subj, n_pairs)
    n_subj = vec.shape[0]
    mean_rdm = vec.mean(axis=0)
    low, high = [], []
    for i in range(n_subj):
        high.append(np.corrcoef(vec[i], mean_rdm)[0, 1])
        loo = np.delete(vec, i, axis=0).mean(axis=0)
        low.append(np.corrcoef(vec[i], loo)[0, 1])
    return float(np.mean(low)), float(np.mean(high))


def session_geometry_correlations(sns=None, atlas_name='ROI'):
    """Between-session RDM correlations and noise ceiling for every ROI.

    For each hemisphere and ROI in ``atlas`` the per-session RDMs (from
    ``G_obs.{H}.{roi}.npy``) are correlated between sessions within each
    participant, and an across-participant noise ceiling is computed per session.
    Both are collected into a single tidy long-format dataframe.

    Args:
        atlas (str): key into ``gl.rois`` selecting the ROIs to analyse.
        sns (list, optional): participants; defaults to ``gl.participants``.
        sessions (list, optional): session labels; defaults to ``gl.sessions``.
        method (str): correlation type, 'pearson' (default) or 'spearman'.
        save (bool): if True, write the dataframe as a .tsv into ``gl.pcmDir``.

    Returns:
        pandas.DataFrame: long-format table with columns ``sn``, ``Hem``,
            ``roi``, ``session_x``, ``session_y``, ``measure`` and ``value``.
            ``measure`` is 'corr' for the per-participant between-session
            correlations and 'nc_low'/'nc_high' for the group noise ceiling
            (with ``sn`` = NaN and ``session_x`` = ``session_y`` = session).
    """
    sns = gl.participants if sns is None else sns

    path_pcm = os.path.join(gl.baseDir, gl.pcmDir)

    rows = []
    for H in gl.Hem:
        for roi in gl.rois[atlas_name]:

              # load every participant's RDMs, skipping any that are missing
            D_all = {}
            for sn in sns: 
                fpath     = os.path.join(path_pcm, f'subj{sn}', f'G_obs.{H}.{roi}.npy')
                G         = np.load(fpath)
                D_all[sn] = pcm.G_to_dist(G)

            # between-session correlations within each participant
            for sn, D in D_all.items():
                for r, r_tr, r_untr in zip(_corr_sess(D), _corr_sess(D[:, :4, :4]), _corr_sess(D[:, 4:, 4:])):
                    rows.append({'sn': sn, 'Hem': H, 'roi': roi,
                                 'session_x': r['session_x'], 'session_y': r['session_y'],
                                 'measure': 'corr', 'value': r['corr'], 'value_trained': r_tr['corr'], 'value_untrained': r_untr['corr']})

            # noise ceiling across participants, one per session
            D_stack = np.stack(list(D_all.values()))  # (n_subj, n_sess, n, n)
            for s, sess in enumerate(gl.sessions):
                nc_low     , nc_high      = noise_ceiling(D_stack[:, s])
                nc_low_tr  , nc_high_tr   = noise_ceiling(D_stack[:, s, :4, :4])
                nc_low_untr, nc_high_untr = noise_ceiling(D_stack[:, s, 4:, 4:])
                for measure, value, value_tr, value_untr in (
                        ('nc_low' , nc_low , nc_low_tr , nc_low_untr ),
                        ('nc_high', nc_high, nc_high_tr, nc_high_untr)):
                    rows.append({'sn': np.nan, 'Hem': H, 'roi': roi,
                                 'session_x': sess, 'session_y': sess,
                                 'measure': measure, 'value': value,
                                 'value_trained': value_tr, 'value_untrained': value_untr})

    df = pd.DataFrame(rows)
    fout = os.path.join(gl.baseDir, gl.pcmDir, f'session_geometry_corr.{atlas_name}.tsv')
    df.to_csv(fout, sep='\t', index=False)


def dissimilarity_prediction(sns=None, glm=3, atlas_name='ROI'):
    """Observed vs magnitude-predicted mean crossnobis dissimilarity per session.

    For every hemisphere, ROI and participant the per-session second-moment
    matrices are loaded from ``G_obs.{H}.{roi}.npy`` (first axis ordered as
    ``gl.sessions``, i.e. sessions 3, 9, 23). Session 3 (the first session) is the
    reference. For each session the *observed* mean crossnobis dissimilarity
    (:func:`avg_crossnobis`) is compared to the value *predicted* if the reference
    geometry had merely scaled with the change in pattern magnitude (root-mean
    diagonal) between that session and session 3 (:func:`rescaled_crossnobis`),
    separately for trained and untrained chords. A session whose observed value
    departs from its prediction has a geometry that changed beyond a simple
    magnitude rescaling. The reference session has ``predicted == observed``.

    Args:
        sns (list, optional): participants; defaults to ``gl.participants``.
        atlas_name (str): key into ``gl.rois`` selecting the ROIs (default 'ROI').

    Returns:
        pandas.DataFrame: long-format table with columns ``sn``, ``Hem``, ``roi``,
            ``session``, ``chord`` ('trained'/'untrained'), ``predicted`` and
            ``observed``.
    """
    sns = gl.participants if sns is None else sns
    path_pcm = os.path.join(gl.baseDir, gl.pcmDir)
    ref_idx = 0  # session 3 is the first session in gl.sessions

    rows = []
    for H in gl.Hem:
        for roi in gl.rois[atlas_name]:
            for sn in sns:
                fpath = os.path.join(path_pcm, f'subj{sn}', f'G_obs.{H}.{roi}.npy')
                G = np.load(fpath)  # (n_sess, n, n), 

                act_tr, act_untr = sqrt_mean_diag(G)
                obs_tr , obs_untr  = avg_crossnobis(G) # (n_sess,) each
                pred_tr, pred_untr = rescaled_crossnobis(G, G[ref_idx])

                for s, sess in enumerate(gl.sessions):
                    for chord, pred, obs, act in (
                            ('trained'  , pred_tr[s]  , obs_tr[s], act_tr[s]  ),
                            ('untrained', pred_untr[s], obs_untr[s], act_untr[s])):
                        rows.append({'sn': sn, 'Hem': H, 'roi': roi, 'activity': act, 'session': sess, 'chord': chord, 'predicted': pred, 'observed': obs})

    df = pd.DataFrame(rows)
    df['diff'] = df.observed - df.predicted
    fout = os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_prediction.{atlas_name}.glm{glm}.tsv')
    df.to_csv(fout, sep='\t', index=False)


def dissimilarity_prediction(sns=None, glm=3, atlas_name='ROI'):
    """Observed vs magnitude-predicted mean crossnobis dissimilarity per session.

    For every hemisphere, ROI and participant the per-session second-moment
    matrices are loaded from ``G_obs.{H}.{roi}.npy`` (first axis ordered as
    ``gl.sessions``, i.e. sessions 3, 9, 23). Session 3 (the first session) is the
    reference. For each session the *observed* mean crossnobis dissimilarity
    (:func:`avg_crossnobis`) is compared to the value *predicted* if the reference
    geometry had merely scaled with the change in pattern magnitude (root-mean
    diagonal) between that session and session 3 (:func:`rescaled_crossnobis`),
    separately for trained and untrained chords. A session whose observed value
    departs from its prediction has a geometry that changed beyond a simple
    magnitude rescaling. The reference session has ``predicted == observed``.

    Args:
        sns (list, optional): participants; defaults to ``gl.participants``.
        atlas_name (str): key into ``gl.rois`` selecting the ROIs (default 'ROI').

    Returns:
        pandas.DataFrame: long-format table with columns ``sn``, ``Hem``, ``roi``,
            ``session``, ``chord`` ('trained'/'untrained'), ``predicted`` and
            ``observed``.
    """
    sns = gl.participants if sns is None else sns
    path_pcm = os.path.join(gl.baseDir, gl.pcmDir)
    ref_idx = 0  # session 3 is the first session in gl.sessions

    rows = []
    for H in gl.Hem:
        for roi in gl.rois[atlas_name]:
            for sn in sns:
                fpath = os.path.join(path_pcm, f'subj{sn}', f'G_obs.{H}.{roi}.npy')
                G = np.load(fpath)  # (n_sess, n, n), 
                cnb_tr, cnb_untr = pcm.G_to_dist(G[:, :4, :4]), pcm.G_to_dist(G[:, 4:, 4:])

                act_tr, act_untr = sqrt_mean_diag(G)
                obs_tr , obs_untr  = avg_crossnobis(G) # (n_sess,) each
                pred_tr, pred_untr = rescaled_crossnobis(G, G[ref_idx])

                for s, sess in enumerate(gl.sessions):
                    for chord, pred, obs, act in (
                            ('trained'  , pred_tr[s]  , obs_tr[s], act_tr[s]  ),
                            ('untrained', pred_untr[s], obs_untr[s], act_untr[s])):
                        rows.append({'sn': sn, 'Hem': H, 'roi': roi, 'activity': act, 'session': sess, 'chord': chord, 'predicted': pred, 'observed': obs})

    df = pd.DataFrame(rows)
    df['diff'] = df.observed - df.predicted
    fout = os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_prediction.{atlas_name}.glm{glm}.tsv')
    df.to_csv(fout, sep='\t', index=False)

