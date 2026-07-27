import PcmPy as pcm
import os
import argparse
import itertools
import pandas as pd
import numpy as np
from scipy.stats import rankdata, chi2
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import imaging_pipelines.model as md
from imaging_pipelines.model import calc_prewhitened_betas
import nibabel as nb
import nitools as nt
import time
from EFC_learningfMRI.util import get_trained_and_untrained
from EFC_learningfMRI.betas import load_betas, load_residuals, load_roi_mask
import AnatSearchlight.searchlight as sl
import EFC_learningfMRI.globals as gl


def runs_to_keep(session, totRuns):

        session_dict = {3: 0, 9: 1, 23: 2}

        if session=='all':
            keep = np.ones(totRuns, dtype=bool)
        else:
            session                                   = session_dict[session]
            nRuns                                     = totRuns // 3
            keep                                      = np.zeros(totRuns, dtype=bool)
            keep[session * nRuns:(session + 1)*nRuns] = True
        return keep


def calc_G(data, cond_vec, part_vec, session='all', centred=False):
    """
    calc G matrix for runs in session

    ``centred`` removes each regressor's mean across voxels. It returns a new
    array rather than centring in place, so the same betas can be reused for
    another session.
    """
    if centred:
        data = data - data.mean(axis=1, keepdims=True)

    keep     = runs_to_keep(session, part_vec.size)
    G_obs, _ = pcm.est_G_crossval(data[keep],
                                  cond_vec[keep],
                                  part_vec[keep],
                                  X=pcm.indicator(part_vec[keep]))
    return G_obs


def split_trained(M):
    """Mean of the trained (first 4) and untrained (last 4) condition blocks."""
    mask4 = np.tri(4, k=-1, dtype=bool)
    mask8 = np.tri(8, k=-1, dtype=bool)
    if M.ndim==2:
        trained   = M[:4, :4][mask4]
        untrained = M[4:, 4:][mask4]
        tot       = M[mask8].mean()
    elif M.ndim==3:
        trained   = M[:, :4, :4][:, mask4]
        untrained = M[:, 4:, 4:][:, mask4]
        tot       = M[:, mask8].mean(axis=1)
    return tot, trained, untrained


def _calc_avg_dissimilarity(data, cond_vec, part_vec, session='all'):
    """Default searchlight metric: crossnobis distance, overall / trained / untrained.

    Returns three scalars (mean distance over all chord pairs, over the trained
    block and over the untrained block), so it pairs with three ``metric_labels``.
    """
    data  = data[:, ~np.isnan(data).any(axis=0)]
    G_obs = calc_G(data, cond_vec, part_vec, session)
    D     = pcm.G_to_dist(G_obs)
    tot, trained, untrained = split_trained(D)
    return tot, trained, untrained


def searchlight_fname(glm, H, label, session):
    """Path of the surface maps saved by :class:`Searchlight`."""
    return os.path.join(gl.baseDir, gl.surfDir,
                        f'searchlight.{label}.within_session.{gl.sessions[session]}.glm{glm}.{H}.func.gii')


class Searchlight():

    def __init__(self, sns=None, glm=None, residual_fname='ResMS.nii', metric_fn=_calc_avg_dissimilarity,
                 metric_labels=('encoding', 'encoding-trained', 'encoding-untrained'), sessions=(0, 1, 2)):
        self.sns            = gl.participants if sns is None else sns
        self.glm            = glm
        self.residual_fname = residual_fname
        self.metric_fn      = metric_fn
        self.metric_labels  = metric_labels
        self.sessions       = sessions





    def _load_betas(self, sn):
        """Prewhitened betas of one subject, as a volume."""

        path_glm   = os.path.join(gl.baseDir, f'glm{self.glm}', f'subj{sn}')
        beta_cifti = nb.load(os.path.join(path_glm, 'beta.dscalar.nii'))
        beta_vol   = nt.volume_from_cifti(beta_cifti)
        res_vol    = nb.load(os.path.join(path_glm, self.residual_fname))
        beta_pw    = beta_vol.get_fdata() / np.sqrt(res_vol.get_fdata()[:, :, :, None])

        return beta_pw, beta_vol


    def _searchlight_subject(self, sn, H):
        """Run the metric over the searchlight of one subject, once per session.

        The searchlight and the betas are loaded and prewhitened once and reused
        for every session, which only selects its own runs (see runs_to_keep).
        Returns the searchlight and {session: (n_centers, n_out) array}.
        """

        print('loading searchlight...')
        SL = sl.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'searchlight.{H}.h5'))

        print('loading and prewhitening betas...')
        beta_pw, beta_vol  = self._load_betas(sn)
        cond_vec, part_vec = get_cond_part(sn, self.glm)

        n_out   = len(self.metric_labels)
        results = {}
        for session in self.sessions:
            print(f'running searchlight, session {gl.sessions[session]}...')

            keep    = runs_to_keep(session, part_vec.size)
            obs_des = {'cond_vec': cond_vec[keep], 'part_vec': part_vec[keep]}

            beta_pw_vol = nb.Nifti2Image(beta_pw[:, :, :, keep], affine=beta_vol.affine, header=beta_vol.header)

            result = np.asarray(SL.run_parallel(beta_pw_vol, self.metric_fn, obs_des, nargout=n_out))

            # run_parallel returns (n_centers,) when n_out==1
            results[session] = result[:, None] if result.ndim==1 else result

        return results


    def run(self):
        """Run the searchlight for every subject, and save one surface map per metric and session."""

        for H in gl.Hem:
            maps = {session: [] for session in self.sessions}
            for sn in self.sns:
                print(f'doing participant {sn}, {H}')

                results = self._searchlight_subject(sn, H)
                for session, result in results.items():
                    maps[session].append(result)

            for session, result in maps.items():
                result = np.array(result)
                for i, fname in enumerate(self.output_fname):
                    gifti = nt.make_func_gifti(result[:, :, i].T, 
                                               anatomical_struct='CortexLeft' if H=='L' else 'CortexRight', 
                                               column_names=self.sns)
                    nb.save(gifti, os.path.join(self.output_folder, fname))


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


def G_sorted(sn, H, roi, order=None):
    """Reindex a subject's G so its rows/cols follow `order`, a list of the 8
    chord IDs, instead of that subject's own trained-first order.

    With the default order (np.sort(gl.chordID)) the returned matrix is aligned
    by *chord identity*, so entry [s, a, b] is the second moment between the same
    physical chords order[a], order[b] in every subject -- i.e. the pairwise
    second moment between each pair of chords. Returns (nSess, 8, 8)."""
    if order is None:
        order = np.sort(gl.chordID)
    order   = np.asarray(order, dtype=int)
    G       = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs.{H}.{roi}.npy'))
    tr_untr = np.array(get_trained_and_untrained(sn), dtype=int)
    pos     = {c: i for i, c in enumerate(tr_untr)}                                    # chord -> position in this G
    idx     = [pos[c] for c in order]
    return G[:, idx][:, :, idx]


def G_sorted_mean(sns=None, H='L', roi='M1', order=None, exclude=None, symmetrize=True):
    """Average the chord-indexed second-moment matrices across subjects.

    `exclude` drops one subject (leave-one-out), so an artificial G can be built
    purely from the *other* subjects. `symmetrize` averages the two triangles so
    the result is a proper (symmetric) second moment. Returns
    (nSess, 8, 8) in `order` (default np.sort(gl.chordID))."""
    if sns is None:
        sns = gl.participants
    use = [s for s in sns if s != exclude]
    G   = np.mean([G_sorted(s, H, roi, order) for s in use], axis=0)
    
    return G

def G_reliability(sns=None, atlas_name='ROI', include_diag=True):
    """Leave-one-out cross-subject reliability of the second-moment matrix G.

    Each participant's G is aligned by chord identity (:func:`G_sorted`, canonical
    order) so the same physical chords line up across people. For every
    hemisphere, ROI and session, a participant's vectorised G is correlated
    (Pearson) with the mean G of the *other* participants (leave-one-out). A high
    value means the chord-by-chord second moment is consistent across
    participants; it is an upper bound on how well any model can predict a single
    subject's G. Mirrors the LOO lower bound in :func:`noise_ceiling`, but on G
    (incl. its diagonal, the pattern variances) rather than on the RDM.

    Args:
        sns (list, optional): participants; defaults to ``gl.participants``.
        atlas_name (str): key into ``gl.rois`` selecting the ROIs (default 'ROI').
        include_diag (bool): include the diagonal (pattern variances) in the
            vectorised G; if False, only the off-diagonal covariances are used.

    Returns:
        pandas.DataFrame: long-format table with columns ``sn`` (NaN for the group
            mean over participants), ``Hem``, ``roi``, ``session`` and
            ``reliability``.
    """
    sns = gl.participants if sns is None else sns
    order = np.sort(gl.chordID)                     # canonical chord-identity order
    k = 0 if include_diag else 1

    rows = []
    for H in gl.Hem:
        for roi in gl.rois[atlas_name]:

            # chord-aligned G for every participant: (n_subj, n_sess, 8, 8)
            G_all  = np.stack([G_sorted(sn, H, roi, order) for sn in sns])
            iu     = np.triu_indices(G_all.shape[-1], k=k)
            vec    = G_all[..., iu[0], iu[1]]        # (n_subj, n_sess, n_elem)
            n_subj = vec.shape[0]

            for s, sess in enumerate(gl.sessions):
                V = vec[:, s]                        # (n_subj, n_elem)
                r_subj = []
                for i in range(n_subj):
                    loo = np.delete(V, i, axis=0).mean(axis=0)   # mean G of the other subjects
                    r   = np.corrcoef(V[i], loo)[0, 1]
                    r_subj.append(r)
                    rows.append({'sn': sns[i], 'Hem': H, 'roi': roi,
                                 'session': sess, 'reliability': r})
                rows.append({'sn': np.nan, 'Hem': H, 'roi': roi,
                             'session': sess, 'reliability': float(np.mean(r_subj))})

    df = pd.DataFrame(rows)
    fout = os.path.join(gl.baseDir, gl.pcmDir, f'G_reliability.{atlas_name}.tsv')
    df.to_csv(fout, sep='\t', index=False)
    return df


def decompose_dissimilarity(sns=None, atlas_name='ROI', subtract_mean=True, save=True):
    """Group/subject/noise variance decomposition of the chord-aligned RDMs.

    functional-fusion-style method-of-moments decomposition (see
    https://functional-fusion.readthedocs.io/en/latest/08reliability.html) of the
    model ``y_{i,j} = g + s_i + eps_{i,j}``, where ``y_{i,j}`` is the vectorised
    crossnobis RDM (28 chord-identity pairs, via :func:`G_sorted`) of subject *i*
    in session *j*. **Sessions play the role of repetitions.** Component norms are
    estimated from cross-products::

        v_g               = mean over different-subject pairs   of  y_{i,j} . y_{k,l}
        v_g + v_s         = mean over same-subject, different-session pairs
        v_g + v_s + v_eps = mean over the same observation      (y_{i,j} . y_{i,j})

    so ``v_s`` (the reproducible subject-specific RDM) is the idiosyncrasy, and

        r_run   = v_s / (v_s + v_eps)                     # single-session reliability
        r_whole = r_run * N / (r_run * (N - 1) + 1)       # Spearman-Brown over N sessions

    This is the method-of-moments counterpart of the ``sn:chord_pair`` component
    in :func:`dissimilarity_idiosyncrasy` and the LOO ``noise_ceiling`` /
    :func:`G_reliability`: ``v_s > 0`` means the geometry is participant-specific
    beyond the group RDM. Because sessions are the repetitions, ``v_eps`` mixes
    measurement noise with training-driven session-to-session change (same
    conservative caveat; use run partitions as ``j`` for a clean noise estimate).

    Args:
        sns (list, optional): participants; defaults to ``gl.participants``.
        atlas_name (str): key into ``gl.rois`` (default 'ROI').
        subtract_mean (bool): center each subject-session RDM (remove its overall
            dissimilarity level) so ``v_s`` reflects the subject-specific *shape*
            rather than overall magnitude. Default True.
        save (bool): write the table into ``gl.pcmDir`` (default True).

    Returns:
        pandas.DataFrame: columns ``Hem``, ``roi``, ``v_g``, ``v_s``, ``v_eps``,
            ``r_run`` and ``r_whole``.
    """
    sns       = gl.participants if sns is None else sns
    canonical = np.sort(gl.chordID)
    n_sess    = len(gl.sessions)

    rows = []
    for H in gl.Hem:
        for roi in gl.rois[atlas_name]:

            # chord-aligned RDM vectors: (n_subj, n_sess, n_pairs)
            Y = np.stack([
                np.stack([rdm_vector(pcm.G_to_dist(G_sorted(sn, H, roi, order=canonical)[s]))
                          for s in range(n_sess)])
                for sn in sns])
            if subtract_mean:
                Y = Y - Y.mean(axis=-1, keepdims=True)

            n_subj = Y.shape[0]
            X      = Y.reshape(n_subj * n_sess, -1)            # observations x pairs
            subj   = np.repeat(np.arange(n_subj), n_sess)      # subject id per observation
            P      = X @ X.T                                   # all pairwise cross-products

            same_subj = subj[:, None] == subj[None, :]
            eye       = np.eye(len(X), dtype=bool)

            v_obs    = P[eye].mean()                           # same observation
            v_within = P[same_subj & ~eye].mean()             # same subject, different session
            v_g      = P[~same_subj].mean()                    # different subjects
            v_s      = v_within - v_g
            v_eps    = v_obs - v_within

            r_run    = v_s / (v_s + v_eps)
            r_whole  = r_run * n_sess / (r_run * (n_sess - 1) + 1)

            rows.append({'Hem': H, 'roi': roi, 'v_g': v_g, 'v_s': v_s, 'v_eps': v_eps,
                         'r_run': r_run, 'r_whole': r_whole})

    out = pd.DataFrame(rows)
    if save:
        fout = os.path.join(gl.baseDir, gl.pcmDir, f'decompose_dissimilarity.{atlas_name}.tsv')
        out.to_csv(fout, sep='\t', index=False)
    return out
