import argparse
from imaging_pipelines.searchlight import searchlight_surf
from sklearn.covariance import ledoit_wolf
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.betas as betas
import EFC_learningfMRI.G_matrix as G_matrix
import EFC_learningfMRI.util as util
import AnatSearchlight.searchlight as sl
from joblib import Parallel, delayed
import nibabel as nb
import nitools as nt
import PcmPy as pcm
import time
import os
import numpy as np


def make_searchlight(sn):
    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in gl.Hem]
    pial = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in gl.Hem]
    mask = [os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'Hem.{H}.nii') for H in gl.Hem]
    savedir = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    searchlight_surf(white, pial, mask, savedir, maxradius=10, maxvoxels=100)


# Column names each metric writes into its gifti, in the order calc_avg_distance
# returns them. The pooling below finds the two chord-group columns by their
# '-trained'/'-untrained' suffix, so a new metric only has to keep that suffix.
METRIC_LABELS = {
    'crossnobis': ('encoding', 'encoding-trained', 'encoding-untrained'),
    'theta'     : ('theta',    'theta-trained',    'theta-untrained'),
}
METRICS = tuple(METRIC_LABELS)


def calc_avg_distance(data, cond_vec, part_vec, session, metric='crossnobis'):
    """Default searchlight metric: distance between chords, overall / trained / untrained.

    Returns three scalars (mean distance over all chord pairs, over the trained
    block and over the untrained block), so it pairs with three ``metric_labels``.
    ``metric`` selects the dissimilarity (see :func:`G_to_distance`).
    """
    data = data[:, ~np.isnan(data).any(axis=0)]
    if data.shape[1] == 0:
        return np.nan, np.nan, np.nan

    G_obs = G_matrix.calc_G(data, cond_vec, part_vec, session, fixed_effect=False)

    if not np.isfinite(G_obs).all():
        # edge searchlights can leave G undefined; G_to_cosine raises on non-finite input
        return np.nan, np.nan, np.nan

    if metric == 'crossnobis':
        D = pcm.G_to_dist(G_obs)
    elif metric == 'cosine':
        D = pcm.G_to_cosine(G_obs)
    elif metric == 'theta':
        D = np.arccos(pcm.G_to_cosine(G_obs))     # G_to_cosine already clips to [-1, 1]

    tot, trained, untrained = util.split_trained_untrained(D)

    return tot, trained.mean(), untrained.mean()


def _whiten_mnn(B, R, eps=1e-8):
    """Whiten patterns ``B`` (n_cond, V) by ``Sigma^-1/2``, with ``Sigma`` the Ledoit-Wolf
    shrinkage estimate of the V x V spatial noise covariance from residuals ``R``
    (n_timepoints, V).

    Uses the functional ``ledoit_wolf`` rather than the ``LedoitWolf`` object: the estimate
    is identical but ~12x faster, which matters across ~30k searchlights. This mirrors the
    whitening ``imaging_pipelines.model._multivariate_prewhitening`` applies per ROI.
    """
    Sigma, _ = ledoit_wolf(R)
    w, U = np.linalg.eigh(Sigma)
    w    = np.clip(w, eps, None)
    W    = (U * (1.0 / np.sqrt(w))) @ U.T          # Sigma^-1/2
    return B @ W


def calc_avg_distance_mnn(data, cond_vec, part_vec, session, n_cond, metric='crossnobis'):
    """:func:`calc_avg_distance` with searchlight-local multivariate noise normalization (MNN).

    ``data`` stacks this session's raw betas (first ``n_cond`` rows) on top of the
    residual timeseries (remaining rows), both sampled for the same searchlight. The
    betas are whitened by the searchlight-local noise covariance (Sigma^-1/2, Ledoit-Wolf)
    before the distance is computed, following Walther et al. (2016). Estimating the
    covariance within each ~100-voxel sphere keeps it tiny, so no global voxel x voxel
    matrix is ever formed -- see :func:`calc_avg_distance` for the non-whitened version.
    """
    beta  = data[:n_cond]
    resid = data[n_cond:]

    # keep only voxels usable for whitening: present in the cifti (finite residuals),
    # with non-degenerate noise, and non-empty betas. Edge searchlights can lose them
    # all, which would make the covariance undefined -- return NaN there.
    ok = np.isfinite(resid).all(axis=0) & (resid.var(axis=0) > 1e-10) & ~np.isnan(beta).all(axis=0)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan

    beta_white = _whiten_mnn(beta[:, ok], resid[:, ok])     # (n_cond, V_ok), Sigma^-1/2 whitened
    return calc_avg_distance(beta_white, cond_vec, part_vec, session, metric=metric)


def _residuals_at_candidate_voxels(R_all, bmf, voxel_indx, struct):
    """Sample the residual timeseries at a searchlight's candidate voxels, without densifying.

    ``R_all`` is the raw cifti residual matrix (n_timepoints, n_grayordinates) and ``bmf`` its
    BrainModelAxis. Each candidate voxel ``(i, j, k)`` is mapped to its cifti column via the
    brain-model axis (which stores the voxel of every column), so the residuals never get
    expanded to the full volume grid . Candidate voxels absent from the cifti become NaN columns and
    are dropped later by the prewhitening. Returns (n_timepoints, n_candidate_voxels).
    """
    target = nb.cifti2.BrainModelAxis.to_cifti_brain_structure_name(struct)
    col_of = {}
    for nam, slc, bm in bmf.iter_structures():
        if nam == target:
            cols   = np.arange(R_all.shape[1])[slc]
            col_of = {tuple(int(x) for x in v): int(c) for v, c in zip(bm.voxel, cols)}
            break

    idx  = np.array([col_of.get((int(i), int(j), int(k)), -1) for i, j, k in zip(*voxel_indx)])
    good = idx >= 0
    out  = np.full((R_all.shape[0], idx.size), np.nan, dtype=np.float32)
    out[:, good] = R_all[:, idx[good]]
    return out


class Searchlight():

    def __init__(self, sns=None, glm=None, metric_fn=None,
                 metric_labels=('encoding', 'encoding-trained', 'encoding-untrained'), 
                 sessions=(3, 9, 23), out_fname='searchlight_crossnobis', multivariate_pw=False):
        """
        Metric labels should be as many as the output in metric_fcn

        With ``multivariate_pw=True`` the betas are whitened by the searchlight-local noise
        covariance (multivariate noise normalization) instead of being prewhitened once with
        ``ResMS``. The residual file is chosen accordingly (the ``residual.dtseries.nii`` timeseries
        for MNN, the ``ResMS.nii`` volume otherwise) and ``metric_fn`` must accept the stacked
        betas+residuals (e.g. :func:`calc_avg_distance_mnn`).
        """
        self.sns             = gl.participants if sns is None else sns
        self.glm             = glm
        self.metric_fn       = metric_fn
        self.metric_labels   = metric_labels
        self.sessions        = sessions
        self.out_fname       = out_fname
        self.multivariate_pw = multivariate_pw

        if multivariate_pw:
            self.residual_fname = 'residual.dtseries.nii'
        else:
            self.residual_fname = 'ResMS.nii'


    def _searchlight_subject_univarite_pw(self, sn):
        """Run the metric over one subject's searchlight and save one gifti per
        hemisphere and session (metrics as columns).

        Betas and residuals are loaded and prewhitened once; the searchlight is loaded
        per hemisphere, and each session selects only its own runs (see runs_to_keep).
        """

        print('loading and prewhitening betas...')
        reginfo   = betas.RegInfo(sn, self.glm)
        beta_img  = betas.load_betas(sn, self.glm)                             # volume image (X, Y, Z, n_cond)
        residuals = betas.load_residuals(sn, self.glm, self.residual_fname)    # ResMS volume
        beta_pw   = beta_img.get_fdata() / np.sqrt(residuals.get_fdata()[:, :, :, None])   # univariate prewhitening

        n_out = len(self.metric_labels)
        for H in gl.Hem:
            print(f'loading searchlight {H}...')
            SL     = sl.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'searchlight.{H}.h5'))
            struct = 'CortexLeft' if H == 'L' else 'CortexRight'

            for session in self.sessions:
                print(f'running searchlight {H}, session {session}...')

                keep           = util.runs_to_keep(reginfo.cond_vec, session=session)
                function_args  = {'cond_vec': reginfo.cond_vec[keep], 
                                  'part_vec': reginfo.part_vec[keep],
                                  'session' : session}
                beta_vol       = nb.Nifti2Image(beta_pw[:, :, :, keep], affine=beta_img.affine, header=beta_img.header)

                result = np.asarray(SL.run_parallel(beta_vol, self.metric_fn, function_args=function_args, nargout=n_out))
                result = result[:, None] if result.ndim == 1 else result   # (n_centers, n_out)

                gifti = nt.make_func_gifti(result, anatomical_struct=struct, column_names=list(self.metric_labels))
                fname = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'{self.out_fname}.{session}.glm{self.glm}.{H}.func.gii')
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                nb.save(gifti, fname)


    @staticmethod
    def _run_presampled(data, voxlists, metric_fn, function_args, nargout, n_jobs=8):
        """Run ``metric_fn`` over pre-sampled searchlight data (mirrors ``sl.run_parallel``).

        ``data`` is (n_measurements, n_candidate_voxels) already sampled at the searchlight's
        candidate voxels, so the residual timeseries never has to be densified to a volume for
        ``run_parallel`` to sample it. Each center slices its own voxels out of ``data``.
        """
        def _process_one(vl):
            if len(vl) == 0:
                return np.full(nargout, np.nan, dtype=float)
            return metric_fn(data[:, vl], **function_args)

        with Parallel(n_jobs=n_jobs, batch_size=100, verbose=10) as parallel:
            results = parallel(delayed(_process_one)(vl) for vl in voxlists)
        return np.array(results)


    def _searchlight_subject_multivarite_pw(self, sn):
        """Multivariate-noise-normalized variant of :meth:`_searchlight_subject_univarite_pw`.

        Raw betas and the residual timeseries are sampled at the searchlight's candidate voxels
        (never densified to a volume). For each center the metric whitens the betas by the local
        ~100-voxel noise covariance before computing crossnobis, so no global voxel x voxel matrix
        is formed. Betas and residuals are stacked row-wise so one array serves both.
        """

        print('loading betas and residual timeseries...')
        reginfo   = betas.RegInfo(sn, self.glm)
        beta_img  = betas.load_betas(sn, self.glm)                             # volume image (X, Y, Z, n_cond)
        beta_data = beta_img.get_fdata(dtype=np.float32)                       # raw betas, whitened per searchlight
        res_cifti = betas.load_residuals(sn, self.glm, self.residual_fname)    # residual.dtseries.nii (cifti)
        R_all     = np.asarray(res_cifti.dataobj, dtype=np.float32)            # (n_timepoints, n_grayordinates)
        bmf       = res_cifti.header.get_axis(1)

        n_out = len(self.metric_labels)
        for H in gl.Hem:
            print(f'loading searchlight {H}...')
            SL     = sl.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'searchlight.{H}.h5'))
            struct = 'CortexLeft' if H == 'L' else 'CortexRight'
            vidx   = SL.voxel_indx

            beta_cand  = beta_data[vidx[0], vidx[1], vidx[2]].T                              # (n_cond, n_cand)
            resid_cand = _residuals_at_candidate_voxels(R_all, bmf, vidx, struct)            # (n_timepoints, n_cand)

            for session in self.sessions:
                print(f'running searchlight {H}, session {session}...')

                keep          = util.runs_to_keep(reginfo.cond_vec, session=session)
                data          = np.vstack([beta_cand[keep], resid_cand])   # betas on top of residuals
                function_args = {'cond_vec': reginfo.cond_vec[keep],
                                 'part_vec': reginfo.part_vec[keep],
                                 'session' : session,
                                 'n_cond'  : int(keep.sum())}

                result = self._run_presampled(data, SL.voxlist, self.metric_fn, function_args, n_out)
                result = result[:, None] if result.ndim == 1 else result   # (n_centers, n_out)

                gifti = nt.make_func_gifti(result, anatomical_struct=struct, column_names=list(self.metric_labels))
                fname = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'{self.out_fname}.{session}.glm{self.glm}.{H}.func.gii')
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                nb.save(gifti, fname)


    def run(self):
        """Run the searchlight for every subject; each writes one gifti per hemisphere and
        session. Pool across subjects afterwards with :func:`pool_searchlight`."""

        for sn in self.sns:
            print(f'doing participant {sn}')
            if self.multivariate_pw:
                self._searchlight_subject_multivarite_pw(sn)
            else:
                self._searchlight_subject_univarite_pw(sn)


def _trained_untrained_columns(cols):
    """Indices of the trained and untrained columns of a searchlight gifti.

    Found by suffix rather than by a hard-coded name, so the same pooling serves
    every metric in ``METRIC_LABELS`` ('encoding-trained', 'theta-trained', ...).
    """
    trained   = [i for i, c in enumerate(cols) if c.endswith('-trained')]
    untrained = [i for i, c in enumerate(cols) if c.endswith('-untrained')]
    if len(trained) != 1 or len(untrained) != 1:
        raise ValueError(f'expected exactly one -trained and one -untrained column, got {cols}')
    return trained[0], untrained[0]


def pool_searchlight(sns=None, glm=None, fname='searchlight_crossnobis', sessions=gl.sessions):
    """Average the per-subject searchlight maps into group maps, per hemisphere and session.

    Writes two files per hemisphere and session, in ``surfDir`` next to the subject
    folders they are pooled from:

    ``<fname>.<session>.glm<glm>.<H>.func.gii``
        group mean of every metric column (nan-safe, so a subject missing a searchlight
        centre does not blank it for everyone).
    ``<fname>_diff.<session>.glm<glm>.<H>.func.gii``
        group mean of each subject's trained-minus-untrained difference, i.e. the
        learning contrast, computed within subject before averaging.
    """
    sns         = gl.participants if sns is None else sns
    struct_dict = dict(zip(gl.Hem, gl.struct_cortex))

    for H in gl.Hem:
        struct = struct_dict[H]
        for session in sessions:
            data, data_diff = [], []
            for sn in sns:
                fpath = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'{fname}.{session}.glm{glm}.{H}.func.gii')
                gifti = nb.load(fpath)
                cols  = nt.get_gifti_column_names(gifti)
                tr_col, untr_col = _trained_untrained_columns(cols)
                data_ = nt.get_gifti_data_matrix(gifti)
                data.append(data_)
                data_diff.append(data_[:, tr_col] - data_[:, untr_col])

            group = np.nanmean(np.stack(data, axis=0), axis=0)
            gifti = nt.make_func_gifti(group, anatomical_struct=struct, column_names=cols)
            nb.save(gifti, os.path.join(gl.baseDir, gl.surfDir, f'{fname}.{session}.glm{glm}.{H}.func.gii'))

            group_diff = np.nanmean(np.stack(data_diff, axis=0), axis=0)
            gifti      = nt.make_func_gifti(group_diff, anatomical_struct=struct, column_names=['trained-untrained'])
            nb.save(gifti, os.path.join(gl.baseDir, gl.surfDir, f'{fname}_diff.{session}.glm{glm}.{H}.func.gii'))


