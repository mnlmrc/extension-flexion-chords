import numpy as np
import PcmPy as pcm
import pickle
import os
import nibabel as nb
import EFC_learningfMRI.globals as gl
import nitools as nt
import time
import pandas as pd
import argparse
import itertools
from dataclasses import dataclass, field
from typing import Sequence
from collections import defaultdict
from EFC_learningfMRI.betas import BetasPrewithenedLoader
from imaging_pipelines.util import calc_mle_corr
import imaging_pipelines.model as md


sessions = [1, 2, 3]
path_pcm = os.path.join(gl.baseDir, gl.pcmDir)
Mflex    = pcm.CorrelationModel("flex", num_items=4, corr=None, cond_effect=True)
corrs    = list(itertools.combinations(sessions, 2))
chords   = ['trained', 'untrained']


@dataclass
class CorrelationBetweenSessions():
    """Per-subject Datasets for between-session correlation.

    Betas are prewhitened once per (subject, ROI) by BetasPrewithenedLoader and
    reused for every session-pair and chord set, so the expensive step runs once.
    """

    glm: int
    sns: Sequence[int]  = field(default_factory=lambda: gl.participants)
    atlas_name: str     = 'ROI'
    residual_fname: str = 'residual.dtseries.nii'
    Hem: Sequence[str]  = ('L', 'R')

    @staticmethod
    def _runs_to_keep(cond_vec, sessions, chord):
        """Boolean mask over regressors: one chord set within a pair of sessions.

        cond_vec entries are 'session,chord' (chord index 0..7, trained = 0..3).
        """
        parts    = pd.Series(cond_vec).str.split(',', expand=True)
        sessx    = parts[0].astype(int)
        chordIDx = parts[1].astype(int)
        in_sess  = sessx.isin(sessions).to_numpy()

        if chord == 'trained':
            return (chordIDx < 4).to_numpy() & in_sess
        elif chord == 'untrained':
            return (chordIDx >= 4).to_numpy() & in_sess
        raise ValueError(f"chord must be 'trained' or 'untrained', got {chord!r}")


    def group_datasets(self, session_pairs, chords):
        """Per-subject Datasets, cross-validated G and SNR for every group.

        Drives the loader once (one prewhitening per subject/ROI) and slices each
        subject's betas into every (session-pair, chord) group. Betas are centred
        across voxels (``axis=1``) so ``G_to_cosine`` reads out a Pearson-style
        correlation. Returns three dicts keyed by (Hem, roi, session_pair, chord):
        ``Y[key]`` a list of one pcm Dataset per subject, ``cov[key]`` a
        (n_subj, 8, 8) array, and ``snr[key]`` a (n_subj,) signal/noise ratio.
        Subjects follow loader order (``self.sns``).

        SNR is ``mean(diag(G)) / mean(diag(Sig))`` where ``Sig`` is the second
        output of ``est_G_crossval`` (the noise covariance of the single-run
        condition estimates): cross-validated signal variance over noise variance.
        """
        loader = BetasPrewithenedLoader(self.glm, sns=self.sns, atlas_name=self.atlas_name, residual_fname=self.residual_fname, Hem=self.Hem)
        Y, cov, snr = defaultdict(list), defaultdict(list), defaultdict(list)
        for data in loader:
            for sessions, chord in itertools.product(session_pairs, chords):
                keep = self._runs_to_keep(data.cond_vec, sessions, chord)

                betas     = data.betas[keep]
                betas     = betas - betas.mean(axis=1, keepdims=True)   # centre across voxels -> cosine == Pearson r
                cond_vec  = data.cond_vec[keep]
                part_vec  = data.part_vec[keep]

                dataset = pcm.dataset.Dataset(betas, obs_descriptors={'cond_vec': cond_vec, 'part_vec': part_vec})

                cov_, Sig_ = pcm.est_G_crossval(dataset.measurements, cond_vec, part_vec, X=pcm.indicator(part_vec))

                Y[data.Hem, data.roi, sessions, chord].append(dataset)
                cov[data.Hem, data.roi, sessions, chord].append(np.asarray(cov_))
                snr[data.Hem, data.roi, sessions, chord].append(np.diagonal(cov_).mean() / np.diagonal(Sig_).mean())

        return (Y,
                {key: np.array(v) for key, v in cov.items()},
                {key: np.array(v) for key, v in snr.items()})


def fit_correlation(Y, model):
    """Fit the correlation model (individual + group) to a list of Datasets and
    return per-subject individual r, group r and SNR.

    Note on `cond_effect`: with one block regressor per run, the condition mean is
    already absorbed by the fixed effect ([Z X] is rank deficient by one per
    session), so the cond_effect thetas are unidentifiable and sit at their
    est_theta0 floor. r is identical either way; cond_effect=False just drops two
    dead parameters. The flag is read off the model so the theta indices used by
    calc_mle_corr cannot desync from it.
    """
    _, theta    = pcm.fit_model_individ(Y, model, fixed_effect='block', fit_scale=False, verbose=True)
    _, theta_gr = pcm.fit_model_group(Y, model, fixed_effect='block', fit_scale=False, verbose=True)

    r_indiv, r_group, SNR, _, _, _ = calc_mle_corr(model, theta[0], theta_gr[0], cond_effect=model.cond_effect)

    return r_indiv, r_group, SNR


def item_relabelings(n_items=4, derangements_only=True):
    """The set of item relabelings used to build the null distribution of r.

    Each relabeling is a permutation of the `n_items` item labels applied to one
    session only. `derangements_only` keeps just the permutations with no fixed
    point (9 of the 24 for 4 items); permutations that leave items in place keep
    part of the true correspondence and make the null too close to the observed r.
    """
    perms = [p for p in itertools.permutations(range(n_items))]
    if derangements_only:
        perms = [p for p in perms if all(p[i] != i for i in range(n_items))]
    else:
        perms = [p for p in perms if p != tuple(range(n_items))]
    return perms


def relabel_session_items(cond_vec, perm, session):
    """Relabel the items of `session` in `cond_vec` according to `perm`.

    `cond_vec` holds 'session,item' strings. The relabeling is applied identically
    to every run, so the item patterns of each session stay as reliable as in the
    observed fit and only the *pairing between sessions* is broken. Shuffling
    labels independently within each run instead destroys the within-session item
    signal too, which leaves the correlation parameter unidentifiable and pins the
    MLE at +/-1 regardless of the data.
    """
    cond_vec = np.asarray(cond_vec)
    sess     = np.array([c.split(',')[0] for c in cond_vec])
    item     = np.array([c.split(',')[1] for c in cond_vec])

    labels  = np.unique(item)
    mapping = dict(zip(labels, labels[list(perm)]))

    mask           = sess == str(session)
    out            = cond_vec.copy()
    out[mask]      = [f'{s},{mapping[i]}' for s, i in zip(sess[mask], item[mask])]
    return out


def relabeled_datasets(Y, perm, session):
    """Copies of `Y` with the items of `session` relabeled by `perm`.

    Returns new Datasets: mutating obs_descriptors in place would also change `Y`,
    since a list copy shares the underlying Dataset objects.
    """
    return [pcm.dataset.Dataset(
                D.measurements,
                obs_descriptors={
                    'cond_vec': relabel_session_items(D.obs_descriptors['cond_vec'], perm, session),
                    'part_vec': D.obs_descriptors['part_vec'].copy()})
            for D in Y]



def correlation_between_session(sns, glm, Hem=('L', 'R'), atlas_name='ROI', residual_fname='residual.dtseries.nii'):
    """Between-session correlation per (Hem, ROI, session-pair, chord set).

    Writes two tsvs to ``path_pcm``:
      - ``MLE_correlation`` : the PCM correlation-model fit (r_group/r_indiv + the
        model's own SNR).
      - ``xval_correlation``: the cross-validated cosine on voxel-centred betas
        (matched-chord diagonal of the across-session block -> Pearson-style r),
        plus the diag(G)/diag(Sig) SNR. The (n_subj, 8, 8) cov of each group is
        also saved as ``cov.corr_across_sess.*.npy``.
    """
    correlation  = CorrelationBetweenSessions(glm, sns=sns, atlas_name=atlas_name, Hem=Hem, residual_fname=residual_fname)
    Y, cov, snr  = correlation.group_datasets(corrs, chords)

    mle, xval = [], []
    for (H, roi, n_sess, chord), Y_ in Y.items():

        sessions = gl.sessions[n_sess[0] - 1], gl.sessions[n_sess[1] - 1]
        spair    = f'{sessions[0]}-{sessions[1]}'

        # PCM correlation model (MLE) -- keeps the model's own SNR
        r_indiv, r_group, SNR = fit_correlation(Y_, Mflex)
        mle.append(pd.DataFrame({'sn'     : sns,
                                 'r_group': r_group,
                                 'r_indiv': r_indiv,
                                 'SNR'    : SNR,
                                 'chord'  : chord,
                                 'corr'   : spair,
                                 'roi'    : roi,
                                 'Hem'    : H,}))

        # cross-validated cosine -- matched-chord diagonal + diag(G)/diag(Sig) SNR
        G_group = cov[H, roi, n_sess, chord]
        np.save(os.path.join(path_pcm, f'cov.corr_across_sess.glm{glm}.{spair}.{chord}.{H}.{roi}.npy'), G_group)

        r_xval = pcm.G_to_cosine(G_group)
        r_avg  = np.diagonal(r_xval[:, :4, 4:], axis1=1, axis2=2).mean(axis=1)
        xval.append(pd.DataFrame({'sn'     : sns,
                                  'r_indiv': r_avg,
                                  'r_group': r_avg.mean(),
                                  'SNR'    : snr[H, roi, n_sess, chord],
                                  'corr'   : spair,
                                  'chord'  : chord,
                                  'roi'    : roi,
                                  'Hem'    : H}))

    pd.concat(mle, ignore_index=True).to_csv(os.path.join(path_pcm, f'MLE_correlation.{atlas_name}.glm{glm}.tsv'),  sep='\t', index=False)
    pd.concat(xval, ignore_index=True).to_csv(os.path.join(path_pcm, f'xval_correlation.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


if __name__=='__main__':
    sns   = gl.participants
    glm   = 3
    atlas = 'ROI'

    correlation_between_session(sns, glm, atlas_name=atlas, residual_fname='residual.dtseries.nii')




