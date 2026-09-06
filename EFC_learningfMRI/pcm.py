import PcmPy as pcm
import os
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from EFC_learningfMRI.util import  get_trained_and_untrained, runs_to_keep
from EFC_learningfMRI.G_matrix import G_sorted
from EFC_learningfMRI.betas import BetasPrewithenedLoader
import EFC_learningfMRI.globals as gl
from imaging_pipelines.util import calc_mle_corr
import nitools as nt
import nibabel as nb
import pandas as pd
import pickle

C = pcm.centering(8)

FINGER = {
     21911: np.array([1, 1, 0, 1, 1]),
     92122: np.array([0, 1, 1, 1, 1]),
     91211: np.array([0, 1, 1, 1, 1]),
     22911: np.array([1, 1, 0, 1, 1]),
     21291: np.array([1, 1, 1, 0, 1]),
     12129: np.array([1, 1, 1, 1, 0]),
     12291: np.array([1, 1, 1, 0, 1]),
     11911: np.array([1, 1, 0, 1, 1])
}

PATTERN = {
     21911: np.array([-1,  1,  0,  1,  1]),
     92122: np.array([ 0, -1,  1, -1, -1]),
     91211: np.array([ 0,  1, -1,  1,  1]),
     22911: np.array([-1, -1,  0,  1,  1]),
     21291: np.array([-1,  1, -1,  0,  1]),
     12129: np.array([ 1, -1,  1, -1,  0]),
     12291: np.array([ 1, -1, -1,  0,  1]),
     11911: np.array([ 1,  1,  0,  1,  1])
}

FLEXION = {
    21911: 1,
    92122: 3,
    91211: 1,
    22911: 2,
    21291: 2,
    12129: 2,
    12291: 2,
    11911: 0
}


def fixed_models():
    
    # trained untrained
    v_tr_untr = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    G_tr_untr = C @ np.outer(v_tr_untr, v_tr_untr)

    # trained
    tr = np.zeros(8)
    tr[:4] = 1
    G_tr = C @ np.diag(tr)

    # untrained
    untr = np.zeros(8)
    untr[4:] = 1
    G_untr = C @ np.diag(untr)

    return G_tr_untr, G_tr, G_untr


def subj_spec_models(order=None, glm=3):

    """
    order: list of chords in the order they appear in the G matrices
    """

    finger  = np.zeros((8, 5))
    pattern = np.zeros_like(finger)
    flexion = np.zeros(8)
    for i, ch in enumerate(order):
        flexion[i] = FLEXION[ch]
        finger[i]  = FINGER[ch]
        pattern[i] = PATTERN[ch]

    G_finger  = C @ (finger @ finger.T)
    G_pattern = C @ (pattern @ pattern.T)
    G_flexion = C @ np.outer(flexion, flexion)

    return G_finger, G_pattern, G_flexion


def make_models(sn):

    G_tr_untr, G_tr, G_untr = fixed_models()

    order = np.array(get_trained_and_untrained(sn)).astype(int)

    G_force_raw = G_sorted(np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', 'G_obs_raw.within_session.3.force.raw.npy')), sn, order)
    G_force_abs = G_sorted(np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', 'G_obs_raw.within_session.3.force.abs.npy')), sn, order)
    G_force_der = G_sorted(np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', 'G_obs_raw.within_session.3.force.der.npy')), sn, order)

    G_finger, G_pattern, G_flexion = subj_spec_models(order=order)

    M = []
    M.append(pcm.FixedModel(    'null',        np.zeros((8, 8))))
    M.append(pcm.FixedModel(    'type',        G_tr_untr)) 
    M.append(pcm.FixedModel(    'trained',     G_tr)) 
    M.append(pcm.FixedModel(    'untrained',   G_untr))
    M.append(pcm.FixedModel(    'finger',      G_finger))
    M.append(pcm.FixedModel(    'pattern',     G_pattern))
    M.append(pcm.FixedModel(    'flexion',     G_flexion))
    # M.append(pcm.FixedModel(    'force_raw',   G_force_raw))
    # M.append(pcm.FixedModel(    'force_abs',   G_force_abs))
    # M.append(pcm.FixedModel(    'force_der',   G_force_der))
    M.append(pcm.ComponentModel('component', np.array([G_tr_untr   / np.trace(G_tr_untr),
                                                       G_tr        / np.trace(G_tr),
                                                       G_untr      / np.trace(G_untr),
                                                       G_finger    / np.trace(G_finger),
                                                       G_pattern   / np.trace(G_pattern),
                                                       G_flexion   / np.trace(G_flexion),
                                                    #    G_force_raw / np.trace(G_force_raw),
                                                    #    G_force_abs / np.trace(G_force_abs),
                                                    #    G_force_der / np.trace(G_force_der)
                                                       ])))
    M.append(pcm.FreeModel('ceil', 8))

    comp_names = ['type', 
                  'trained',
                  'untrained',
                  'finger', 
                  'pattern',
                  'flexion',
                #   'force_raw',
                #   'force_abs',
                #   'force_der',
                  ]

    return M, comp_names


def _dump(obj, fname, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, fname), 'wb') as f:
        pickle.dump(obj, f)


def fit_component_model(loader, sessions=None):
    """Fit the models of :func:`make_models` to every (subject, Hem, roi, session).

    Follows the fitting conventions of the pcm-toolbox ``demo_fingers`` notebook:

    - :func:`PcmPy.fit_model_individ` on the full model list, every subject with its
      own theta. **Every model is fitted with** ``fit_scale=True``: a ``FixedModel``
      has ``n_param == 0``, so without a scale parameter its G cannot be rescaled to
      the subject's signal strength and the likelihood ends up reporting the
      arbitrary size of that G rather than its shape -- which is what puts every
      fixed model far below the null when ``fit_scale=False``.
    - :func:`PcmPy.fit_model_group` and :func:`PcmPy.fit_model_group_crossval` on
      :data:`GROUP_MODELS`, which is what gives the noise ceiling its two bounds:
      the group fit of ``ceil`` (fitted to every subject, the one it is scored on
      included, so it overfits) is the upper bound, the crossvalidated group fit
      (fitted to the other N-1) the lower one.

    The group fits need every subject of a cell at once, so the datasets are kept as
    the loader yields them -- the loader is the expensive part and only runs once.

    Writes to the pcm dir, one file per cell: ``component_model.T_in.*`` and
    ``component_model.theta_in.*`` under ``subj<sn>/``, and ``component_model.T_gr.*``
    and ``component_model.T_cv.*`` at the top level, indexed by subject number.
    """
    glm      = loader.glm
    atlas    = loader.atlas_name
    sessions = gl.sessions if sessions is None else sessions

    datasets = defaultdict(dict)          # (Hem, roi, session) -> {sn: Dataset}
    for data in loader:
        for session in sessions:
            keep = runs_to_keep(data.cond_vec, session=session)

            betas    = data.betas[keep]
            cond_vec = data.cond_vec[keep]
            part_vec = data.part_vec[keep]

            # make dataset
            obs_des = {'cond_vec': cond_vec, 'part_vec': part_vec}
            Y       = pcm.dataset.Dataset(betas, obs_descriptors=obs_des)

            datasets[(data.Hem, data.roi, session)][data.sn] = Y

            model, comp_names = make_models(data.sn)

            T_in, _ = pcm.fit_model_individ(Y, model, fit_scale=True, verbose=True, fixed_effect='block')
            _, theta_in = pcm.fit_model_individ(Y, model[-2], fit_scale=False, verbose=True, fixed_effect='block')

            path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
            stem = f'{atlas}.glm{glm}.{session}.{data.Hem}.{data.roi}.p'
            _dump(theta_in, f'component_model.theta_in.{stem}', path)
            _dump(T_in,     f'component_model.T_in.{stem}',     path)


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

    def group_datasets(self, session_pairs, chords):
        """Per-subject Datasets, cross-validated G and SNR for every group.

        Drives the loader once (one prewhitening per subject/ROI) and slices each
        subject's betas into every (session-pair, chord) group with ``runs_to_keep``,
        which is the only reader of the cond_vec format. ``session_pairs`` are pairs of
        real session numbers (3, 9, 23), not label indices. Betas are centred
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
                keep = runs_to_keep(data.cond_vec, session=sessions, chord=chord)

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
    



    