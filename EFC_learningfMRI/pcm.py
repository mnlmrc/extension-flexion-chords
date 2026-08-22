import PcmPy as pcm
import os
import numpy as np
from EFC_learningfMRI.util import  get_trained_and_untrained, runs_to_keep
import EFC_learningfMRI.globals as gl
from imaging_pipelines.model import calc_prewhitened_betas
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

    comp_names = ['type', 'trained', 'untrained', 'finger', 'pattern', 'flexion']

    G_tr_untr, G_tr, G_untr = fixed_models()
    order = np.array(get_trained_and_untrained(sn)).astype(int)
    G_finger, G_pattern, G_flexion = subj_spec_models(order=order)

    M = []
    M.append(pcm.FixedModel('null',      np.zeros((8, 8))))
    M.append(pcm.FixedModel('type',      G_tr_untr)) 
    M.append(pcm.FixedModel('trained',   G_tr)) 
    M.append(pcm.FixedModel('untrained', G_untr))
    M.append(pcm.FixedModel('finger',    G_finger))
    M.append(pcm.FixedModel('pattern',   G_pattern))
    M.append(pcm.FixedModel('flexion',   G_flexion))
    M.append(pcm.ComponentModel('component', np.array([G_tr_untr / np.trace(G_tr_untr),
                                                       G_tr      / np.trace(G_tr),
                                                       G_untr    / np.trace(G_untr),
                                                       G_finger  / np.trace(G_finger),
                                                       G_pattern / np.trace(G_pattern),
                                                       G_flexion / np.trace(G_flexion)
                                                      ]))) 
    M.append(pcm.FreeModel('ceil', 8)) 

    return M, comp_names


def fit_component_model(loader):

    glm   = loader.glm
    atlas = loader.atlas_name

    df     = pd.DataFrame()
    thetas = {}                       # (sn, Hem, roi, session) -> theta_in, pickled below
    for data in loader:
        for session in gl.sessions:
            keep = runs_to_keep(data.part_vec.size, session=session)

            betas    = data.betas[keep]
            cond_vec = data.cond_vec[keep]
            part_vec = data.part_vec[keep]

            # make dataset
            obs_des = {'cond_vec': cond_vec, 'part_vec': part_vec}
            Y       = pcm.dataset.Dataset(betas, obs_descriptors=obs_des)

            model, comp_names = make_models(data.sn)
            model = model[-2] # select component model

            G_session3 = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.within_session.3.glm{glm}.{data.Hem}.{data.roi}.npy'))
            G_session3 = G_session3.mean(axis=0)
            comp_names.append('base')
            model.Gc      = np.r_[model.Gc, G_session3[None, :, :]]
            model.n_param = model.Gc.shape[0]     # sync param count with the extended Gc

            _, theta_in = pcm.fit_model_individ(Y, model, fit_scale=False, verbose=True, fixed_effect='block')

            path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f'component_model.theta_in.{atlas}.glm{glm}.{session}.{data.Hem}.{data.roi}.p'), 'wb') as f:
                pickle.dump(theta_in, f)


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
    



    