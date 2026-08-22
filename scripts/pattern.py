import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import PcmPy as pcm
import pandas as pd
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.G_matrix import calc_G
from EFC_learningfMRI.util import get_trained_and_untrained
from EFC_learningfMRI.betas import BetasPrewithenedLoader
from EFC_learningfMRI.behaviour import force_patterns
from imaging_pipelines.util import calc_mle_corr
import warnings
import itertools

def _pair_index():
    """(row, col) indices of the chord pairs, split into the three pair groups.

    """
    mask                           = np.tri(8, k=-1, dtype=bool)
    mask_trained                   = mask.copy()
    mask_untrained                 = mask.copy()
    mask_trained[4:]               = False
    mask_untrained[:, :4]          = False
    mask_trained_untrained         = np.zeros((8, 8), dtype=bool)
    mask_trained_untrained[:4, 4:] = True

    # (classification, row indices, col indices) for the three chord-pair groups
    masks = {'trained'          : mask_trained,
             'untrained'        : mask_untrained,
             'trained_untrained': mask_trained_untrained}
    return {chord: np.where(m) for chord, m in masks.items()}


def _G_rows(G, sn, **labels):
    """One row per chord pair of a single 8x8 G: its crossnobis distance and cosine.

    ``labels`` are copied onto every row, and are what identifies the G the pair
    came from — Hem/roi/session for a neural G, metric/session for a force one.
    """
    chords = get_trained_and_untrained(sn)
    D      = pcm.G_to_dist(G)
    cos    = pcm.G_to_cosine(G)

    rows = []
    for chord, (r, c) in _pair_index().items():
        for ri, ci in zip(r, c):
            rows.append({'sn'        : sn,
                         **labels,
                         'chord'     : chord,
                         'pair'      : f'{chords[ri]}-{chords[ci]}',
                         'crossnobis': D[ri, ci],
                         'cosine'    : cos[ri, ci]})
    return rows


def _add_group_reference(df, keys, ref_session=3, crossval=False):
    """Attach the reference-session group geometry.

    """

    # group-mean geometry: across-subject mean of each chord pair in the reference
    # session, pooled over trained/untrained/mixed (no 'chord'/'session' in the key,
    # so all subjects contribute). Merged onto every row, so the *_group columns hold
    # the ref-session reference for every session.
    s_ref = df[df.session == ref_session]
    if crossval:
        # leave-one-subject-out: subtract each subject's own value from the pair sum,
        # so their reference is the mean over the other subjects. Keyed by subject too,
        # then merged so it broadcasts across that subject's sessions.
        ref = s_ref[keys + ['sn']].copy()
        for metric in ('crossnobis', 'cosine'):
            g = s_ref.groupby(keys)[metric]
            ref[f'{metric}_group'] = (g.transform('sum') - s_ref[metric]) / (g.transform('size') - 1)
        df = df.merge(ref, on=keys + ['sn'], how='left')
    else:
        ref = (s_ref.groupby(keys, as_index=False)
                 .agg(crossnobis_group=('crossnobis', 'mean'),
                      cosine_group    =('cosine',     'mean')))
        df  = df.merge(ref, on=keys, how='left')

    df['theta_group'] = np.arccos(df.cosine_group)

    return df


def make_G_dataframe_rois(glm=3, atlas_name='ROI', sns=None, ref_session=3, crossval=False):

    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

    rows = []
    for H, roi, sess in itertools.product(gl.Hem, rois, gl.sessions):
        for sn in sns:

            print(f'doing participant {sn}, session {sess}, {H}, {roi}...')

            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.glm{glm}.{H}.{roi}.npy'))
            rows += _G_rows(G, sn, Hem=H, roi=roi, session=sess)

    df = pd.DataFrame(rows)
    df['theta'] = np.arccos(df.cosine) # calculate angle from cosine
    df['pair']  = df.pair.map(lambda s: '-'.join(sorted(s.split('-'))))
    df = _add_group_reference(df, ['Hem', 'roi', 'pair'], ref_session, crossval)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'{atlas_name}.geometry.glm{glm}.tsv'), sep='\t', index=False)
    return df


def make_G_dataframe_force(metrics=('abs', 'der'), sns=gl.participants, ref_session=3, crossval=False):
    """The neural dataframe's counterpart over the force Gs written by pattern_G_matrix.

    Same rows, same columns, with ``metric`` (the force measure) standing in for
    ``Hem``/``roi``: the force Gs have the identical 8x8 trained-first layout, so
    every chord pair lines up one-to-one with its neural row.
    """
    sns = gl.participants if sns is None else sns

    rows = []
    for metric, sess in itertools.product(metrics, gl.sessions):
        for sn in sns:

            print(f'doing participant {sn}, session {sess}, force {metric}...')

            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.force.{metric}.npy'))
            rows += _G_rows(G, sn, metric=metric, session=sess)

    df = pd.DataFrame(rows)
    df['theta'] = np.arccos(df.cosine) # calculate angle from cosine
    df['pair']  = df.pair.map(lambda s: '-'.join(sorted(s.split('-'))))
    df = _add_group_reference(df, ['metric', 'pair'], ref_session, crossval)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t', index=False)
    return df


def _make_fname(session, repetition):
    """The 'within_session.3.1' / 'across_session' part of a G filename."""
    fname = 'across_session' if session == 'all' else 'within_session'
    fname = fname + '.' + str(session) if session != 'all' else fname
    fname = fname + '.' + str(repetition) if repetition != 'all' else fname
    return fname  


def calc_G_rois(sns=gl.participants, glm=3, sessions=('all', *gl.sessions), repetitions=('all', 1, 2)):
    """G_rois: build the betas loader, then the ROI Gs."""
    loader = BetasPrewithenedLoader(sns=sns, glm=glm)
    for data in loader:
        for session in sessions:
            for repetition in repetitions:

                print(f'rois, doing participant {data.sn}...')

                G        = calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=False)
                cov      = calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=True)
                G_raw    = calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=False, fixed_effect=False)
                G_noxval = calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=False, fixed_effect=False, crossval=False)

                save_path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
                os.makedirs(save_path, exist_ok=True)

                fname = _make_fname(session, repetition)

                np.save(os.path.join(save_path, f'G_obs.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G)
                np.save(os.path.join(save_path, f'cov.{fname}.glm{glm}.{data.Hem}.{data.roi}'), cov)
                np.save(os.path.join(save_path, f'G_obs_raw.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G_raw)
                np.save(os.path.join(save_path, f'G_obs_noxval.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G_noxval)


def calc_G_force(sns=gl.participants, metrics=('abs', 'der'),
                 sessions=('all', *gl.sessions), repetitions=('all', 1, 2)):
    """The same Gs as calc_G_rois, but over the five fingers' force instead of voxels.

    Fingers take the place of the voxels, so the matrices come out with the same
    layout as the ROI ones — 8x8 within a session, 24x24 across, trained chords
    first — and land next to them in the pcm directory as
    ``G_obs.<epoch>.force.<metric>.npy``.
    """
    force = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, 'force.run.wide.tsv'), sep='\t')

    for metric in metrics:
        for sn in sns:
            for session in sessions:
                for repetition in repetitions:

                    print(f'force {metric}, doing participant {sn}...')

                    data, cond_vec, part_vec = force_patterns(force, sn, metric, session=session, repetition=repetition)

                    G        = calc_G(data, cond_vec, part_vec, centred=False)
                    cov      = calc_G(data, cond_vec, part_vec, centred=True)
                    G_raw    = calc_G(data, cond_vec, part_vec, centred=False, fixed_effect=False)
                    G_noxval = calc_G(data, cond_vec, part_vec, centred=False, fixed_effect=False, crossval=False)

                    save_path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
                    os.makedirs(save_path, exist_ok=True)

                    fname = _make_fname(session, repetition)

                    np.save(os.path.join(save_path, f'G_obs.{fname}.force.{metric}'), G)
                    np.save(os.path.join(save_path, f'cov.{fname}.force.{metric}'), cov)
                    np.save(os.path.join(save_path, f'G_obs_raw.{fname}.force.{metric}'), G_raw)
                    np.save(os.path.join(save_path, f'G_obs_noxval.{fname}.force.{metric}'), G_noxval)


def correlation_between_sessions(sns=gl.participants, glm=3, Hem=('L', 'R'), atlas_name='ROI', residual_fname='residual.dtseries.nii'):
    """Between-session correlation per (Hem, ROI, session-pair, chord set).

    Writes two tsvs to ``path_pcm``:
      - ``MLE_correlation`` : the PCM correlation-model fit (r_group/r_indiv + the
        model's own SNR).
      - ``xval_correlation``: the cross-validated cosine on voxel-centred betas
        (matched-chord diagonal of the across-session block -> Pearson-style r),
        plus the diag(G)/diag(Sig) SNR. The (n_subj, 8, 8) cov of each group is
        also saved as ``cov.corr_across_sess.*.npy``.
    """
    path_pcm = os.path.join(gl.baseDir, gl.pcmDir)
    Mflex    = pcm.CorrelationModel("flex", num_items=4, corr=None, cond_effect=True)
    corrs    = list(itertools.combinations([1, 2, 3], 2))
    chords   = ['trained', 'untrained']

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


FUNC = {
    'G_rois'          : calc_G_rois,
    'G_force'         : calc_G_force,
    'dataframe_force' : make_G_dataframe_force,
    'dataframe_rois'  : make_G_dataframe_rois,
    'correlation'     : correlation_between_sessions,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`glm=`, `metrics=`, `repetitions=`, ...).
    """

    if what is not None:
        FUNC[what](**kwargs)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate the second moment matrices of the neural and force patterns.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: dataframe_neural)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from, G_rois and dataframe_neural only (default: the step default, 3)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

