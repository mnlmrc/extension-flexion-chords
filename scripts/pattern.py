import argparse
import os
import pickle
import itertools
import numpy as np
import PcmPy as pcm
import pandas as pd
from scipy.stats import spearmanr
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.G_matrix import calc_G, G_sorted
from EFC_learningfMRI.util import get_trained_and_untrained
from EFC_learningfMRI.betas import BetasPrewithenedLoader
from EFC_learningfMRI.behaviour import force_patterns
from EFC_learningfMRI.pcm import CorrelationBetweenSessions, fit_correlation, fit_component_model, make_models

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
    """One row per chord pair of a single 8x8 G: its crossnobis distance, cosine and angle.

    ``labels`` are copied onto every row, and are what identifies the G the pair
    came from — Hem/roi/session for a neural G, metric/session for a force one.
    The ``pair`` label is order-normalised so both Gs key on the same pair id.
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
                         'pair'      : '-'.join(sorted([str(chords[ri]), str(chords[ci])])),
                         'crossnobis': D[ri, ci],
                         'cosine'    : cos[ri, ci],
                         'theta'     : np.arccos(cos[ri, ci])})
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


def make_dataframe_rois(glm=3, atlas_name='ROI', sns=None, ref_session=3, crossval=False):

    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

    rows = []
    for H, roi, sess in itertools.product(gl.Hem, rois, gl.sessions):
        for sn in sns:

            print(f'doing participant {sn}, session {sess}, {H}, {roi}...')

            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.glm{glm}.{H}.{roi}.npy'))
            rows += _G_rows(G, sn, Hem=H, roi=roi, session=sess)

    df = pd.DataFrame(rows)
    df = _add_group_reference(df, ['Hem', 'roi', 'pair'], ref_session, crossval)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)
    return df


def make_dataframe_force(metrics=('abs', 'der'), sns=gl.participants, ref_session=3, crossval=False):
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


def _rdm_vectors(sns, H, roi, session, glm, prefix='G_obs_raw', order=None):
    """Vectorised dissimilarity matrix of every participant, on a common chord order.

    Each participant's G is stored in their own trained-first order, so it is put
    through ``G_sorted`` first: without that, entry k of one participant's vector
    is a different chord pair than entry k of the next, and the correlations below
    are meaningless. ``order`` defaults to ``gl.chordID`` (line participants up by
    chord identity); pass a trained/untrained framing to align by training status.

    Returns an (n_subj, n_pairs) array, the lower triangle of each participant's RDM.
    """
    mask = np.tri(len(gl.chordID), k=-1, dtype=bool)

    rdms = []
    for sn in sns:
        G   = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}',
                    f'{prefix}.within_session.{session}.glm{glm}.{H}.{roi}.npy'))
        G_s = G_sorted(G, sn, order=order)
        D   = pcm.G_to_dist(G_s)
        rdms.append(D[mask])

    return np.array(rdms)


def _corr(a, b, method='pearson'):
    """Correlation between two vectorised RDMs."""
    if method == 'spearman':
        return spearmanr(a, b)[0]
    return np.corrcoef(a, b)[0, 1]


def _noise_ceiling(rdms, method='pearson'):
    """Leave-one-participant-out lower bound and the upper bound of the RSA noise ceiling.

    lower: each participant's RDM against the mean of the *other* participants. The
           group mean is estimated without them, so this is what a model that
           generalises across participants can be expected to reach.
    upper: the same, but against the mean of *all* participants, that participant
           included. The reference is fitted to the participant it is scored on, so
           it overestimates, and no model should beat it.

    Returns lower, upper: (n_subj,) arrays, one correlation per participant.
    """
    n     = len(rdms)
    if n < 3:
        raise ValueError(f'need at least 3 participants for a leave-one-out ceiling, got {n}')
    total = rdms.sum(axis=0)

    lower, upper = [], []
    for rdm in rdms:
        others_mean = (total - rdm) / (n - 1)  # group mean leaving this participant out
        group_mean  = total / n                # group mean including this participant

        lower.append(_corr(rdm, others_mean, method))
        upper.append(_corr(rdm, group_mean, method))

    return np.array(lower), np.array(upper)


def make_noise_ceiling_dataframe(glm=3, atlas_name='ROI', sns=None, prefix='G_obs_raw',
                                 method='pearson', order=None):
    """Noise ceiling of the crossnobis RDM for every hemisphere, roi and session.

    One row per participant, so the ceiling of a given roi/Hem/session is the mean
    of its rows (``df.groupby(['Hem', 'roi', 'session'])[['lower', 'upper']].mean()``).
    Writes ``noise_ceiling.within_session.<atlas_name>.glm<glm>.tsv`` to the pcm dir.
    """
    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

    rows = []
    for H, roi, session in itertools.product(gl.Hem, rois, gl.sessions):

        print(f'doing {H}, {roi}, session {session}...')

        rdms         = _rdm_vectors(sns, H, roi, session, glm, prefix=prefix, order=order)
        lower, upper = _noise_ceiling(rdms, method=method)

        for i, sn in enumerate(sns):
            rows.append({'sn'     : sn,
                         'Hem'    : H,
                         'roi'    : roi,
                         'session': session,
                         'lower'  : lower[i],
                         'upper'  : upper[i]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'noise_ceiling.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)
    return df


def fit_component_model_rois(sns=gl.participants, glm=3, atlas_name='ROI', residual_fname='residual.dtseries.nii'):
    """Fit the PCM component model per (subject, Hem, roi, session).

    Prewhitens the betas once per (subject, roi) with BetasPrewithenedLoader, then
    fits the component model (with the session-3 group G appended as a 'base'
    component) and pickles each fit's ``theta_in`` next to the participant's Gs, as
    ``component_model.theta_in.<atlas_name>.glm<glm>.<session>.<Hem>.<roi>.p`` --
    which component_summary reads back.
    """
    loader = BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    fit_component_model(loader)


def make_component_model_dataframe(sns=None, glm=3, atlas_name='ROI'):
    """Collect the component weights fitted by component_fit into one dataframe.

    Reads each pickled ``theta_in``, exponentiates to component weights (trailing
    params are noise scales), and writes one long-form row per component to
    ``component_model.<atlas_name>.glm<glm>.tsv`` in the pcm dir.
    """
    sns           = gl.participants if sns is None else sns
    _, comp_names = make_models(sns[0])          # component names don't depend on the subject
    comp_names    = comp_names + ['base']

    df = pd.DataFrame()
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, gl.rois[atlas_name]):

        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}',
            f'component_model.theta_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p')
        with open(path, 'rb') as f:
            theta_in = pickle.load(f)

        weight = np.asarray(np.exp(theta_in)).ravel()
        if len(weight) > len(comp_names):                    # trailing params are noise scales
            labels = comp_names + ['noise'] * (len(weight) - len(comp_names))
        else:
            labels = comp_names[:len(weight)]

        df_tmp = pd.DataFrame({'weight': weight, 'component': labels})
        df_tmp['sn']      = sn
        df_tmp['Hem']     = H
        df_tmp['roi']     = roi
        df_tmp['session'] = session
        df = pd.concat([df, df_tmp], ignore_index=True)

    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'component_model.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)
    return df


FUNC = {
    'G_rois'           : calc_G_rois,
    'G_force'          : calc_G_force,
    'dataframe_force'  : make_dataframe_force,
    'dataframe_rois'   : make_dataframe_rois,
    'correlation'      : correlation_between_sessions,
    'noise_ceiling'    : make_noise_ceiling_dataframe,
    'component_fit'    : fit_component_model_rois,
    'component_summary': make_component_model_dataframe,
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

