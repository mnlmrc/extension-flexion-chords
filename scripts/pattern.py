import argparse
import os
from collections import defaultdict
import numpy as np
import PcmPy as pcm
import pandas as pd
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.G_matrix import calc_G
from EFC_learningfMRI.util import get_trained_and_untrained
from EFC_learningfMRI.betas import BetasPrewithenedLoader
from EFC_learningfMRI.behaviour import force_patterns
import warnings
import itertools

def _pair_index():
    """(row, col) indices of the chord pairs, split into the three pair groups.

    Rows/cols 0-3 are the trained chords and 4-7 the untrained ones, the layout
    every G in the pcm directory uses, so the same indices apply to a neural and
    a force G alike.
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


def _add_group_reference(rows, keys, ref_session, crossval):
    """Assemble the rows and attach the reference-session group geometry.

    ``keys`` identifies a chord pair *within one G family* — ['Hem', 'roi', 'pair']
    for the neural Gs, ['metric', 'pair'] for the force ones.
    """
    df = pd.DataFrame(rows)

    df['pair'] = df.pair.map(lambda s: '-'.join(sorted(s.split('-'))))

    # group-mean geometry: across-subject mean of each chord pair in the reference
    # session, pooled over trained/untrained/mixed (no 'chord'/'session' in the key,
    # so all subjects contribute). Merged onto every row, so the *_group columns hold
    # the ref-session reference for every session.
    s3 = df[df.session == ref_session]
    if crossval:
        # leave-one-subject-out: subtract each subject's own value from the pair sum,
        # so their reference is the mean over the other subjects. Keyed by subject too,
        # then merged so it broadcasts across that subject's sessions.
        ref = s3[keys + ['sn']].copy()
        for metric in ('crossnobis', 'cosine'):
            g = s3.groupby(keys)[metric]
            ref[f'{metric}_group'] = (g.transform('sum') - s3[metric]) / (g.transform('size') - 1)
        df = df.merge(ref, on=keys + ['sn'], how='left')
    else:
        ref = (s3.groupby(keys, as_index=False)
                 .agg(crossnobis_group=('crossnobis', 'mean'),
                      cosine_group    =('cosine',     'mean')))
        df  = df.merge(ref, on=keys, how='left')

    # calculate angle from cosine
    df['theta']       = np.arccos(df.cosine)
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

    df = _add_group_reference(rows, ['Hem', 'roi', 'pair'], ref_session, crossval)
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

    df = _add_group_reference(rows, ['metric', 'pair'], ref_session, crossval)
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


# Step name -> function. Each step builds its own input, so they can be run
# separately: the ROI Gs need the betas, the force Gs only force.run.wide.tsv.
FUNC = {
    'G_rois'          : calc_G_rois,
    'G_force'         : calc_G_force,
    'dataframe_force' : make_G_dataframe_force,
    'dataframe_rois'  : make_G_dataframe_rois
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

    # only the flags actually given are forwarded, so a step never sees a keyword it
    # does not take (--glm, say, reaches G_rois and dataframe_neural but not the force steps)
    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('G_force')
        main('dataframe_force')
