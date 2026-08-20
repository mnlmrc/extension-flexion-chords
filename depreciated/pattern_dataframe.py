import os
import itertools
import numpy as np
import pandas as pd
import PcmPy as pcm
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import get_trained_and_untrained


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


def make_G_dataframe_neural(glm, atlas_name='ROI', sns=None, ref_session=3, crossval=False):

    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

    rows = []
    for H, roi, sess in itertools.product(gl.Hem, rois, gl.sessions):
        for sn in sns:

            print(f'doing participant {sn}, session {sess}, {H}, {roi}...')

            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.glm{glm}.{H}.{roi}.npy'))
            rows += _G_rows(G, sn, Hem=H, roi=roi, session=sess)

    return _add_group_reference(rows, ['Hem', 'roi', 'pair'], ref_session, crossval)


def make_G_dataframe_force(metrics=('abs', 'der', 'der_peak'), sns=None, ref_session=3, crossval=False):
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

    return _add_group_reference(rows, ['metric', 'pair'], ref_session, crossval)


if __name__=='__main__':
    sns        = gl.participants
    glm        = 3
    atlas_name = 'ROI'

    within_long = make_G_dataframe_neural(glm, atlas_name, crossval=True)
    within_long.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)

    within_force = make_G_dataframe_force(crossval=True)
    within_force.to_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t', index=False)

    


