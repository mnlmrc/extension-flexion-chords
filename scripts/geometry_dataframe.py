import os
import itertools
import numpy as np
import pandas as pd
import PcmPy as pcm
# from EFC_learningfMRI.geometry import split_trained
# from EFC_learningfMRI.G_matrix import lme_trained_untrained
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import get_trained_and_untrained



def shuffle_trained_untrained(df, rng):
    """Give each subject a random 4-of-8 'trained' set and reclassify every pair."""
    df[['a', 'b']] = df.pair.str.split('-', expand=True)
    out            = df.copy()
    for sn, g in df.groupby('sn'):
        tr   = set(rng.choice(np.array(gl.chordID, dtype=str), 4, replace=False))   # this subject's trained chords
        a, b = g.a.isin(tr), g.b.isin(tr)
        out.loc[g.index, 'chord'] = np.where(a & b, 'trained', np.where(~a & ~b, 'untrained', 'trained_untrained'))
    return out



def make_G_dataframe(glm, atlas_name='ROI', sns=None, ref_session=3, crossval=False):

    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

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
    idx   = {chord: np.where(m) for chord, m in masks.items()}

    rows = []
    for H, roi, sess in itertools.product(gl.Hem, rois, gl.sessions):
        for sn in sns:
            
            print(f'doing participant {sn}, session {session}, {H}, {roi}...')

            chords = get_trained_and_untrained(sn)
            G   = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.glm{glm}.{H}.{roi}.npy'))
            D   = pcm.G_to_dist(G)
            cos = pcm.G_to_cosine(G)
            for chord, (r, c) in idx.items():
                for ri, ci in zip(r, c):
                    rows.append({'sn'        : sn,
                                 'Hem'       : H,
                                 'roi'       : roi,
                                 'session'   : sess,
                                 'chord'     : chord,
                                 'pair'      : f'{chords[ri]}-{chords[ci]}',
                                 'crossnobis': D[ri, ci],
                                 'cosine'    : cos[ri, ci]})

    df = pd.DataFrame(rows)

    df['pair'] = df.pair.map(lambda s: '-'.join(sorted(s.split('-'))))

    # group-mean geometry: across-subject mean of each chord pair in the reference
    # session, pooled over trained/untrained/mixed (no 'chord'/'session' in the key,
    # so all subjects contribute). Merged onto every row, so the *_group columns hold
    # the ref-session reference for every session.
    keys = ['Hem', 'roi', 'pair']
    s3   = df[df.session == ref_session]
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


if __name__=='__main__':
    sns        = gl.participants
    glm        = 3
    atlas_name = 'ROI'

    within_long = make_G_dataframe(glm, atlas_name, crossval=True)

    within_long.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)

    # # permutation loop: shuffle labels -> refit -> append
    # rng    = np.random.default_rng(0)
    # n_perm = 1000
    # rows   = []
    # for i in range(n_perm):
    #     print(f'{i + 1}/{n_perm}') if (i + 1) % (n_perm // 10) == 0 else None
    #     dist_perm    = shuffle_trained_untrained(within_long, rng)
    #     res_perm_tmp = lme_trained_untrained(dist_perm)
    #     rows.append(res_perm_tmp)

    # result_perm = pd.concat(rows, axis=0)

    # result_perm.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_perm.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


