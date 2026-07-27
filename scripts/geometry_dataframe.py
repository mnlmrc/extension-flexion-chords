import os
import itertools
import numpy as np
import pandas as pd
import PcmPy as pcm
from EFC_learningfMRI.geometry import split_trained
from EFC_learningfMRI.G_matrix import Intercept, make_G_dataframe
import EFC_learningfMRI.globals as gl

N_CHORDS = 8


def session_block(M, s1=0, s2=0):
    """The 8 chords of session s1 (rows) against the 8 chords of session s2 (columns).

    Conditions are ordered session-major, so session s occupies 8*s to 8*s+8.
    The block always has the trained chords first, which is what split_trained
    expects. A within-session matrix holds one session, hence the default s1=s2=0.
    """
    return M[N_CHORDS*s1:N_CHORDS*(s1 + 1), N_CHORDS*s2:N_CHORDS*(s2 + 1)]


def crossnobis(G, s1=0, s2=0):
    """One row per chord set, with its mean euclidean and cosine dissimilarity.
    
    """
    
    dist   = pcm.G_to_dist(G)
    M_dist = session_block(dist, s1, s2)

    _, dist_trained, dist_untrained = split_trained(M_dist)

    return {'chord': 'trained',   'dist': dist_trained.mean()}, {'chord': 'untrained', 'dist': dist_untrained.mean()}


def cosine(G, s1=0, s2=0):
    """One row per chord set, with its mean euclidean and cosine dissimilarity.
    
    """
    
    cos    = pcm.G_to_cosine(G)
    M_cos  = session_block(cos, s1, s2)

    if (s1>0) or (s2>0):
        M_cos_diag = np.diag(M_cos)
        cos_trained, cos_untrained = M_cos_diag[:4], M_cos_diag[4:]
    else:
        _, cos_trained, cos_untrained   = split_trained(M_cos)

    return {'chord': 'trained', 'cos': cos_trained.mean()}, {'chord': 'untrained', 'cos': cos_untrained.mean()}


def within_session(sns, glm, atlas_name):
    """Dissimilarity within the trained and within the untrained chords of each session."""

    rows = []
    for H, roi, sess in itertools.product(gl.Hem, gl.rois[atlas_name], gl.sessions):
        G_obs = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.within_session.{sess}.glm{glm}.{H}.{roi}.npy'))
        for sn, G in zip(sns, G_obs):
            for dist, cos in zip(crossnobis(G), cosine(G)):
                rows.append({'sn': sn, 'Hem': H, 'roi': roi, 'session': sess, **dist})
                rows.append({'sn': sn, 'Hem': H, 'roi': roi, 'session': sess, **cos})

    return pd.DataFrame(rows)


def across_session(sns, glm, atlas_name):
    """Dissimilarity within the trained and within the untrained chords, between two sessions."""

    rows = []
    for H, roi in itertools.product(gl.Hem, gl.rois[atlas_name]):
        cov = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'cov.across_session.glm{glm}.{H}.{roi}.npy'))
        for sn, cov_ in zip(sns, cov):
            for s1, s2 in itertools.combinations(range(len(gl.sessions)), 2):
                for cos in cosine(cov_, s1, s2):
                    rows.append({'sn': sn, 'Hem': H, 'roi': roi, 'session': f'{gl.sessions[s1]}-{gl.sessions[s2]}', **cos})

    return pd.DataFrame(rows)


def shuffle_trained_untrained(df, rng):
    """Give each subject a random 4-of-8 'trained' set and reclassify every pair."""
    df[['a', 'b']] = df.pair.str.split('-', expand=True)
    out            = df.copy()
    for sn, g in df.groupby('sn'):
        tr   = set(rng.choice(np.array(gl.chordID, dtype=str), 4, replace=False))   # this subject's trained chords
        a, b = g.a.isin(tr), g.b.isin(tr)
        out.loc[g.index, 'chord'] = np.where(a & b, 'trained', np.where(~a & ~b, 'untrained', 'trained_untrained'))
    return out


if __name__=='__main__':
    sns        = gl.participants
    glm        = 3
    atlas_name = 'ROI'

    within_long = make_G_dataframe(glm, atlas_name, crossval=True)

    within_long.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


    # within = within_session(sns, glm, atlas_name)
    # across = across_session(sns, glm, atlas_name)

    # within.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)
    # across.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.across_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)

    # intercept = Intercept(sns, glm)
    # dist      = intercept.dataframe(pcm.G_to_dist)
    # cos       = intercept.dataframe(pcm.G_to_cosine)

    # dist.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'intercept.within_session.crossnobis.ROI.glm{glm}.tsv'), sep='\t', index=False)

    # cos.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'intercept.within_session.cosine.ROI.glm{glm}.tsv'), sep='\t', index=False)
