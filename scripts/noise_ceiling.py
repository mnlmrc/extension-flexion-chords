import os
import itertools
import numpy as np
import pandas as pd
import PcmPy as pcm
from scipy.stats import spearmanr

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.G_matrix import G_sorted


def rdm_vectors(sns, H, roi, session, glm, prefix='G_obs_raw', order=None):
    """Vectorised dissimilarity matrix of every participant, on a common chord order.

    Each participant's G is stored in their own trained-first order, so it is put
    through ``G_sorted`` first: without that, entry k of one participant's vector
    is a different chord pair than entry k of the next, and the correlations below
    are meaningless.

    Args:
        sns:     participant numbers, in the order the rows come out.
        H:       hemisphere.
        roi:     region name.
        session: session number.
        glm:     glm number.
        prefix:  which saved G to read (``G_obs_raw``, ``G_obs``, ``cov``, ...).
        order:   chord order to put every participant on. Defaults to ``gl.chordID``,
                 which matches chord *identity* across participants. Pass
                 ``get_trained_and_untrained(sn)`` framing instead to line the
                 participants up by training status rather than by chord.

    Returns:
        (n_subj, n_pairs) array, the lower triangle of each participant's RDM.
    """
    mask = np.tri(len(gl.chordID), k=-1, dtype=bool)

    rdms = []
    for sn in sns:
        G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}',
                    f'{prefix}.within_session.{session}.glm{glm}.{H}.{roi}.npy'))
        rdms.append(pcm.G_to_dist(G_sorted(G, sn, order=order))[mask])

    return np.array(rdms)


def corr(a, b, method='pearson'):
    """Correlation between two vectorised RDMs."""
    if method == 'spearman':
        return spearmanr(a, b)[0]
    return np.corrcoef(a, b)[0, 1]


def noise_ceiling(rdms, method='pearson'):
    """Leave-one-participant-out lower bound and the upper bound of the RSA noise ceiling.

    lower: each participant's RDM against the mean of the *other* participants. The
           group mean is estimated without them, so this is what a model that
           generalises across participants can be expected to reach.
    upper: the same, but against the mean of *all* participants, that participant
           included. The reference is fitted to the participant it is scored on, so
           it overestimates, and no model should beat it.

    Args:
        rdms:   (n_subj, n_pairs) vectorised RDMs, all on the same chord order.
        method: 'pearson' or 'spearman'.

    Returns:
        lower, upper: (n_subj,) arrays, one correlation per participant.
    """
    n     = len(rdms)
    if n < 3:
        raise ValueError(f'need at least 3 participants for a leave-one-out ceiling, got {n}')
    total = rdms.sum(axis=0)

    lower = np.array([corr(rdm, (total - rdm) / (n - 1), method) for rdm in rdms])
    upper = np.array([corr(rdm, total / n, method) for rdm in rdms])

    return lower, upper


def make_noise_ceiling_dataframe(glm, atlas_name='ROI', sns=None, prefix='G_obs_raw',
                                 method='pearson', order=None):
    """Noise ceiling of the crossnobis RDM for every hemisphere, roi and session.

    Returns one row per participant, so the ceiling of a given roi/Hem/session is
    the mean of its rows:

        df.groupby(['Hem', 'roi', 'session'])[['lower', 'upper']].mean()
    """
    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

    rows = []
    for H, roi, session in itertools.product(gl.Hem, rois, gl.sessions):

        print(f'doing {H}, {roi}, session {session}...')

        rdms         = rdm_vectors(sns, H, roi, session, glm, prefix=prefix, order=order)
        lower, upper = noise_ceiling(rdms, method=method)

        for i, sn in enumerate(sns):
            rows.append({'sn'     : sn,
                         'Hem'    : H,
                         'roi'    : roi,
                         'session': session,
                         'lower'  : lower[i],
                         'upper'  : upper[i]})

    return pd.DataFrame(rows)


if __name__=='__main__':
    sns        = gl.participants
    glm        = 3
    atlas_name = 'ROI'
    method     = 'pearson'

    nc = make_noise_ceiling_dataframe(glm, atlas_name, sns=sns, method=method)

    nc.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'noise_ceiling.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)

    print(nc.groupby(['Hem', 'roi', 'session'])[['lower', 'upper']].mean().round(3).to_string())
