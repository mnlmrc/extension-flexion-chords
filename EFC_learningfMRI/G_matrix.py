import os
import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence
import statsmodels.formula.api as smf   # heavy + optional; import only when used
import numpy as np
import pandas as pd
import PcmPy as pcm
from scipy.stats import linregress

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import get_trained_and_untrained, runs_to_keep, split_trained_untrained


def G_sorted(G, sns, order=None):
    """Reorder the 8 chords (rows AND columns) of one or more subjects' matrices.

    In G, subject s's rows/cols follow get_trained_and_untrained(sns[s]) — that
    subject's trained chord IDs first, then untrained. This permutes every subject
    onto a common `order` of chord IDs, so slot k holds the same chord for all.

    Args:
        G:     (8, 8) for a single subject, or (n_subj, 8, 8) with the first axis
               aligned with `sns`.
        sns:   participant number(s): a scalar for a 2D `G`, one per row for a 3D `G`.
        order: desired sequence of the 8 chord IDs. Any permutation is fine.
               Defaults to `gl.chordID` (the IDs sorted numerically), which is the
               same for every subject — so the default framing never depends on `sns`.

    Returns:
        Array shaped like `G` — (8, 8) in, (8, 8) out — reordered to `order`.
    """
    G      = np.asarray(G)
    sns    = np.atleast_1d(sns)
    single = G.ndim == 2                      # remember, so we hand back a 2D result

    if G.ndim not in (2, 3) or G.shape[-1] != G.shape[-2]:
        raise ValueError(f"G must be (8, 8) or (n_subj, 8, 8), got shape {G.shape}")
    if single:
        G = G[None]

    if len(sns) != len(G):
        raise ValueError(f"got {len(sns)} participant(s) for {len(G)} matrix/matrices")

    order = np.asarray(gl.chordID if order is None else order, dtype=int)

    chords_per_subj = [np.asarray(get_trained_and_untrained(sn), dtype=int) for sn in sns]

    if order.size != G.shape[1]:
        raise ValueError(f"order lists {order.size} chords but G is {G.shape[1]}x{G.shape[1]}")

    out = np.empty_like(G)
    for s, (sn, chords) in enumerate(zip(sns, chords_per_subj)):
        if set(chords.tolist()) != set(order.tolist()):
            raise ValueError(f"sn {sn}: chord set {sorted(chords.tolist())} "
                             f"!= order {sorted(order.tolist())}")
        slot_of = {chord: i for i, chord in enumerate(chords.tolist())}  # chordID -> its index in G[s]
        perm    = [slot_of[chord] for chord in order.tolist()]          # source index for each target slot
        out[s]  = G[s][np.ix_(perm, perm)]

    return out[0] if single else out


def calc_G(data, cond_vec, part_vec, session='all', repetition='all', centred=False):
    """
    calc G matrix for runs in session

    ``centred`` removes each regressor's mean across voxels. It returns a new
    array rather than centring in place, so the same betas can be reused for
    another session.

    ``repetition`` restricts the estimate to a single repetition; it is read out of
    ``cond_vec``, so it only works for a glm whose regressors carry a repetition
    field (``'session,repetition,chordID'``, e.g. glm2).
    """
    if centred:
        data = data - data.mean(axis=1, keepdims=True)

    rep_vec = None
    if repetition != 'all':
        parts = pd.Series(cond_vec).str.split(',', expand=True)
        if parts.shape[1] < 3:
            raise ValueError("cond_vec has no repetition field (expected 'session,repetition,chordID')")
        rep_vec = parts[1].astype(int).to_numpy()

    keep     = runs_to_keep(part_vec.size, session=session, repetition=repetition, rep_vec=rep_vec)
    G_obs, _ = pcm.est_G_crossval(data[keep],
                                  cond_vec[keep],
                                  part_vec[keep],
                                  X=pcm.indicator(part_vec[keep]))
    return G_obs


def G_scaling(G_ref, G_tar):
    """Full comparison of observed vs. scaling-predicted target dissimilarity.

    Returns a one-row DataFrame with all intermediate quantities and the column
    `residual` (observed - predicted), which is the statistic tested in the paper.

    Parameters
    ----------
    G_ref, G_tar : (K, K) array_like

    Returns
    -------
    pandas.DataFrame, one row, with columns:
        act_ref, act_target, scale,
        diss_ref, diss_target_observed, diss_pred,
        residual  (= diss_target_observed - diss_pred)
    """
    K = G_ref.shape[0]
    
    act_ref = np.sqrt(np.diag(G_ref)).mean()
    act_tar = np.sqrt(np.diag(G_tar)).mean()

    scale = act_tar / act_ref

    D_ref = pcm.G_to_dist(G_ref)
    D_tar = pcm.G_to_dist(G_tar)
 
    diss_ref = np.sqrt(D_ref.mean())
    diss_obs = np.sqrt(D_tar.mean())
    diss_pred = scale * diss_ref
 
    return pd.DataFrame({
        "act_ref"             : act_ref,
        "act_target"          : act_tar,
        "scale"               : scale,
        "diss_ref"            : diss_ref,
        "diss_target_observed": diss_obs,
        "diss_pred"           : diss_pred,
        "residual"            : diss_obs - diss_pred,
    }, index=[0])