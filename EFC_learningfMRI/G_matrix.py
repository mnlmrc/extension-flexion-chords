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
from EFC_learningfMRI.util import get_trained_and_untrained, runs_to_keep
from EFC_learningfMRI.geometry import split_trained


@dataclass
class Intercept:
    """Per-subject offset of within-session geometry from the group mean.

    For each (Hem, ROI, session) the within-session crossnobis distances are read
    (``G_obs.within_session`` -> ``G_to_dist``) and every subject is aligned to a
    common chord order so the group mean is over matched chords. Each subject's
    trained and untrained pairwise dissimilarities are then regressed on the
    group-mean dissimilarities of the *same* chords: ``intercept`` is how far that
    subject's geometry is shifted from the group, ``slope`` how it scales.

    The group mean is computed once per (Hem, ROI, session) and the chord ordering
    read once per subject, so ``dataframe`` loads each G exactly once.
    """

    sns: Sequence[int]      = field(default_factory=lambda: gl.participants)
    glm: int                = 3      # a default is required: a non-default field can't follow `sns`
    atlas_name: str         = 'ROI'
    Hem: Sequence[str]      = ('L', 'R')
    sessions: Sequence[int] = field(default_factory=lambda: gl.sessions)
    ref_session: int        = 3      # session whose group mean is the reference geometry


    # --- chord bookkeeping, read from participants.tsv once ------------------
    @cached_property
    def order(self):
        """{sn: trained-first chord IDs} — one participants.tsv read per subject."""
        return {sn: np.asarray(get_trained_and_untrained(sn), dtype=int) for sn in self.sns}
        

    @property
    def rois(self):
        return gl.rois[self.atlas_name]


    # --- io / maths ----------------------------------------------------------
    def _load(self, sess, H, roi):
        return np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.within_session.{sess}.glm{self.glm}.{H}.{roi}.npy'))


    @staticmethod
    def calc_intercept(subj, group):
        """Slope, intercept and r of one subject's dissimilarities on the group mean."""
        model = linregress(group, subj)
        return model.slope, model.intercept, model.rvalue


    def dataframe(self, fcn=pcm.G_to_dist):
        """Long-form table of per-subject slope/intercept vs the group mean.

        ``fcn`` maps a (n_subj, 8, 8) G to a (n_subj, 8, 8) matrix, e.g.
        ``pcm.G_to_dist`` (crossnobis) or ``pcm.G_to_cosine`` (cosine).
        """
        rows = []
        for H, roi, sess in itertools.product(self.Hem, self.rois, self.sessions):
            G_base = self._load(self.ref_session, H, roi)
            G_sess = self._load(sess, H, roi)
            for i, sn in enumerate(self.sns):
                # base            = G_sorted(G_base, self.sns, order=self.order[sn])
                keep            = np.arange(len(self.sns)) != i        # leave subject sn out of the group mean
                # M_group         = fcn(base)[keep].mean(axis=0)
                M_0             = fcn(G_base)[i]
                M_subj          = fcn(G_sess)[i]
                _, tr_0, untr_0     = split_trained(M_subj)                  
                _, tr_g, untr_g = split_trained(M_0)
                for chord, s, g in (('trained', tr_0, tr_g), ('untrained', untr_0, untr_g)):
                    slope, intercept, r = self.calc_intercept(s, g)
                    rows.append({'sn'       : sn, 
                                 'Hem'      : H,
                                 'roi'      : roi,
                                 'session'  : sess,
                                 'chord'    : chord,
                                 'slope'    : slope,
                                 'intercept': intercept,
                                 'r'        : r})
        return pd.DataFrame(rows)


def make_G_dataframe(glm, atlas_name='ROI', sns=None, ref_session=3, crossval=False):
    """Long-form table of within-session crossnobis and cosine dissimilarities.

    Each row is one chord pair for one subject/hemisphere/ROI/session. The pair is
    classified as ``trained`` (both chords trained), ``untrained`` (both untrained)
    or ``trained_untrained`` (one of each), and labelled with the two chord IDs,
    e.g. ``11911-22911``. Per subject the 8 chords follow
    :func:`get_trained_and_untrained` (first 4 trained, last 4 untrained), so the
    labels are read from that subject's own chord order.

    ``crossnobis_group`` / ``cosine_group`` add the across-subject mean of each
    metric for that chord pair in the reference session ``ref_session`` (default 3),
    pooled over the trained/untrained/mixed classes so every subject contributes. The
    same reference value is broadcast to every session, so a row's ``*_group`` is the
    ref-session group geometry for that pair regardless of the row's own session --
    the fixed reference to plot / regress each session's subject distances against.

    ``crossval`` makes that reference leave-one-subject-out: a subject's ``*_group`` is
    then the mean over the *other* subjects only, so it is never regressed against a
    group mean that contains its own data. Without it the group mean includes the
    subject, biasing subject-vs-group fits toward the diagonal.
    """
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
        G   = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.within_session.{sess}.glm{glm}.{H}.{roi}.npy'))
        D   = pcm.G_to_dist(G)
        cos = pcm.G_to_cosine(G)
        for i, sn in enumerate(sns):
            chords = get_trained_and_untrained(sn)
            for chord, (r, c) in idx.items():
                for ri, ci in zip(r, c):
                    rows.append({'sn'        : sn,
                                 'Hem'       : H,
                                 'roi'       : roi,
                                 'session'   : sess,
                                 'chord'     : chord,
                                 'pair'   : f'{chords[ri]}-{chords[ci]}',
                                 'crossnobis': D[i, ri, ci],
                                 'cosine'    : cos[i, ri, ci]})

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

    return df


def fit_lme(df, formula, re_formula=None, groups='sn', by=('Hem', 'roi', 'session')):
    """Fit one linear mixed model per cell of ``by`` and return a tidy fixed-effects table.

    The model is entirely caller-specified -- ``fit_lme`` makes no assumption about the
    columns or the design:

    * ``formula``    -- the fixed-effects patsy formula, e.g.
                        ``'crossnobis ~ crossnobis_group * C(chord)'`` for an interaction.
    * ``re_formula`` -- the random-effects structure passed to ``mixedlm``; ``None`` is a
                        random intercept, ``'~C(chord)'`` adds a random slope.
    * ``groups``     -- column identifying the random-effect groups (default ``'sn'``).
    * ``by``         -- columns whose every combination gets its own fit (default
                        ``('Hem', 'roi', 'session')``); pass e.g. ``('roi',)`` to pool
                        sessions, or ``()`` for a single model over the whole frame.

    Any reference-level coding or row filtering is the caller's job (set the ``chord``
    categorical, subset the rows) before calling. Terms keep their raw statsmodels names.

    Returns:
        One row per (cell, fixed-effect term) with ``beta``, ``se``, ``z``, ``p``,
        ``ci_low``, ``ci_high``, the first random-effect variance ``group_var``, whether
        the fit ``converged``, and ``n_obs``/``n_subj``.
    """
    by   = list(by)
    rows = []
    groups_iter = df.groupby(by, sort=False) if by else [((), df)]
    for keys, d in groups_iter:
        cell = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                          # convergence chatter, esp. random slopes
            m = smf.mixedlm(formula, d, groups=d[groups], re_formula=re_formula).fit()
        ci = m.conf_int()
        for term in m.fe_params.index:
            rows.append({**cell,
                         'term'     : term,
                         'beta'     : m.fe_params[term],
                         'se'       : m.bse[term],
                         'z'        : m.tvalues[term],
                         'p'        : m.pvalues[term],
                         'ci_low'   : ci.loc[term, 0],
                         'ci_high'  : ci.loc[term, 1],
                         'group_var': m.cov_re.iloc[0, 0],
                         'converged': m.converged,
                         'n_obs'    : int(m.nobs),
                         'n_subj'   : d[groups].nunique()})
    return pd.DataFrame(rows)


def lme_trained_untrained(df, metric='crossnobis', chords=('untrained', 'trained'),
                          formula=None, re_formula=None, groups='sn', by=('Hem', 'roi', 'session')):
    """Trained/untrained mixed-model contrast -- a thin :func:`fit_lme` wrapper.

    Sets up the chord contrast (keeps ``chords``, makes the first the reference level),
    then fits, by default,

        metric ~ 1 + {metric}_group + C(chord),   random intercept per subject,

    i.e. ``beta0 + beta1 * group_mean + beta2 * chord``. ``beta1`` scales with the
    across-subject group geometry; ``beta2`` is the offset of the second ``chords`` level
    relative to the first. Everything about the model is overridable:

    * ``re_formula='~C(chord)'`` -- random slope for chord.
    * ``formula='{y} ~ {x} * C(chord)'`` -- assess the group_mean x chord interaction
    * ``groups`` / ``by`` -- forwarded to :func:`fit_lme`.

    Term names are prettified: ``Intercept`` -> ``intercept``, ``{metric}_group`` ->
    ``group_mean``, ``C(chord)[T.*]`` -> ``chord`` (so an interaction reads
    ``group_mean:chord``). ``subject_var`` is the random-intercept variance. Otherwise the
    returned columns are exactly those of :func:`fit_lme`.
    """
    y, x    = metric, f'{metric}_group'
    formula = formula or f'{y} ~ {x} + C(chord)'

    data = df[df.chord.isin(chords)].copy()
    data['chord'] = pd.Categorical(data['chord'], categories=chords)  # first level = reference

    res = fit_lme(data, formula, re_formula=re_formula, groups=groups, by=by)

    # prettify term names by substitution, so interactions like `{x}:C(chord)[T.*]`
    # become `group_mean:chord` rather than being mislabelled.
    res['term'] = (res['term']
                   .str.replace('Intercept', 'intercept', regex=False)
                   .str.replace(x, 'group_mean', regex=False)
                   .str.replace(r'C\(chord\)\[T\.[^\]]+\]', 'chord', regex=True))
    return res.rename(columns={'group_var': 'subject_var'})




def G_sorted(G, sns, order=None):
    """Reorder the 8 chords (rows AND columns) of each subject's matrix.

    In G, subject s's rows/cols follow get_trained_and_untrained(sns[s]) — that
    subject's trained chord IDs first, then untrained. This permutes every subject
    onto a common `order` of chord IDs, so slot k holds the same chord for all.

    Args:
        G:     (n_subj, 8, 8) array; first axis aligned with `sns`.
        sns:   participant numbers, one per row of G.
        order: desired sequence of the 8 chord IDs. Any permutation is fine.
               Defaults to the chord IDs sorted numerically (canonical order).

    Returns:
        (n_subj, 8, 8) array, every subject reordered to `order`.
    """
    orders = [np.asarray(get_trained_and_untrained(sn), dtype=int) for sn in sns]
    target = np.sort(orders[0]) if order is None else np.asarray(order, dtype=int)

    out = np.empty_like(G)
    for s, current in enumerate(orders):
        if set(current.tolist()) != set(target.tolist()):
            raise ValueError(f"sn {sns[s]}: chord set {sorted(current)} != target {sorted(target)}")
        pos  = {c: i for i, c in enumerate(current.tolist())}   # chordID -> current index in G[s]
        perm = [pos[c] for c in target.tolist()]                # source index for each target slot
        out[s] = G[s][np.ix_(perm, perm)]
    return out


def _calc_G(data, cond_vec, part_vec, session='all', centred=False):
    """
    calc G matrix for runs in session

    ``centred`` removes each regressor's mean across voxels. It returns a new
    array rather than centring in place, so the same betas can be reused for
    another session.
    """
    if centred:
        data -= data.mean(axis=1, keepdims=True)

    keep     = runs_to_keep(session, part_vec.size)
    G_obs, _ = pcm.est_G_crossval(data[keep],
                                  cond_vec[keep],
                                  part_vec[keep],
                                  X=pcm.indicator(part_vec[keep]))
    return G_obs


def calc_G_rois(loader, sessions=('all', *gl.sessions)):
    """G of every ROI and session, with the subjects on the first dimension.

    Returns {(hemisphere, roi, session): (n_subj, n_cond, n_cond) array}, where
    n_cond is 8 within a session and 24 across sessions. Subjects come out in
    the order of ``loader.sns``.
    """
    glm = loader.glm
    
    G   = defaultdict(list)
    cov = defaultdict(list)
    for data in loader:
        for session in sessions:

            G_   = _calc_G(data.betas, data.cond_vec, data.part_vec, session, centred=False)
            cov_ = _calc_G(data.betas, data.cond_vec, data.part_vec, session, centred=True)
            
            G[data.Hem, data.roi, session].append(G_)

            cov[data.Hem, data.roi, session].append(cov_)

    for (H, roi, session), G_ in G.items():
        if session=='all':
            np.save(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.across_session.glm{glm}.{H}.{roi}'), G_)
        else:
            np.save(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.within_session.{session}.glm{glm}.{H}.{roi}'), G_)

    for (H, roi, session), cov_ in cov.items():
        if session=='all':
            np.save(os.path.join(gl.baseDir, gl.pcmDir, f'cov.across_session.glm{glm}.{H}.{roi}'), cov_)
        else:
            np.save(os.path.join(gl.baseDir, gl.pcmDir, f'cov.within_session.{session}.glm{glm}.{H}.{roi}'), cov_)

