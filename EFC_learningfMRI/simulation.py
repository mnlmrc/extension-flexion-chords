"""Simulate single-ROI activity patterns for testing geometry hypotheses.

Lightweight simulation: no real brain, no real ROI masks. For each simulated subject it
generates the beta patterns of ONE ROI and saves them as a CIFTI ``beta.dscalar.nii`` on a
tiny **synthetic** brain axis (``n_vox`` voxels laid out on an ``(n_vox, 1, 1)`` grid,
unrelated to any real subject), next to an all-ones dummy ROI ``mask.nii``. Extract the ROI
exactly like real data -- ``volume_from_cifti`` + ``sample_image`` at the mask coords, which
:func:`load` does for you -- then compute G with ``pcm.est_G_crossval``; the voxels are
white noise, so no prewhitening is needed.

``Simulation`` generates **null (H0)** data only, and is **split-agnostic**: 8 ground-truth
chord patterns shared by the group, one fresh noisy measurement of all 8 per run, no
trained/untrained structure baked in. The trained/untrained split is taken from
``participants.tsv`` (via ``util.get_trained_and_untrained``) at load time -- **exactly as for
real subjects** -- so add rows for the sim subjects there (this module never touches
participants.tsv). Because generation carries no split, you can run it before adding those
rows. Alternative hypotheses are separate ``h1_*`` functions that grab the H0 data and
manipulate it into the target effect (e.g. :func:`h1_trained_hypoactive`), keeping one clean
null with each manipulation explicit.

Labels live in a glm3-style ``reginfo.tsv`` (columns ``sn, run, name`` with name
``"{chordID},sessNN"`` using the real ``gl.chordID``, chord-id-sorted rows -- indistinguishable
in format from a real glm3 reginfo). From it plus participants.tsv, :func:`load` recovers a
``util.RegInfo``-convention ``cond_vec`` (``"{session},{slot}"``, slot = trained-first chord
index, 0-3 trained / 4-7 untrained), ``part_vec`` (session-blocked run numbers) and ``chords``
(trained-first chord order, the analog of ``util.get_trained_and_untrained``). So
``util.runs_to_keep``, ``G_matrix._calc_G`` and ``geometry.split_trained`` work unchanged.

Data for simulation ``N`` lands in ``{gl.baseDir}/sim{N}/subj{sn}/`` as ``beta.dscalar.nii``
+ ``mask.nii`` + ``reginfo.tsv``.

Examples
--------
Null hypothesis -- trained and untrained come from the same ground truth::

    from EFC_learningfMRI.simulation import Simulation, load

    sns = list(range(9001, 9011))               # 10 simulated participants
    sim = Simulation(sim=1, sim_sns=sns)         # H0 only: no effect to configure
    sim.run()                                    # writes beta.dscalar.nii + mask.nii + labels

Compute G per session and look at the trained vs untrained blocks -- under the null they
have the same expected geometry::

    import PcmPy as pcm
    from EFC_learningfMRI.util import runs_to_keep
    from EFC_learningfMRI.geometry import split_trained

    betas, cond_vec, part_vec, chords = load(1, 9001)   # extracts via the dummy mask
    keep = runs_to_keep(part_vec.size, session=3)       # session-3 rows
    G, _ = pcm.est_G_crossval(betas[keep], cond_vec[keep], part_vec[keep],
                              X=pcm.indicator(part_vec[keep]))
    _, trained, untrained = split_trained(pcm.G_to_dist(G))   # trained ~ untrained

To test an alternative, derive it from the H0 data with an ``h1_*`` function -- e.g. trained
chords eliciting less activity than untrained (read sim 1, write sim 2)::

    from EFC_learningfMRI.simulation import h1_trained_hypoactive

    h1_trained_hypoactive(1, 2, amount=1.0)
"""

import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence

import numpy as np
import pandas as pd
import nibabel as nb
import nitools as nt

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import get_trained_and_untrained


@dataclass
class Simulation:
    """Generate and save null (H0) single-ROI betas (as CIFTI) for a set of subjects.

    Attributes
    ----------
    sim            : simulation id; data goes to ``{gl.baseDir}/sim{sim}/subj{sn}/``.
    sns            : subject numbers to create.
    sessions       : session ids (written into ``reginfo.name``); under H0 they are identical.
    n_run, n_vox   : runs per session and voxels in the ROI (the synthetic brain size).
    signal, noise  : ground-truth pattern amplitude and per-measurement noise sd.
    subj_var       : >0 adds participant-specific deviations to the true patterns.
    seed           : seeds the shared ground truth and each subject's draws.
    """

    sim     : int           = 1
    sns     : Sequence[int] = field(default_factory=lambda: list(range(9001, 9011)))
    sessions: Sequence[int] = field(default_factory=lambda: list(gl.sessions))
    n_run   : int           = 10
    n_vox   : int           = 100
    n_cond  : int           = 8
    n_train : int           = 4                                                       # split_trained is hard-wired to a 4/4 split
    signal  : float         = 1.0
    noise   : float         = 1.0
    subj_var: float         = 0.0
    seed    : int           = 0
    dtype   : type          = np.float32

    # --- shared design and ground truth --------------------------------------
    @cached_property
    def _design(self):
        """Row-wise (session index, global run number, chord index), session-blocked.

        The chord index runs 0..n_cond-1 in ``gl.chordID`` order (chord-id-sorted, as in a real
        glm3 reginfo); the trained/untrained split is *not* part of the design -- it is read from
        participants.tsv at load time, exactly as for real subjects.
        """
        sess, run, chord = [], [], []
        for k in range(len(self.sessions)):
            for r in range(self.n_run):
                for c in range(self.n_cond):
                    sess.append(k)
                    chord.append(c)
                    run.append(k * self.n_run + r + 1)
        return np.array(sess), np.array(run), np.array(chord)

    @cached_property
    def ground_truth(self):
        """(n_cond, n_vox) intrinsic chord patterns (``gl.chordID`` order), shared across subjects."""
        rng = np.random.default_rng(self.seed)
        return self.signal * rng.standard_normal((self.n_cond, self.n_vox))

    # --- per subject ---------------------------------------------------------
    def simulate_subject(self, sn):
        """Return the H0 betas for one subject (rows in chord-id order, session -> run -> chord).

        Pure null: each run is an independent noisy measurement of the shared 8 chord patterns.
        No trained/untrained structure is baked in -- that labelling comes from participants.tsv.
        """
        rng   = np.random.default_rng(np.random.SeedSequence([self.seed, sn]))
        U     = self.ground_truth + self.subj_var * rng.standard_normal((self.n_cond, self.n_vox))
        betas = [U + self.noise * rng.standard_normal((self.n_cond, self.n_vox)) for _ in range(len(self.sessions) * self.n_run)]        # one measurement of all chords per run
        return np.concatenate(betas).astype(self.dtype)

    def reginfo(self, sn):
        """glm3-style reginfo for one subject: columns ``sn, run, name`` (name = ``"{chordID},sessNN"``).

        Rows are in chord-id order with the real ``gl.chordID``, so the file is indistinguishable
        in format from a real glm3 reginfo and carries no trained/untrained information (that is in
        participants.tsv). Row order matches the betas.
        """
        sess_i, run, chord = self._design
        session = np.asarray(self.sessions)[sess_i]
        chordID = np.asarray(gl.chordID)[chord]
        name    = [f'{c},sess{s:02d}' for c, s in zip(chordID, session)]
        return pd.DataFrame({'sn': sn, 'run': run, 'name': name})

    def run(self):
        """Generate and save every subject (betas + reginfo). Returns the subject numbers.

        Split-agnostic: does not need participants.tsv. Add the sim rows there before analysing.
        """
        for sn in self.sns:
            save(self.sim, sn, self.simulate_subject(sn), self.reginfo(sn))
            print(f'sim subject {sn}: betas + reginfo written')
        return list(self.sns)


# --- synthetic brain ---------------------------------------------------------
def _brain_axis(n_vox):
    """A BrainModelAxis for ``n_vox`` voxels on a tiny synthetic ``(n_vox, 1, 1)`` grid."""
    return nb.cifti2.BrainModelAxis.from_mask(np.ones((n_vox, 1, 1), np.int8), name='cortex_left', affine=np.eye(4))


def _sim_dir(sim):
    return os.path.join(gl.baseDir, f'glm{sim + 100}')


# --- io ----------------------------------------------------------------------
def save(sim, sn, betas, reginfo):
    """Write one subject: betas as a CIFTI on a synthetic brain + all-ones dummy mask + reginfo.

    ``beta.dscalar.nii`` (row axis = ``reginfo.name``), ``mask.nii`` (all ones over the whole
    synthetic brain, so masking it recovers every voxel as one ROI) and ``reginfo.tsv`` land in
    ``{gl.baseDir}/sim{sim}/subj{sn}/``. ``reginfo.tsv`` is the single label source -- ``load``
    recovers ``cond_vec`` / ``part_vec`` / ``chords`` from it, mirroring how the real pipeline
    reads labels from its own reginfo.
    """
    path_sim = os.path.join(_sim_dir(sim), f'subj{sn}')
    os.makedirs(path_sim, exist_ok=True)
    betas = np.asarray(betas, dtype=np.float32)
    n_vox = betas.shape[1]

    # make cifti
    row   = nb.cifti2.ScalarAxis([str(x) for x in reginfo['name']])
    cifti = nb.Cifti2Image(betas, header=nb.Cifti2Header.from_axes((row, _brain_axis(n_vox))))
    nb.save(cifti, os.path.join(path_sim, 'beta.dscalar.nii'))

    # make sim roi mask
    path_roi = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}',)
    os.makedirs(path_roi, exist_ok=True)
    mask = np.ones((n_vox, 1, 1), np.int8)
    mask_vol = nb.Nifti1Image(mask, np.eye(4))
    nb.save(mask_vol, os.path.join(path_roi, 'SIM.mask.nii'))

    # save sim reginfo
    reginfo.to_csv(os.path.join(path_sim, 'reginfo.tsv'), sep='\t', index=False)


def _parse_reginfo(reginfo):
    """Recover (cond_vec, part_vec, chords) from a sim ``reginfo``, using participants.tsv.

    Mirrors ``util.RegInfo``: the trained-first slot of each chord id comes from
    ``get_trained_and_untrained`` (participants.tsv), exactly as for real subjects, so the sim
    row order (chord-id-sorted) is irrelevant to the split. ``cond_vec`` is the RegInfo-style
    ``"{session_index},{slot}"`` label, ``part_vec`` the run numbers, ``chords`` the trained-first
    chord order as ``gl.chordID`` indices (0..7).
    """
    sn       = int(reginfo['sn'].iloc[0])
    part_vec = reginfo['run'].to_numpy()
    parts    = reginfo['name'].str.split(',', expand=True)
    chordID  = parts[0].astype(int).to_numpy()                           # real chord id per row
    session  = parts[1].str[len('sess'):].astype(int).to_numpy()         # 3 / 9 / 23 per row

    trained_first = np.asarray(get_trained_and_untrained(sn), dtype=int)  # trained-first chord ids
    slot_of  = {int(c): i for i, c in enumerate(trained_first)}           # chord id -> trained-first slot
    slot     = np.array([slot_of[c] for c in chordID])
    sess_idx = {s: i + 1 for i, s in enumerate(dict.fromkeys(session.tolist()))}
    cond_vec = np.array([f'{sess_idx[s]},{sl}' for s, sl in zip(session, slot)])

    to_abstract = {int(c): i for i, c in enumerate(gl.chordID)}           # real chord id -> 0..7
    chords   = np.array([to_abstract[c] for c in trained_first])
    return cond_vec, part_vec, chords


def load(sim, sn):
    """Load one subject: ROI betas (extracted from the CIFTI via the dummy mask) + labels."""
    cifti                      = nb.load(os.path.join(_sim_dir(sim), f'subj{sn}', 'beta.dscalar.nii'))
    vol                        = nt.volume_from_cifti(cifti, struct_names=gl.struct_cortex)
    mask                       = nb.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', 'SIM.mask.nii'))
    coords                     = nt.get_mask_coords(mask)
    betas                      = nt.sample_image(vol, coords[0], coords[1], coords[2], interpolation=0).T
    cond_vec, part_vec, chords = _parse_reginfo(pd.read_csv(os.path.join(_sim_dir(sim), f'subj{sn}', 'reginfo.tsv'), sep='\t'))
    return betas, cond_vec, part_vec, chords


def _subjects_in(sim):
    """Subject numbers found as ``subj*`` folders under ``sim{sim}``."""
    d = _sim_dir(sim)
    return sorted(int(x[len('subj'):]) for x in os.listdir(d) if x.startswith('subj'))


# --- derived files -----------------------------------------------------------
def make_sim_contrasts(sim, sns=None):
    """Average each condition across runs and write ``contrast.dscalar.nii`` per subject.

    Mirrors the real ``contrast.dscalar.nii``: one frame per unique condition
    (``"{session},{slot}"``, in first-appearance / session-blocked order), each the mean beta
    over that condition's runs. Written on the same synthetic brain axis as
    ``beta.dscalar.nii``, into ``{gl.baseDir}/sim{sim}/subj{sn}/``.
    """
    for sn in (sns if sns is not None else _subjects_in(sim)):
        betas, cond_vec, _, _ = load(sim, sn)
        conds    = list(dict.fromkeys(cond_vec.tolist()))               # unique, order preserved
        contrast = np.stack([betas[cond_vec == c].mean(0) for c in conds]).astype(np.float32)

        row   = nb.cifti2.ScalarAxis([str(c) for c in conds])
        cifti = nb.Cifti2Image(contrast, header=nb.Cifti2Header.from_axes((row, _brain_axis(contrast.shape[1]))))
        nb.save(cifti, os.path.join(_sim_dir(sim), f'subj{sn}', 'contrast.dscalar.nii'))
        print(f'sim subject {sn}: contrast {contrast.shape}')


def _trained_rows(cond_vec, n_train=4):
    """Boolean mask of the trained-chord measurements (slot < n_train)."""
    return np.array([int(c.split(',')[1]) < n_train for c in cond_vec])


# --- alternative hypotheses: transform the H0 data ---------------------------
def h1_trained_hypoactive(in_sim, out_sim, sns=None, amount=1.0, n_train=4):
    """H1: trained chords are *less active* than untrained.

    Reads the H0 simulation ``in_sim`` and subtracts ``amount`` from every voxel of the
    trained-chord measurements -- a uniform drop in activation -- writing the result to
    ``out_sim`` (``cond_vec`` / ``part_vec`` / ``chords`` unchanged). This lowers the
    univariate activation of the trained block while leaving its *within-block* crossnobis
    geometry intact (a constant offset cancels in pairwise differences), so it is picked up
    by an activation analysis but not by within-block distances.
    """
    for sn in (sns if sns is not None else _subjects_in(in_sim)):
        betas, cond_vec, _, _ = load(in_sim, sn)
        reginfo = pd.read_csv(os.path.join(_sim_dir(in_sim), f'subj{sn}', 'reginfo.tsv'), sep='\t')
        betas = betas.copy()
        betas[_trained_rows(cond_vec, n_train)] -= amount
        save(out_sim, sn, betas, reginfo)                       # labels unchanged: pass reginfo through
        print(f'sim subject {sn}: trained activity -{amount}')
