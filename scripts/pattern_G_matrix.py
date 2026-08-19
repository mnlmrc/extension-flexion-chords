import os
from collections import defaultdict
import numpy as np
import PcmPy as pcm
import pandas as pd
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.G_matrix import calc_G
from EFC_learningfMRI.util import get_trained_and_untrained
from EFC_learningfMRI.betas import BetasPrewithenedLoader
import warnings


def _make_fname(session, repetition):
    """The 'within_session.3.1' / 'across_session' part of a G filename."""
    fname = 'across_session' if session == 'all' else 'within_session'
    fname = fname + '.' + str(session) if session != 'all' else fname
    fname = fname + '.' + str(repetition) if repetition != 'all' else fname
    return fname

FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinkie']

def force_patterns(force, sn, metric, session='all', repetition='all'):
    """Force patterns of one participant, laid out like the ROI betas.

    The five fingers take the place of the voxels, so the returned triple can be
    handed straight to :func:`calc_G`. The layout mirrors the neural pipeline:
    chords are relabelled 0-7 with this participant's *trained* chords first (see
    ``get_trained_and_untrained``), and the conditions are the same
    ``'session,chord'`` strings ``RegInfo.cond_vec`` builds — so the G comes out
    8x8 within a session and 24x24 across, with the same rows in the same order
    as the ROI Gs.

    ``session`` and ``repetition`` are filtered here rather than in ``calc_G``:
    ``runs_to_keep`` splits the rows into three equal positional blocks, which the
    force table does not satisfy (failed trials are missing, and session 23 has an
    extra run). Pass the result to ``calc_G`` with its own filters left at 'all'.

    Args:
        force:      the table from :func:`load_force`.
        sn:         participant number.
        metric:     force measure, one of 'abs', 'der', 'der_peak'.
        session:    3, 9, 23, or 'all'.
        repetition: 1, 2, or 'all'. With 'all' the two repetitions of a run are
                    averaged, giving one pattern per run x chord.

    Returns:
        (data, cond_vec, part_vec): ``data`` is (n_obs, 5), ``cond_vec`` the
        condition strings and ``part_vec`` the run numbers, made unique across
        sessions (BN restarts at 1 in every session).
    """
    cols = [f'{finger}_{metric}' for finger in FINGERS]

    df = force[force.subNum == sn]
    if session != 'all':
        df = df[df.session == session]
    if repetition != 'all':
        df = df[df.Repetition == repetition]

    # one pattern per run x chord: averages the two repetitions when both are kept,
    # and is a no-op once a single repetition has been selected
    df = df.groupby(['session', 'BN', 'chordID'], as_index=False, observed=True)[cols].mean()

    data = df[cols].to_numpy()

    slot_of  = {chord: i for i, chord in enumerate(np.asarray(get_trained_and_untrained(sn), dtype=int))}

    sess_idx = df.session.map({s: i + 1 for i, s in enumerate(gl.sessions)})
    cond_vec = (sess_idx.astype(str) + ',' + df.chordID.astype(int).map(slot_of).astype(str)).to_numpy()
    part_vec = (sess_idx * 100 + df.BN).to_numpy()   # BN restarts at 1 every session

    return data, cond_vec, part_vec


def calc_G_rois(loader, sessions=('all', *gl.sessions), repetitions=('all', 1, 2)):
    """G of every ROI and session, with the subjects on the first dimension.

    Returns {(hemisphere, roi, session): (n_subj, n_cond, n_cond) array}, where
    n_cond is 8 within a session and 24 across sessions. Subjects come out in
    the order of ``loader.sns``.
    """
    glm = loader.glm
    
    for data in loader:
        for session in sessions:
            for repetition in repetitions:

                print(f'rois, doing participant {sn}...')

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

def calc_G_force(sns=gl.participants, metrics=('abs', 'der', 'der_peak'),
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



if __name__=='__main__':
    sns         = gl.participants
    glm         = 3
    sessions    = ('all', *gl.sessions)
    repetitions = ['all'] #, 1, 2]

    #loader      = BetasPrewithenedLoader(sns=sns, glm=glm)
    #calc_G_rois(loader, sessions, repetitions)

    calc_G_force(sns=sns, sessions=sessions, repetitions=repetitions)
