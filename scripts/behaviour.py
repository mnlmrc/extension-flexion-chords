import argparse
import os

import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.behaviour as behaviour

# Days of the experiment.
N_SESSIONS = 24

# Descriptor columns.
ID_COLS = ['subNum', 'BN', 'TN', 'Repetition', 'chordID', 'chord', 'session', 'session_type', 'week']

# Performance measures kept in the behaviour.* tables.
PERF_COLS = ['trialPoint', 'ET', 'MD']

# Force columns
FORCE_MEASURES = ['raw', 'abs', 'der',]
FORCE_COLS = {m: [f'{f}_{m}' for f in gl.fingers] for m in FORCE_MEASURES}

# Descriptors kept in the force tables.
FORCE_ID_COLS = ['subNum', 'TN', 'BN', 'session', 'chord', 'chordID', 'Repetition', 'session_type', 'week']

# Groupings
SESSION_BY = ['subNum', 'session', 'chord', 'session_type', 'week']
BLOCK_BY = ['subNum', 'BN', 'session', 'chord', 'chordID', 'Repetition', 'session_type', 'week']

TRIAL     = 'behaviour.trial.tsv'
BSESS     = 'behaviour.session.tsv'
BSESS_REP = 'behaviour.session.repetition.tsv'
FWIDE     = 'force.trial.wide.tsv'
FFMRI     = 'force.run.wide.tsv'
FLONG     = 'force.trial.long.tsv'
FSESS     = 'force.session.avg.tsv'
FSESS_REP = 'force.session.repetition.avg.tsv'


def behaviour_single_session(sns=gl.participants, sessions=None):
    """STTSV: parse the raw .dat/.mov files into one table per participant x session.

    `sn` defaults to every participant, `sessions` to all `N_SESSIONS` days.
    """
    # sns      = gl.participants if sn is None else [sn]
    sessions = range(1, N_SESSIONS + 1) if sessions is None else sessions

    for sn in sns:
        for session in sessions:
            behaviour.analyse_session(sn, session)


def behaviour_by_trial(n_sessions=N_SESSIONS):
    """TRIAL: all single trials of all participants."""
    df = behaviour.concatenate_sessions(n_sessions, ID_COLS)
    df.to_csv(os.path.join(gl.baseDir, gl.behavDir, TRIAL), sep='\t', index=False)


def performance_by_session():
    """PERF / PERF_REP: trial-wise -> session-wise performance."""
    trial    = behaviour.load(TRIAL, ID_COLS)[SESSION_BY + ['Repetition'] + PERF_COLS]
    df_norep = behaviour.group_trials_by(trial, SESSION_BY)
    df_rep   = behaviour.group_trials_by(trial, SESSION_BY + ['Repetition'])
    df_norep.to_csv(os.path.join(gl.baseDir, gl.behavDir, BSESS), sep='\t', index=False)
    df_rep.to_csv(os.path.join(gl.baseDir, gl.behavDir, BSESS_REP), sep='\t', index=False)


def force_by_trial_wide():
    """FWIDE: trial-wise force, one column per finger."""
    trial = behaviour.load(TRIAL, ID_COLS)
    force_wide = trial[FORCE_ID_COLS + ['trialPoint'] + [c for m in FORCE_MEASURES for c in FORCE_COLS[m]]]
    force_wide.to_csv(os.path.join(gl.baseDir, gl.behavDir, FWIDE), sep='\t', index=False)


def force_by_run_wide():
    """FFMRI: trial-wise -> run-wise force, scanning sessions only."""
    force_wide = behaviour.load(FWIDE, ID_COLS)
    force_fmri = force_wide[force_wide.session_type == 'scanning']
    force_fmri = force_fmri.groupby(BLOCK_BY, observed=True).mean(numeric_only=True).reset_index()
    force_fmri.to_csv(os.path.join(gl.baseDir, gl.behavDir, FFMRI), sep='\t', index=False)


def force_by_trial_long():
    """FLONG: trial-wise force, one row per trial x finger."""
    force_wide = behaviour.load(FWIDE, ID_COLS)
    force_long = behaviour.force_wide_to_long(force_wide, FORCE_ID_COLS, FORCE_COLS)
    force_long.to_csv(os.path.join(gl.baseDir, gl.behavDir, FLONG), sep='\t', index=False)


def force_by_session_avg():
    """FSESS / FSESS_REP: trial-wise -> session-wise force.

    Averaged over fingers as well as trials, since the fingers are stacked.
    """
    force_long = behaviour.load(FLONG, ID_COLS)

    force_long_succ = force_long[force_long.trialPoint == 1].drop(columns='trialPoint')
    sess_force      = force_long_succ.groupby(SESSION_BY, observed=True).mean(numeric_only=True).reset_index()
    sess_force.to_csv(os.path.join(gl.baseDir, gl.behavDir, FSESS), sep='\t', index=False)

    # Same success filter as FSESS: failed trials have much lower force, and the
    # success rate differs by repetition (day 1: .33 rep1 vs .39 rep2), so mixing
    # them in would make the repetition contrast partly a success-rate contrast.
    sess_rep_force = force_long_succ.groupby(SESSION_BY + ['Repetition'], observed=True).mean(numeric_only=True).reset_index()
    sess_rep_force.to_csv(os.path.join(gl.baseDir, gl.behavDir, FSESS_REP), sep='\t', index=False)


# Step name -> function, in the order the full run does them.
FUNC = {
    'single_session'  : behaviour_single_session,
    'trial'           : behaviour_by_trial,
    'session'         : performance_by_session,
    'force.trial.wide': force_by_trial_wide,
    'force.run.wide'  : force_by_run_wide,
    'force.trial.long': force_by_trial_long,
    'force.session'   : force_by_session_avg,
}


def main(what=None, **kwargs):
    """Run one step. Extra kwargs (`sn`, `sessions`) only apply to the single-trial step."""
    if what is not None:
        FUNC[what](**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--sn', type=int, default=None, help='participant number, single.trial only (default: all participants)')
    parser.add_argument('--sessions', nargs='+', type=int, default=None, help='session numbers, single.trial only (default: all sessions)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        # main('single_session')
        main('trial')
        main('session')
        main('force.trial.wide')
        main('force.run.wide')
        main('force.trial.long')
        main('force.session')
