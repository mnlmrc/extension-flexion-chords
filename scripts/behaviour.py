import argparse

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.behaviour import (
    BLOCK_BY, FFMRI, FLONG, FORCE_COLS, FORCE_ID_COLS, FORCE_MEASURES, FSESS, FSESS_REP,
    FWIDE, N_SESSIONS, PERF, PERF_COLS, PERF_REP, SESSION_BY, TRIAL,
    concatenate_sessions, force_wide_to_long, load, save, single_trial_session,
    summarise_trials_by,
)


def single_trial_behaviour(sn=None, sessions=None):
    """STTSV: parse the raw .dat/.mov files into one table per participant x session.

    `sn` defaults to every participant, `sessions` to all `N_SESSIONS` days.
    """
    sns      = gl.participants if sn is None else [sn]
    sessions = range(1, N_SESSIONS + 1) if sessions is None else sessions

    for s in sns:
        for session in sessions:
            single_trial_session(s, session)


def behaviour_by_trial(n_sessions=N_SESSIONS):
    """TRIAL: all single trials of all participants."""
    df = concatenate_sessions(n_sessions)
    save(df, TRIAL)


def performance_by_session():
    """PERF / PERF_REP: trial-wise -> session-wise performance."""
    perf     = load(TRIAL)[SESSION_BY + ['Repetition'] + PERF_COLS]
    df_norep = summarise_trials_by(perf, SESSION_BY)
    df_rep   = summarise_trials_by(perf, SESSION_BY + ['Repetition'])
    save(df_norep, PERF)
    save(df_rep, PERF_REP)


def force_by_trial_wide():
    """FWIDE: trial-wise force, one column per finger."""
    trial = load(TRIAL)
    force_wide = trial[FORCE_ID_COLS + ['trialPoint'] + [c for m in FORCE_MEASURES for c in FORCE_COLS[m]]]
    save(force_wide, FWIDE)


def force_by_run_wide():
    """FFMRI: trial-wise -> run-wise force, scanning sessions only."""
    force_wide = load(FWIDE)
    force_fmri = force_wide[force_wide.session_type == 'scanning']
    force_fmri = force_fmri.groupby(BLOCK_BY, observed=True).mean(numeric_only=True).reset_index()
    save(force_fmri, FFMRI)


def force_by_trial_long():
    """FLONG: trial-wise force, one row per trial x finger."""
    force_wide = load(FWIDE)
    force_long = force_wide_to_long(force_wide)
    save(force_long, FLONG)


def force_by_session_avg():
    """FSESS / FSESS_REP: trial-wise -> session-wise force.

    Averaged over fingers as well as trials, since the fingers are stacked.
    """
    force_long = load(FLONG)

    force_long_succ = force_long[force_long.trialPoint == 1].drop(columns='trialPoint')
    sess_force      = force_long_succ.groupby(SESSION_BY, observed=True).mean(numeric_only=True).reset_index()
    save(sess_force, FSESS)

    # NB: unlike FSESS this averages over all trials, successful or not
    sess_rep_force = force_long.groupby(SESSION_BY + ['Repetition'], observed=True).mean(numeric_only=True).reset_index()
    save(sess_rep_force, FSESS_REP)


# Step name -> function, in the order the full run does them.
FUNC = {
    'single.trial'    : single_trial_behaviour,
    'trial'           : behaviour_by_trial,
    'session'         : performance_by_session,
    'force.trial.wide': force_by_trial_wide,
    'force.run.wide'  : force_by_run_wide,
    'force.trial.long': force_by_trial_long,
    'force.session'   : force_by_session_avg,
}


def main(what=None, sn=None, sessions=None):
    """Run one step. `sn` and `sessions` only apply to the single-trial step."""
    if what == 'single.trial':
        FUNC[what](sn=sn, sessions=sessions)
    elif what is not None:
        FUNC[what]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--sn', type=int, default=None, help='participant number, single.trial only (default: all participants)')
    parser.add_argument('--sessions', nargs='+', type=int, default=None, help='session numbers, single.trial only (default: all sessions)')
    args = parser.parse_args()

    main(args.what, sn=args.sn, sessions=args.sessions)

    if args.what==None:
        single_trial_behaviour(sn=args.sn, sessions=args.sessions)
        behaviour_by_trial()
        performance_by_session()
        force_by_trial_wide()
        force_by_run_wide()
        force_by_trial_long()
        force_by_session_avg()
