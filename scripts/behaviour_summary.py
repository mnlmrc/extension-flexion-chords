import argparse
import os

import pandas as pd

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.behaviour import FINGERS

# Descriptor columns.
ID_COLS = ['subNum', 'BN', 'TN', 'Repetition', 'chordID', 'chord', 'session', 'session_type', 'week']

# Performance measures kept in the behaviour.* tables.
PERF_COLS = ['trialPoint', 'ET', 'MD']

# Force columns
FORCE_MEASURES = ['abs', 'der', 'der_peak']
FORCE_COLS = {m: [f'{f}_{m}' for f in FINGERS] for m in FORCE_MEASURES}

# Descriptors kept in the force tables.
FORCE_ID_COLS = ['subNum', 'TN', 'BN', 'session', 'chord', 'chordID', 'Repetition', 'session_type', 'week']

# Groupings
SESSION_BY = ['subNum', 'session', 'chord', 'session_type', 'week']
BLOCK_BY = ['subNum', 'BN', 'session', 'chord', 'chordID', 'Repetition', 'session_type', 'week']

TRIAL     = 'behaviour.trial.tsv'
PERF      = 'behaviour.session.tsv'
PERF_REP  = 'behaviour.session.repetition.tsv'
FWIDE     = 'force.trial.wide.tsv'
FFMRI     = 'force.run.wide.tsv'
FLONG     = 'force.trial.long.tsv'
FSESS     = 'force.session.avg.tsv'
FSESS_REP = 'force.session.repetition.avg.tsv'


def save(df, fname):
    """Write one summary table to `<baseDir>/<behavDir>/<fname>`."""
    df.to_csv(os.path.join(gl.baseDir, gl.behavDir, fname), sep='\t', index=False)


def load(fname):
    """Read one summary table back, restoring the identifier columns as categories.

    The cast is what `concatenate_sessions` applies, so a step gets the same
    dtypes whether it is run on its own or right after the step that wrote its
    input.
    """
    df = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, fname), sep='\t')
    cat_cols = [c for c in ID_COLS if c in df.columns]
    return df.astype({c: 'category' for c in cat_cols})


def concatenate_sessions(n_sessions=24):
    """Concatenate single-session dataframes (STTSV -> TRIAL)."""
    path = os.path.join(gl.baseDir, gl.behavDir)

    # concatenate
    trials = []
    for sn in gl.participants:
        for session in range(1, n_sessions + 1):
            print(f'doing participant {sn}, session {session}')
            trials.append(pd.read_csv(os.path.join(path, f'day{session}', f'efc4_{sn}_single_trial.tsv'), sep='\t'))
    trial = pd.concat(trials)

    cat_cols = [c for c in ID_COLS if c in trial.columns]
    trial = trial.astype({c: 'category' for c in cat_cols})
    return trial[cat_cols + [c for c in trial.columns if c not in cat_cols]]


def summarise_trials(trial, by):
    """Average trials within each cell of `by`.

    Every measure is averaged over successful trials only, except `trialPoint`,
    which is averaged over all trials and is therefore the success rate.
    """
    success = trial[trial.trialPoint == 1].groupby(by, observed=True).mean(numeric_only=True).reset_index()
    rate = trial.groupby(by, observed=True)[['trialPoint']].mean().reset_index()
    return success.drop(columns='trialPoint').merge(rate, on=by)


def force_wide_to_long(force_wide):
    """Stack the per-finger force columns into one row per trial x finger (FWIDE -> FLONG).

    The measures are melted separately and pasted back together column-wise:
    each melt stacks the fingers in the same order, so the rows stay aligned.
    `finger` is named after the first measure melted (`thumb_abs`, ...).
    """
    force_long = None
    for measure in FORCE_MEASURES:
        value_name = f'force_{measure}'
        melted = force_wide.melt(id_vars=FORCE_ID_COLS + ['trialPoint'], value_vars=FORCE_COLS[measure],
                                 var_name='finger', value_name=value_name)
        if force_long is None:
            force_long = melted
        else:
            force_long[value_name] = melted[value_name].to_numpy()
    return force_long


def make_trial(n_sessions=24):
    """TRIAL: all single trials of all participants."""
    df = concatenate_sessions(n_sessions)
    save(df, TRIAL)


def make_session():
    """PERF / PERF_REP: trial-wise -> session-wise performance."""
    perf     = load(TRIAL)[SESSION_BY + ['Repetition'] + PERF_COLS]
    df_norep = summarise_trials(perf, SESSION_BY)
    df_rep   = summarise_trials(perf, SESSION_BY + ['Repetition'])
    save(df_norep, PERF)
    save(df_rep, PERF_REP)


def make_force_trial_wide():
    """FWIDE: trial-wise force, one column per finger."""
    trial = load(TRIAL)
    force_wide = trial[FORCE_ID_COLS + ['trialPoint'] + [c for m in FORCE_MEASURES for c in FORCE_COLS[m]]]
    save(force, FWIDE)


def make_force_run_wide():
    """FFMRI: trial-wise -> block-wise force, scanning sessions only."""
    force_wide = load(FWIDE)
    force_fmri = force_wide[force_wide.session_type == 'scanning']
    force_fmri = force_fmri.groupby(BLOCK_BY, observed=True).mean(numeric_only=True).reset_index()
    save(force_fmri, FFMRI)


def make_force_trial_long():
    """FLONG: trial-wise force, one row per trial x finger."""
    force_wide = load(FWIDE)
    force_long = force_wide_to_long(force_wide)
    save(force_long, FLONG)


def make_force_session():
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


# Step name -> function, in the order `all` runs them.
FUNC = {
    None              : None,
    'trial'           : make_trial,
    'session'         : make_session,
    'force.trial.wide': make_force_trial_wide,
    'force.run.wide'  : make_force_run_wide,
    'force.trial.long': make_force_trial_long,
    'force.session'   : make_force_session,
}


def main(what): 
    FUNC[what]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--what', nargs='+', default=None, choices=list(FUNC), help='which step(s) to run (default: all)')
    args = parser.parse_args()

    main(args.what)

    if args.what==None:
        make_trial()
        make_session()
        make_force_trial_long()
        make_force_run_wide()
        make_force_trial_long()
        make_force_session()
