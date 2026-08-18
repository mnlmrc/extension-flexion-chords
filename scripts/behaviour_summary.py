"""Aggregate the per-day single-trial behavioural files into the summary tables.

Only successful trials (`trialPoint == 1`) contribute to the
averages; `trialPoint` itself is averaged over all trials, so it is the success
rate.

Files generated
---------------
behaviour.trial.tsv (TRIAL)
    All single trials of all participants concatenated, with the identifier
    columns (`ID_COLS`) cast to category and moved to the front. Source table
    for everything below.
behaviour.session.success.tsv (PERF)
    Performance (`PERF_COLS`) averaged per `SESSION_BY` cell.
behaviour.session.success.repetition.tsv (PERF_REP)
    Same, additionally split by Repetition.
force.trial.wide.tsv (FWIDE)
    Successful single trials with one column per finger x force measure, e.g.
    `thumb_abs`. Source table for the two force branches below.
force.fmri.wide.tsv (FFMRI)
    force.trial.wide restricted to the scanning sessions and averaged per
    `BLOCK_BY` cell — i.e. one row per block, for use as GLM regressors.
force.trial.long.tsv (FLONG)
    force.trial.wide in long format: one row per trial x finger, with columns
    force_abs, force_der and force_der_peak.
force.session.avg.tsv (FSESS)
    force.trial.long averaged (across fingers and trials) per `SESSION_BY` cell.
force.session.avg.repetition.tsv (FSESS_REP)
    Same, additionally split by Repetition.
"""

import os

import pandas as pd

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.force import FINGERS

# Trial identifiers: the columns that label a trial rather than measure it.
ID_COLS = ['subNum', 'BN', 'TN', 'Repetition', 'chordID', 'chord', 'session', 'session_type', 'week']

# Performance measures kept in the behaviour.* tables.
PERF_COLS = ['trialPoint', 'ET', 'MD']

# Force measures, kept as one column per finger in wide format (`thumb_abs`, ...)
# and as one column per measure in long format (`force_abs`, ...).
FORCE_MEASURES = ['abs', 'der', 'der_peak']
FORCE_COLS = {m: [f'{f}_{m}' for f in FINGERS] for m in FORCE_MEASURES}

# Identifiers kept in the force tables (chordID as well as chord, no trialPoint).
FORCE_ID_COLS = ['subNum', 'TN', 'BN', 'session', 'chord', 'chordID', 'Repetition', 'session_type', 'week']

# Groupings: one row per session (the `.repetition` tables append Repetition),
# and one row per scanning block for the fMRI regressors.
SESSION_BY = ['subNum', 'session', 'chord', 'session_type', 'week']
BLOCK_BY = ['subNum', 'BN', 'session', 'chord', 'chordID', 'Repetition', 'session_type', 'week']


def load_single_trials(participants, ndays=24):
    """Concatenate the per-day single-trial tables of every participant (STTSV -> TRIAL)."""
    path = os.path.join(gl.baseDir, gl.behavDir)

    trials = []
    for sn in participants:
        for day in range(1, ndays + 1):
            print(f'doing participant {sn}, day {day}')
            trials.append(pd.read_csv(os.path.join(path, f'day{day}', f'efc4_{sn}_single_trial.tsv'), sep='\t'))
    trial = pd.concat(trials)

    # identifiers as category, so that groupby(observed=True) keeps only the cells
    # that actually occur, and up front, so that the table is readable
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
        melted = force_wide.melt(id_vars=FORCE_ID_COLS, value_vars=FORCE_COLS[measure],
                                 var_name='finger', value_name=value_name)
        if force_long is None:
            force_long = melted
        else:
            force_long[value_name] = melted[value_name].to_numpy()
    return force_long


def save(df, fname):
    """Write one summary table to `<baseDir>/<behavDir>/<fname>`."""
    df.to_csv(os.path.join(gl.baseDir, gl.behavDir, fname), sep='\t', index=False)


if __name__ == '__main__':

    # ---- TRIAL: all single trials of all participants -----------------------
    trial = load_single_trials(gl.participants)
    save(trial, 'behaviour.trial.tsv')

    # ---- PERF / PERF_REP: trial-wise -> session-wise performance ------------
    perf = trial[SESSION_BY + ['Repetition'] + PERF_COLS]
    save(summarise_trials(perf, SESSION_BY), 'behaviour.session.success.tsv')
    save(summarise_trials(perf, SESSION_BY + ['Repetition']), 'behaviour.session.success.repetition.tsv')

    # ---- FWIDE: trial-wise force, one column per finger ---------------------
    force_wide = trial[FORCE_ID_COLS + ['trialPoint'] + [c for m in FORCE_MEASURES for c in FORCE_COLS[m]]]
    force_wide = force_wide[force_wide.trialPoint == 1].drop(columns='trialPoint')
    save(force_wide, 'force.trial.wide.tsv')

    # ---- FFMRI: trial-wise -> block-wise force, scanning sessions only ------
    fmri_force = force_wide[force_wide.session_type == 'scanning']
    fmri_force = fmri_force.groupby(BLOCK_BY, observed=True).mean(numeric_only=True).reset_index()
    save(fmri_force, 'force.fmri.wide.tsv')

    # ---- FLONG: trial-wise force, one row per trial x finger ----------------
    force_long = force_wide_to_long(force_wide)
    save(force_long, 'force.trial.long.tsv')

    # ---- FSESS / FSESS_REP: trial-wise -> session-wise force ----------------
    # averaged over fingers as well as trials, since the fingers are stacked
    sess_force = force_long.groupby(SESSION_BY, observed=True).mean(numeric_only=True).reset_index()
    save(sess_force, 'force.session.avg.tsv')

    sess_rep_force = force_long.groupby(SESSION_BY + ['Repetition'], observed=True).mean(numeric_only=True).reset_index()
    save(sess_rep_force, 'force.session.avg.repetition.tsv')
