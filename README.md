# EFC_learningfMRI

Analysis code for the extension–flexion chord (EFC) learning fMRI study: participants
practise a set of finger chords over 24 days, with fMRI scans on days 3, 9 and 23.

## Experimental design

* 8 chords (`gl.chordID`), 4 assigned as **trained** and 4 as **untrained** per
  participant (`participants.tsv`, columns `trained` / `untrained`).
* 24 days (`behavioural/day1` … `behavioural/day24`); each day is associated to a `session_type`:
  `pretraining` (days 1-2), `scanning` (days 3, 9, 23), `testing` (days 4, 10, 24) or `training` (all other sessions).
* Each trial the participant produces one 4-finger chord; force is sampled at
  500 Hz (`gl.fsample['force']`) and each chord is repeated twice in a row
  (`Repetition` 1 vs 2).

## Data flow

```mermaid
flowchart TD
    %% ---------- single trial ----------
    ST["scripts/behaviour.behaviour_single_session()"]:::code
    STTSV[("<b>trial-wise behavioural metrics for each participant and session:</b><br/>behavioural/day&lt;d&gt;/efc4_&lt;sn&gt;_single_trial.tsv")]:::data

    ST --> STTSV

    %% ---------- summary ----------
    F_TRIAL["scripts/behaviour.behaviour_by_trial()"]:::code
    F_SESS["scripts/behaviour.performance_by_session()"]:::code
    F_FWIDE["scripts/behaviour.force_by_trial_wide()"]:::code
    F_FRUN["scripts/behaviour.force_by_run_wide()"]:::code
    F_FLONG["scripts/behaviour.force_by_trial_long()"]:::code
    F_FSESS["scripts/behaviour.force_by_session_avg()"]:::code

    TRIAL[("<b>trial-wise behavioural metrics for all participants and sessions:</b><br/>behavioural/behaviour.trial.tsv")]:::data
    
    PERF_REP[("<b>session-wise success rate, ET, and MD, split by chord type[, repetition]:</b><br/>behavioural/behaviour.session[.repetition].tsv")]:::data
    
    FWIDE[("<b>trial-wise finger force (wide format):</b><br/>behavioural/force.trial.wide.tsv")]:::data
    FLONG[("<b>trial-wise finger force (long format):</b><br/>behavioural/force.trial.long.tsv")]:::data
    FSESS_REP[("<b>session-wise finger force averaged across fingers, split by chord type[, repetition]:</b><br/>behavioural/force.session[.repetition].avg.tsv")]:::data
    FFMRI[("<b>run-wise finger force for scanning sessions (wide format):</b><br/>behavioural/force.run.wide.tsv")]:::data

    %% ---------- force pattern analysis ----------
    P_GFORCE["scripts/pattern.calc_G_force()"]:::code
    P_DFFORCE["scripts/pattern.make_G_dataframe_force()"]:::code

    GFORCE[("<b>8x8 second-moment matrices of 5-finger absolute force and absolute force derivative for each participant, session[, repetition]:</b><br/>pcm/subj&lt;sn&gt;/G_obs_raw.within_session.&lt;session&gt;[.&lt;repetition&gt;].force.&lt;metric&gt;.npy")]:::data
    DFFORCE[("<b>pair-wise force geometry (crossnobis + cosine/angle vs. reference session):</b><br/>pcm/force.geometry.tsv")]:::data

    %% performance
    STTSV --> F_TRIAL
    F_TRIAL --> TRIAL
    TRIAL --> F_SESS
    F_SESS --> PERF_REP

    %% force
    TRIAL --> F_FWIDE
    F_FWIDE --> FWIDE
    FWIDE --> F_FRUN
    F_FRUN --> FFMRI
    FWIDE --> F_FLONG
    F_FLONG --> FLONG
    FLONG --> F_FSESS
    F_FSESS --> FSESS_REP

    %% force pattern analysis
    FFMRI --> P_GFORCE
    P_GFORCE --> GFORCE
    GFORCE --> P_DFFORCE
    P_DFFORCE --> DFFORCE

    classDef code  fill:#eef3ea,stroke:#7a9a5f,color:#2a3a1e;
    classDef data  fill:#f7f1e8,stroke:#c39b56,color:#4a3818;
```

Green = code, amber = saved tables; each green node is the function that writes the tables it
points to. Every step reloads its input from disk, so the arrows are also the order the steps
have to be run in. All paths are relative to `gl.baseDir`; `<sn>` is the participant number,
`<bl>` the block number and `<d>` the day. `<...>` is a placeholder to fill in; `[...]` marks a
part of the name that is only present sometimes (e.g. `[.<repetition>]` is dropped when the G is
not split by repetition).



