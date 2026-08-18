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

## Behavioural data flow

```mermaid
flowchart TD
    %% ---------- raw ----------
    RAW_BEHAV[("<b>.mov and .dat files from each participant and sessions:</b><br/>behavioural/day&lt;1..24&gt;/efc4_&lt;sn&gt;.dat<br/>behavioural/day&lt;1..24&gt;/efc4_&lt;sn&gt;_&lt;bl&gt;.mov")]:::input

    %% ---------- single trial ----------
    ST["scripts/behaviour_single_trial.py"]:::code
    STTSV[("<b>trial-wise behavioural metrics for each participant and session:</b><br/>behavioural/day&lt;d&gt;/efc4_&lt;sn&gt;_single_trial.tsv")]:::data

    RAW_BEHAV --> ST
    ST --> STTSV

    %% ---------- summary ----------
    SUM["scripts/behaviour_summary.py"]:::code
    STTSV --> SUM

    TRIAL[("<b>trial-wise behavioural metrics for all participants and sessions:</b><br/>behavioural/behaviour.trial.tsv")]:::data
    
    PERF[("<b>session-wise success rate, ET, and MD, split by chord type:</b><br/>behavioural/behaviour.session.success.tsv")]:::data
    PERF_REP[("<b>session-wise success rate, ET, and MD, split by chord type and repetition:</b><br/>behavioural/behaviour.session.success.repetition.tsv")]:::data
    
    FWIDE[("<b>trial-wise finger force (wide format):</b><br/>behavioural/force.trial.wide.tsv")]:::data
    FLONG[("<b>trial-wise finger force (long format):</b><br/>behavioural/force.trial.long.tsv")]:::data
    FSESS[("<b>session-wise finger force averaged across fingers, split by chord type:</b><br/>behavioural/force.session.avg.tsv")]:::data
    FSESS_REP[("<b>session-wise finger force averaged across fingers, split by chord type and repetition:</b><br/>behavioural/force.session.avg.repetition.tsv")]:::data
    FFMRI[("<b>block-wise finger force for scanning sessions (wide format):</b><br/>behavioural/force.fmri.wide.tsv")]:::data

    %% performance
    SUM --> TRIAL
    TRIAL --> PERF
    TRIAL --> PERF_REP

    %% force
    SUM --> FWIDE
    FWIDE --> FFMRI
    FWIDE --> FLONG
    FLONG --> FSESS
    FLONG --> FSESS_REP
    
  

    %% ---------- figures / downstream ----------
    NB_BEHAV["notebooks/behaviour.ipynb"]:::code

    PERF --> NB_BEHAV
    PERF_REP --> NB_BEHAV
    FSESS --> NB_BEHAV
    FSESS_REP --> NB_BEHAV

    classDef input fill:#e8eef7,stroke:#5b7fb5,color:#1a2c47;
    classDef code  fill:#eef3ea,stroke:#7a9a5f,color:#2a3a1e;
    classDef data  fill:#f7f1e8,stroke:#c39b56,color:#4a3818;
    classDef out   fill:#f3e8f0,stroke:#a86b97,color:#3f1f39;
```

Blue = raw task output, green = code, amber = saved tables. All paths are relative to
`gl.baseDir`; `<sn>` is the participant number, `<bl>` the block number and `<d>` the day.



