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

## Dataflows
### Behaviour

```mermaid
flowchart TD
    %% ---------- single trial ----------
    RAW_BEHAV[("<b>.dat and .mov files:</b><br/>behavioural/day&lt;d&gt;/efc4_&lt;sn&gt;.dat<br/>behavioural/day&lt;d&gt;/efc4_&lt;sn&gt;_&lt;bn&gt;.dat")]:::data
    ST["scripts/behaviour.behaviour_single_session()"]:::code
    STTSV[("<b>trial-wise behavioural metrics for each participant and session:</b><br/>behavioural/day&lt;d&gt;/efc4_&lt;sn&gt;_single_trial.tsv")]:::data

    RAW_BEHAV --> ST
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
    FFMRI[("<b>run-wise finger force for scanning sessions (wide format):</b><br/>behavioural/force.run.wide.tsv<sup>1</sup>")]:::data

    %% ---------- force pattern analysis ----------
    %%P_GFORCE["scripts/pattern.calc_G_force()"]:::code
    %%P_DFFORCE["scripts/pattern.make_G_dataframe_force()"]:::code

    %%GFORCE[("<b>8x8 second-moment matrices of 5-finger absolute force and absolute force derivative for each participant, session[, repetition]:</b><br/>pcm/subj&lt;sn&gt;/G_obs_raw.within_session.&lt;session&gt;[.&lt;repetition&gt;].force.&lt;metric&gt;.npy")]:::data
    %%DFFORCE[("<b>session-wise crossnobis and angular distance between chord pairs:</b><br/>pcm/dissimilarity.within_session.force.tsv")]:::data

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
    %%FFMRI --> P_GFORCE
    %%P_GFORCE --> GFORCE
    %%GFORCE --> P_DFFORCE
    %%P_DFFORCE --> DFFORCE

    classDef code  fill:#eef3ea,stroke:#7a9a5f,color:#2a3a1e;
    classDef data  fill:#f7f1e8,stroke:#c39b56,color:#4a3818;
```
<sup>1</sup>Used for pattern analysis (see below)

<!-- Green = code, amber = saved tables; each green node is the function that writes the tables it
points to. Every step reloads its input from disk, so the arrows are also the order the steps
have to be run in. All paths are relative to `gl.baseDir`; `<sn>` is the participant number,
`<bl>` the block number and `<d>` the day. `<...>` is a placeholder to fill in; `[...]` marks a
part of the name that is only present sometimes (e.g. `[.<repetition>]` is dropped when the G is
not split by repetition). -->

### Univariate activation

<!-- Starting from the per-participant `contrast.dscalar.nii` (written by
`scripts/cifti.CiftiCortex.contrast()`), `scripts/activation.py` produces the ROI-averaged
activation table and the group surface maps. -->

```mermaid
flowchart TB
    subgraph ROI ["ROI-based"]
        direction TB
        CON[("<b>3D CIFTI contrast maps:</b><br/>glm&lt;glm&gt;/subj&lt;sn&gt;/contrast.dscalar.nii")]:::data
        A_ROI["scripts/activation.roi_activation()"]:::code
        ROITSV[("<b>Univariate activation in each ROI:</b><br/>glm&lt;glm&gt;/&lt;atlas&gt;.activation.tsv")]:::data

        CON --> A_ROI
        A_ROI --> ROITSV
    end

    subgraph SURF ["Surface-based"]
        direction TB
        GIFTI[("<b>surface-projected contrast maps:</b><br/>surfaceWB/subj&lt;sn&gt;/glm&lt;glm&gt;.con.&lt;H&gt;.func.gii")]:::data

        A_SMOOTH["scripts/activation.smooth_contrasts()"]:::code

        SMOOTHD[("<b>smoothened contrast maps:</b><br/>surfaceWB/subj&lt;sn&gt;/glm&lt;glm&gt;.con.session.smooth.dscalar.nii")]:::data

        A_AVG["scripts/activation.average_contrasts()"]:::code
        GRPD[("<b>group-averaged smoothened contrast map:</b><br/>surfaceWB/glm&lt;glm&gt;.con.session.smooth.dscalar.nii")]:::data
        A_DIFF["scripts/activation.average_contrasts_difference()"]:::code
        DIFFD[("<b>group-averaged smoothened contrast map (trained – untrained):</b><br/>surfaceWB/glm&lt;glm&gt;.con.trained_vs_untrained.smooth.dscalar.nii")]:::data

        GIFTI --> A_SMOOTH
        A_SMOOTH --> SMOOTHD
        SMOOTHD --> A_AVG
        A_AVG --> GRPD
        GRPD --> A_DIFF
        A_DIFF --> DIFFD
    end

    classDef code  fill:#eef3ea,stroke:#7a9a5f,color:#2a3a1e;
    classDef data  fill:#f7f1e8,stroke:#c39b56,color:#4a3818;
    classDef group fill:#3d6d99,stroke:#22456b,stroke-width:1.5px,color:#ffffff, font-size:23px;
    class ROI,SURF group;
```

<!-- The ROI step (`roi`) reads `contrast.dscalar.nii` directly. The surface steps (`smooth`,
`average.smooth.surface`) read the per-hemisphere `glm<glm>.con.<H>.func.gii` giftis; those are
the same contrast maps projected onto the surface by `surface.project_cifti_to_surface()`, run
*upstream* of `activation.py` (dashed arrow) — note it keys off the `con` filename stem while the
ROI branch reads `contrast.dscalar.nii`. -->

### Pattern

<!-- `scripts/pattern.py` estimates the second-moment matrices (Gs) of the neural and force
patterns, then summarises their geometry. The neural branch reads the glm betas/residuals (written
by `scripts/cifti.py`) prewhitened within each ROI; the force branch reads `force.run.wide.tsv`
(written by `scripts/behaviour.py`). The dataframe/noise-ceiling/correlation steps reload the
`G_obs_raw` Gs (or the betas, for `correlation`) from disk, so the arrows are also the run order. -->

```mermaid
flowchart TB
    subgraph NEURAL ["Neural (ROI-based)"]
        direction TB
        NBETA[("<b>run-wise coefficients from 1st-level GLM, residuals timeseries and ROI masks:</b><br/>glm&lt;glm&gt;/subj&lt;sn&gt;/beta.dscalar.nii<br/>glm&lt;glm&gt;/subj&lt;sn&gt;/residual.dtseries.nii<br/>ROI/subj&lt;sn&gt;/&lt;atlas&gt;.&lt;H&gt;.&lt;roi&gt;.nii")]:::data

        P_GROIS["scripts/pattern.calc_G_rois()<sup>1</sup>"]:::code
        GROIS[("<b>Second-moment matrix:</b><br/>pcm/subj&lt;sn&gt;/G_obs_raw.&lt;session&gt;.glm&lt;glm&gt;.&lt;H&gt;.&lt;roi&gt;.npy<sup>2</sup>")]:::data

        P_DFROIS["scripts/pattern.make_dataframe_rois()"]:::code
        DFROIS[("<b>pair-wise geometry (crossnobis, cosine, angle):</b><br/>pcm/dissimilarity.within_session.&lt;atlas&gt;.glm&lt;glm&gt;.tsv")]:::data

        P_NC["scripts/pattern.make_noise_ceiling_dataframe()"]:::code
        NC[("<b>RSA noise ceiling (lower/upper) per Hem, roi, session:</b><br/>pcm/noise_ceiling.within_session.&lt;atlas&gt;.glm&lt;glm&gt;.tsv")]:::data

        P_CORR["scripts/pattern.correlation_between_sessions()"]:::code
        CORRMLE[("<b>MLE correlation estimates (individual and group fit) between neural activity patterns for trained and untrained chords:</b><br/>pcm/MLE_correlation.&lt;atlas&gt;.glm&lt;glm&gt;.tsv")]:::data
    %% CORRXVAL[("<b>cross-validated across-session cosine:</b><br/>pcm/xval_correlation.&lt;atlas&gt;.glm&lt;glm&gt;.tsv")]:::data
    %% CORRCOV[("<b>across-session cov per session-pair and chord set:</b><br/>pcm/cov.corr_across_sess.glm&lt;glm&gt;.&lt;spair&gt;.&lt;chord&gt;.&lt;H&gt;.&lt;roi&gt;.npy")]:::data

        NBETA --> P_GROIS
        P_GROIS --> GROIS
        GROIS --> P_DFROIS
        P_DFROIS --> DFROIS
        GROIS --> P_NC
        P_NC --> NC
        NBETA --> P_CORR
        P_CORR --> CORRMLE
    %% P_CORR --> CORRXVAL
    %% P_CORR --> CORRCOV
    end

    subgraph FORCE ["Force"]
        direction TB
        FRUN[("<b>run-wise finger force:</b><br/>behavioural/force.run.wide.tsv")]:::data

        P_GFORCE["scripts/pattern.calc_G_force()"]:::code
        GFORCE[("<b>Second-moment matrices:</b><br/>pcm/subj&lt;sn&gt;/G_obs_raw.&lt;epoch&gt;.force.&lt;metric&gt;.npy<sup>2</sup>")]:::data

        P_DFFORCE["scripts/pattern.make_dataframe_force()"]:::code
        DFFORCE[("<b>pair-wise geometry (crossnobis, cosine, angle):</b><br/>pcm/dissimilarity.within_session.force.tsv")]:::data

        FRUN --> P_GFORCE
        P_GFORCE --> GFORCE
        GFORCE --> P_DFFORCE
        P_DFFORCE --> DFFORCE
    end

    classDef code  fill:#eef3ea,stroke:#7a9a5f,color:#2a3a1e;
    classDef data  fill:#f7f1e8,stroke:#c39b56,color:#4a3818;
    classDef group fill:#3d6d99,stroke:#22456b,stroke-width:1.5px,color:#ffffff, font-size:23px;
    class NEURAL,FORCE group;
```

<sup>1</sup>`calc_G_rois` performs multivariate prewhitening before calculating the G matrix.

<sup>2</sup>`calc_G_rois` and `calc_G_force` save `G_obs_raw.*.npy` (cross-validated G matrix, run mean across conditions not removed) and also `G_obs.*.npy` (cross-validated G matrix, run mean across conditions removed), `cov.*.npy` (cross-validated covariance matrix, run mean across conditions removed, voxel-centred), `G_obs_noxal.*.npy` (non-cross-validated G matrix):

<!-- `dataframe_rois` and `noise_ceiling` read only the `G_obs_raw.within_session.*` Gs; `correlation`
reads the betas/residuals directly (not the saved Gs). `<epoch>` is `within_session.<sess>` or
`across_session`, with an optional `[.<repetition>]`; `<spair>` is a session pair like `3-9`. -->


