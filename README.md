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
    RAW_BEHAV[("<b>.dat and .mov files:</b><br/>behavioural/day#lt;d#gt;/efc4_#lt;sn#gt;.dat<br/>behavioural/day#lt;d#gt;/efc4_#lt;sn#gt;_#lt;bn#gt;.dat")]:::data
    ST["scripts/behaviour.behaviour_single_session()"]:::code
    STTSV[("<b>trial-wise behavioural metrics for each participant and session:</b><br/>behavioural/day#lt;d#gt;/efc4_#lt;sn#gt;_single_trial.tsv")]:::data

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
    FFMRI[("<b>run-wise finger force for scanning sessions (wide format):</b><br/>behavioural/force.run.wide.tsv¹")]:::data

    %% ---------- force pattern analysis ----------
    %%P_GFORCE["scripts/pattern.calc_G_force()"]:::code
    %%P_DFFORCE["scripts/pattern.make_G_dataframe_force()"]:::code

    %%GFORCE[("<b>8x8 second-moment matrices of 5-finger absolute force and absolute force derivative for each participant, session[, repetition]:</b><br/>pcm/subj#lt;sn#gt;/G_obs_raw.within_session.#lt;session#gt;[.#lt;repetition#gt;].force.#lt;metric#gt;.npy")]:::data
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
¹Used for pattern analysis (see below)

<!-- Green = code, amber = saved tables; each green node is the function that writes the tables it
points to. Every step reloads its input from disk, so the arrows are also the order the steps
have to be run in. All paths are relative to `gl.baseDir`; `<sn>` is the participant number,
`<bl>` the block number and `<d>` the day. `<...>` is a placeholder to fill in; `[...]` marks a
part of the name that is only present sometimes (e.g. `[.<repetition>]` is dropped when the G is
not split by repetition). -->

#### Columns

**`behavioural/day<d>/efc4_<sn>_single_trial.tsv`** — one row per trial.

| Column | Description |
|---|---|
| `subNum` | participant number |
| `BN` | block (run) number |
| `Repetition` | 1 = first presentation of the chord, 2 = second presentation of the chord |
| `TN` | trial number |
| `trialPoint` | 1 = successful trial (cued fingers above `gl.ftarget` for 600 ms, no baseline exit during plan time), 0 = failed |
| `RT` | reaction time (s): first sample at which any finger leaves the ±`gl.fthresh` baseline area; `NaN` on failed trials |
| `ET` | execution time (s): from `RT` to the first 600-ms time interval in which the cued fingers hold the instructed chord; `NaN` on failed trials |
| `MD` | mean deviation of the 5-finger force trajectory from the straight line joining its the force pattern at `RT` and the force pattern at `ET`; `NaN` on failed trials |
| `chordID` | code for instructed code (thumb→pinkie): 1 = extension, 2 = flexion, 9 = neutral |
| `chord` | `trained` / `untrained` |
| `session` | day relative to the start of the experiment for that participant, 1–24 |
| `session_type` | `pretraining`->participant practiced all chords in the mock scanner (1,2), `scanning`->participant practiced all chords in the MRI scanner (3, 9, 23),  `testing`->participant practiced all chords sitting at the desk (4, 10, 24), `training`->participant practiced trained chords only sitting at the desk (all the rest),  |
| `week` | week relative to the start of the experiment for that participant, 1–5 |
| `thumb` … `pinkie` | signed mean force of each finger from `RT` to the end of the trial |
| `<finger>_abs` | mean **absolute** force of that finger from `RT` to the end of the trial |
| `<finger>_der` | mean **absolute derivative** of that finger's force from `RT` to the end of the trial |

**`behavioural/behaviour.trial.tsv`** — the single-trial tables of every participant and all 24 days
concatated. Same columns as `behavioural/day<d>/efc4_<sn>_single_trial.tsv`.

**`behavioural/behaviour.session[.repetition].tsv`** — one row per participant × session × chord type
(× repetition in the `.repetition` variant).

| Column | Description |
|---|---|
| `subNum`, `session`, `session_type`, `week`, `chord`, `Repetition` | as above |
| `ET` | execution time averaged over the **successful** trials |
| `MD` | mean deviation averaged over the **successful** trials |
| `trialPoint` | success rate: averaged over **all** trials of the cell |

**`behavioural/force.trial.wide.tsv`** — one row per trial.

| Column | Description |
|---|---|
| `subNum`, `TN`, `BN`, `session`, `chord`, `chordID`, `Repetition`, `session_type`, `week` |  as above |
| `trialPoint` | 1 = successful trial (cued fingers above `gl.ftarget` for 600 ms, no baseline exit during plan time), 0 = failed |
| `<finger>_abs` (×5) | mean absolute force of that finger |
| `<finger>_der` (×5) | mean absolute force derivative of that finger |

**`behavioural/force.run.wide.tsv`** — scanning sessions only, one row per participant × run × session × chordID
(× repetition in the `.repetition` variant). **Failed trials are included**, as they are not excluded when fitting the GLM.

| Column | Description |
|---|---|
| `subNum`, `BN`, `session`, `chord`, `chordID`, `Repetition`, `week`, `session_type` | as above |
| `trialPoint` | fraction of successful trials |
| `<finger>_abs` | mean absolute force |
 `<finger>_der` | mean absolute force derivative |

**`behavioural/force.trial.long.tsv`** — `force.trial.wide.tsv` with the per-finger columns stacked: one row per trial × finger (i.e., five rows/trial).

| Column | Description |
|---|---|
| `subNum`, `TN`, `BN`, `session`, `chord`, `chordID`, `Repetition`, `session_type`, `week`, `trialPoint` | as in `force.trial.wide.tsv` |
| `finger` | which finger the row belongs to: `thumb`, `index`, `middle`, `ring`, `pinkie` |
| `force_abs` | mean absolute force |
| `force_der` | mean absolute force derivative |

**`behavioural/force.session[.repetition].avg.tsv`** — one row per participant × session × chord type
(× repetition), averaged over **successful trials only** and over the five fingers.

| Column | Description |
|---|---|
| `subNum`, `session`, `chord`, `session_type`, `week`,`Repetition` | as above |
| `force_abs` | absolute force averaged over fingers and successful trials |
| `force_der` | absolute force derivative averaged over fingers and successful trials |

### Univariate activation

<!-- Starting from the per-participant `contrast.dscalar.nii` (written by
`scripts/cifti.CiftiCortex.contrast()`), `scripts/activation.py` produces the ROI-averaged
activation table and the group surface maps. -->

```mermaid
flowchart TB
    subgraph ROI ["ROI-based"]
        direction TB
        CON[("<b>3D CIFTI contrast maps:</b><br/>glm#lt;glm#gt;/subj#lt;sn#gt;/contrast.dscalar.nii")]:::data
        A_ROI["scripts/activation.roi_activation()"]:::code
        ROITSV[("<b>Univariate activation in each ROI:</b><br/>glm#lt;glm#gt;/#lt;atlas#gt;.activation.tsv")]:::data

        CON --> A_ROI
        A_ROI --> ROITSV
    end

    subgraph SURF ["Surface-based"]
        direction TB
        GIFTI[("<b>surface-projected contrast maps:</b><br/>surfaceWB/subj#lt;sn#gt;/glm#lt;glm#gt;.con.#lt;H#gt;.func.gii")]:::data

        A_SMOOTH["scripts/activation.smooth_contrasts()"]:::code

        SMOOTHD[("<b>smoothened contrast maps:</b><br/>surfaceWB/subj#lt;sn#gt;/glm#lt;glm#gt;.con.session.smooth.dscalar.nii")]:::data

        A_AVG["scripts/activation.average_contrasts()"]:::code
        GRPD[("<b>group-averaged smoothened contrast map:</b><br/>surfaceWB/glm#lt;glm#gt;.con.session.smooth.dscalar.nii")]:::data
        A_DIFF["scripts/activation.average_contrasts_difference()"]:::code
        DIFFD[("<b>group-averaged smoothened contrast map (trained – untrained):</b><br/>surfaceWB/glm#lt;glm#gt;.con.trained_vs_untrained.smooth.dscalar.nii")]:::data

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

#### Columns

**`glm<glm>/<atlas>.activation.tsv`** — one row per participant × hemisphere × ROI × chordID × session.

| Column | Description |
|---|---|
| `region` | integer label value in the ROI mask |
| `regionname` | region from `gl.rois[<atlas>]` |
| `volume` | volume of the ROI in mm³ |
| `chordID` | chord of that frame (5-digit code, see the behaviour tables) |
| `session` | 3, 9 or 23 (mapped from `sess03` / `sess09` / `sess23`) |
| `sn` | participant number |
| `Hem` | `L` / `R` |
| `chord` | `trained` / `untrained` |
| `nanmean` | **the measure**: the contrast vs. baseline value (in a.u.) averaged over the ROI's voxels, ignoring NaNs |

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
        NBETA[("<b>run-wise coefficients from 1st-level GLM, residuals timeseries and ROI masks:</b><br/>glm#lt;glm#gt;/subj#lt;sn#gt;/beta.dscalar.nii<br/>glm#lt;glm#gt;/subj#lt;sn#gt;/residual.dtseries.nii<br/>ROI/subj#lt;sn#gt;/#lt;atlas#gt;.#lt;H#gt;.#lt;roi#gt;.nii")]:::data

        P_GROIS["scripts/pattern.calc_G_rois()¹"]:::code
        GROIS[("<b>Second-moment matrix:</b><br/>pcm/subj#lt;sn#gt;/G_obs_raw.#lt;session#gt;.glm#lt;glm#gt;.#lt;H#gt;.#lt;roi#gt;.npy²")]:::data

        P_DFROIS["scripts/pattern.make_dataframe_rois()"]:::code
        DFROIS[("<b>pair-wise geometry (crossnobis, cosine, angle):</b><br/>pcm/dissimilarity.within_session.#lt;atlas#gt;.glm#lt;glm#gt;.tsv")]:::data

        P_NC["scripts/pattern.make_noise_ceiling_dataframe()"]:::code
        NC[("<b>RSA noise ceiling (lower/upper) per Hem, roi, session:</b><br/>pcm/noise_ceiling.within_session.#lt;atlas#gt;.glm#lt;glm#gt;.tsv")]:::data

        P_CORR["scripts/pattern.correlation_between_sessions()"]:::code
        CORRMLE[("<b>MLE correlation estimates (individual and group fit) between neural activity patterns for trained and untrained chords:</b><br/>pcm/MLE_correlation.#lt;atlas#gt;.glm#lt;glm#gt;.tsv")]:::data
    %% CORRXVAL[("<b>cross-validated across-session cosine:</b><br/>pcm/xval_correlation.#lt;atlas#gt;.glm#lt;glm#gt;.tsv")]:::data
    %% CORRCOV[("<b>across-session cov per session-pair and chord set:</b><br/>pcm/cov.corr_across_sess.glm#lt;glm#gt;.#lt;spair#gt;.#lt;chord#gt;.#lt;H#gt;.#lt;roi#gt;.npy")]:::data

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
        GFORCE[("<b>Second-moment matrices:</b><br/>pcm/subj#lt;sn#gt;/G_obs_raw.#lt;epoch#gt;.force.#lt;metric#gt;.npy²")]:::data

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

¹`calc_G_rois` performs multivariate prewhitening before calculating the G matrix.

²`calc_G_rois` and `calc_G_force` save `G_obs_raw.*.npy` (cross-validated G matrix, run mean across conditions not removed) and also `G_obs.*.npy` (cross-validated G matrix, run mean across conditions removed), `cov.*.npy` (cross-validated covariance matrix, run mean across conditions removed, voxel-centred), `G_obs_noxal.*.npy` (non-cross-validated G matrix):

<!-- `dataframe_rois` and `noise_ceiling` read only the `G_obs_raw.within_session.*` Gs; `correlation`
reads the betas/residuals directly (not the saved Gs). `<epoch>` is `within_session.<sess>` or
`across_session`, with an optional `[.<repetition>]`; `<spair>` is a session pair like `3-9`. -->

#### Columns

**`pcm/dissimilarity.within_session.<atlas>.glm<glm>.tsv`** — one row per participant × hemisphere ×
ROI × session × chord pair (28 pairs, i.e., the lower triangle of a 8×8 matrix).

| Column | Description |
|---|---|
| `sn` | participant number |
| `Hem` | `L` / `R` |
| `roi` | region from `gl.rois[<atlas>]` |
| `session` | 3, 9 or 23 |
| `chord` | which pair group the pair belongs to: `trained` (both chords trained), `untrained` (both untrained) or `trained_untrained` (one of each) |
| `pair` | the two chord IDs joined by `-` in sorted order, so the same pair carries the same id for every participant |
| `crossnobis` | cross-validated Mahalanobis distance between the two chords' patterns (`pcm.G_to_dist` of `G_obs_raw`) |
| `cosine` | cosine of the angle between the two patterns (`pcm.G_to_cosine`) |
| `theta` | `arccos(cosine)`, in radians |
| `crossnobis_group`, `cosine_group` | the same pair's reference value: the across-participant mean in the reference session (`ref_session`, default 3) |
| `theta_group` | `arccos(cosine_group)` |

**`pcm/dissimilarity.within_session.force.tsv`** — one row per participant × force measure × session × chord pair (28 pairs, i.e., the lower triangle of a 8×8 matrix).

| Column | Description |
|---|---|
| `metric` | force measure the G was built from: `abs` (mean absolute force) or `der` (mean absolute force derivative) |
| `sn`, `session`, `chord`, `pair`, `crossnobis`, `cosine`, `theta`, `*_group` | as in `pcm/dissimilarity.within_session.<atlas>.glm<glm>.tsv` |

**`pcm/noise_ceiling.within_session.<atlas>.glm<glm>.tsv`** — one row per participant × hemisphere ×
ROI × session.

| Column | Description |
|---|---|
| `sn`, `Hem`, `roi`, `session` | as above |
| `lower` | correlation between this participant's crossnobis RDM and the mean RDM of the *other* participants (leave-one-out) |
| `upper` | the same correlation, but against the mean RDM of *all* participants, this one included |

**`pcm/MLE_correlation.<atlas>.glm<glm>.tsv`** — one row per participant × hemisphere × ROI ×
session pair × chord type.

| Column | Description |
|---|---|
| `sn` | participant number |
| `r_group` | correlation from the PCM group fit — a single r shared by all participants, so the value repeats for every `sn` of a cell |
| `r_indiv` | correlation from that participant's own individual fit |
| `SNR` | signal-to-noise of the individual fit, `sqrt(sigma2_1 * sigma2_2) / sigma2_e` (the two sessions' signal variances over the noise variance) |
| `chord` | `trained` / `untrained` |
| `corr` | session pair: `3-9`, `3-23` or `9-23` |
| `roi` | region, one of `gl.rois[<atlas>]` |
| `Hem` | `L` / `R` |


