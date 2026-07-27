# EFC_learningfMRI

Analysis code for the extension–flexion chord (EFC) learning fMRI study: from 1st-level
GLM betas to ROI / searchlight multivariate geometry, PCM component models, and statistics.

## Pipeline overview

```mermaid
flowchart TD
    %% ---------- inputs ----------
    GLM["1st-level GLM &middot; glm*/subj*/<br/>beta.dscalar.nii &middot; ResMS.nii<br/>residual.dtseries.nii &middot; reginfo.tsv"]:::input
    SURF["surfaceWB/ &middot; anatomicals/<br/>white / pial surfaces &middot; Hem masks"]:::input
    BEHAV["behavioural/<br/>force &middot; trial data"]:::input

    %% ---------- ROI & searchlight definition ----------
    ROIS["rois.py<br/>make_cortical_rois &middot; make_hemispheres"]:::code
    ROIMASK[("ROI/subj*/<br/>ROI.*.nii &middot; Hem.*.nii")]:::data
    MKSL["searchlight.py<br/>make_searchlight"]:::code
    SLH5[("ROI/subj*/<br/>searchlight.L/R.h5")]:::data

    SURF --> ROIS --> ROIMASK
    SURF --> MKSL
    ROIMASK --> MKSL --> SLH5

    %% ---------- betas / prewhitening ----------
    REG["betas.py + util.py<br/>RegInfo (cond_vec / part_vec)<br/>load_betas &middot; load_residuals<br/>calc_prewhitened_betas &middot; BetasPrewithenedLoader"]:::code
    GLM --> REG

    %% ---------- ROI multivariate: geometry ----------
    GMAT["scripts/G_matrix.py<br/>calc_G_rois"]:::code
    GOBS[("pcm/<br/>G_obs.*.npy &middot; cov.*.npy")]:::data
    DISS["scripts/geometry_dataframe.py<br/>make_G_dataframe"]:::code
    DISSTSV[("pcm/<br/>dissimilarity.*.tsv")]:::data

    REG --> GMAT
    ROIMASK --> GMAT --> GOBS --> DISS --> DISSTSV

    %% ---------- ROI multivariate: PCM ----------
    PCM["scripts/component_model_fit.py<br/>fit_component_model"]:::code
    THETA[("pcm/subj*/<br/>theta_in.*.p")]:::data
    CWDF["scripts/component_model_dataframe.py<br/>make_component_model_dataframe"]:::code
    CWTSV[("pcm/<br/>component_model.*.tsv")]:::data

    REG --> PCM
    GOBS --> PCM --> THETA --> CWDF --> CWTSV

    %% ---------- surface searchlight ----------
    SLRUN["scripts/searchlight_crossnobis.py<br/>Searchlight.run &middot; calc_avg_crossnobis"]:::code
    SLSUBJ[("surfaceWB/subj*/<br/>*.func.gii (per subject)")]:::data
    POOL["searchlight.py<br/>pool_searchlight"]:::code
    SLGRP[("surfaceWB/<br/>searchlight.*.func.gii (group)")]:::data

    REG --> SLRUN
    SLH5 --> SLRUN --> SLSUBJ --> POOL --> SLGRP

    %% ---------- univariate activation ----------
    ACT["scripts/activation.py<br/>roi_avg &middot; smooth_cifti_contrasts"]:::code
    ACTTSV[("glm*/<br/>ROI.activation.tsv")]:::data
    REG --> ACT --> ACTTSV

    %% ---------- stats & figures ----------
    STATS["G_matrix.py<br/>lme_trained_untrained &middot; fit_lme &middot; Intercept"]:::code
    FIG["notebooks/ + vis.py<br/>plot_im_sess &rarr; figures/"]:::out

    DISSTSV --> STATS --> FIG
    CWTSV --> FIG
    SLGRP --> FIG
    ACTTSV --> FIG
    BEHAV --> FIG

    classDef input fill:#e8eef7,stroke:#5b7fb5,color:#1a2c47;
    classDef code  fill:#eef3ea,stroke:#7a9a5f,color:#2a3a1e;
    classDef data  fill:#f7f1e8,stroke:#c39b56,color:#4a3818;
    classDef out   fill:#f3e8f0,stroke:#a86b97,color:#3f1f39;
```

Legend: blue = external inputs, green = code (module / entry-point script), amber = saved
data artifacts, purple = final figures. Paths are relative to `gl.baseDir`; `*` stands in
for the subject / hemisphere / ROI / session / glm fields (e.g. `subj101`, `L`, `M1`, `glm3`).

Two multivariate branches share the prewhitened betas: an **ROI** branch (`G_obs` →
crossnobis/cosine dissimilarities and PCM component weights) and a **surface searchlight**
branch (per-subject `func.gii` maps pooled into group maps).


