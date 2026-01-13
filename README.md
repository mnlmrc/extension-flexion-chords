
### ROI definition

Surface-based definition of cortical ROIs is implemented in the `rois` module using the `Rois` class 
from [imaging_pipelines](https://github.com/mnlmrc/imaging_pipelines), which uses the `atlas_map` module from [Functional_Fusion](https://github.com/DiedrichsenLab/Functional_Fusion):

- Save ROI and hemisphere masks to NIFTI files: 
    `python rois.py make_cortical_rois --sn <subject number>`

### Univariate activation

The beta coefficients estimated in 1st level GLM 
are stored in a single CIFTI file using the make_cifti_betas function from [imaging_pipelines](https://github.com/mnlmrc/imaging_pipelines):

- Save cortical beta coefficients to single CIFTI file: `python betas.py make_betas_cifti --sn <subject number>`
- Save cortical contrasts to single CIFTI file: `python betas.py save_contrasts_cifti --sn <subject number>`
- Save average contrasts across participants for each ROI to .tsv file: `python betas.py avg_roi_contrasts`