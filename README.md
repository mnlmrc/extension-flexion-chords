
### ROI definition

- `python rois.py make_cortical_rois --sn <subject number>`: save ROI and hemisphere masks to NIFTI files 

### Univariate activation

- `python betas.py make_betas_cifti --sn <subject number>`: save cortical beta coefficients from 1st-level GLM to CIFTI 
file with condition and runs in the rows and voxels in the columns