import os
import ants
import globals as gl
import subprocess
import numpy as np
import nibabel as nb

# ----- paths (edit) -----
sn = 101
BN = 1

t1_path  = os.path.join(gl.baseDir, 'EFC_learningfMRI', gl.anatDir, f'subj{sn}', f'subj{sn}_T1w.nii')
epi_path = os.path.join(gl.baseDir, 'EFC_learningfMRI', gl.imagingDir, f'subj{sn}', f'usubj{sn}_run_{BN:02d}.nii')
out_path = os.path.join(gl.baseDir, 'EFC_learningfMRI', 'bold_normalization', f'subj{sn}_run_{BN:02d}_warped.nii')
temp_path = os.path.join(gl.baseDir, 'EFC_learningfMRI', 'bold_normalization', f'template.nii')

template_ants = ants.image_read(temp_path)
T1 = ants.image_read(t1_path)
epi = ants.image_read(epi_path)
mask_c1 = ants.image_read(os.path.dirname(t1_path) + f'/c1subj{sn}_T1w.nii')
mask_c2 = ants.image_read(os.path.dirname(t1_path) + f'/c2subj{sn}_T1w.nii')
mask_c3 = ants.image_read(os.path.dirname(t1_path) + f'/c3subj{sn}_T1w.nii')

mask = np.clip(mask_c1+mask_c2+mask_c3, 0, 1)
ants.image_write(mask, os.path.dirname(temp_path) + "/mask_T1.nii")

print('Registering...')

reg_t1 = ants.registration(
    fixed=template_ants,
    moving=T1,
    type_of_transform='SyN',
    moving_mask=mask,
    mask_all_stages=True,
    outprefix=os.path.dirname(temp_path) + f'/subj{sn}_to_MNI_',
    verbose=True
)

warped_T1 = ants.apply_transforms(
    fixed=template_ants,
    moving=T1,
    transformlist=reg_t1['fwdtransforms'],
    interpolator='linear'
)

ants.image_write(warped_T1, os.path.dirname(temp_path) + "/T1_in_MNI.nii")

print('Registration done!')

vol = epi[:, :, :, 0]

reg_epi = ants.registration(
    fixed=T1,
    moving=vol,
    type_of_transform='Rigid',  # or 'Affine'
    outprefix=os.path.dirname(temp_path) + f'/subj{sn}_EPI_to_T1_',
    verbose=True
)

warped_epi_T1 = ants.apply_transforms(
    fixed=T1,
    moving=epi[:, :, :, 0],
    transformlist=reg_epi['fwdtransforms'],
    interpolator='linear'
)

ants.image_write(warped_T1, os.path.dirname(temp_path) + "/EPI_in_T1.nii")

xfms = [
    reg_epi['fwdtransforms'][0],
    reg_t1['fwdtransforms'][1],
    reg_t1['fwdtransforms'][0],
]

warped = ants.apply_transforms(
    fixed=template_ants,
    moving=vol,
    transformlist=xfms,
    interpolator='linear',
    verbose=True
)

ants.image_write(
    warped,
    out_path
)