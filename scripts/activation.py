from EFC_learningfMRI.betas import roi_avg
from EFC_learningfMRI.surface import smooth_cifti_contrasts
import EFC_learningfMRI.globals as gl

if __name__ == "__main__":
    glm = 3
    sns = gl.participants

    roi_avg(sns=sns, atlas_name='ROI', glm=glm)
    
    smooth_cifti_contrasts(sns=sns, glm=glm, stat='con')
    
