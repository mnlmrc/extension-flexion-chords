from EFC_learningfMRI.betas import roi_avg
from EFC_learningfMRI.surface import smooth_cifti_contrasts

if __name__ == "__main__":
    glm = 3
    sns = [101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113]
    #smooth_cifti_contrasts(sns=sns, glm=glm, stat='con')
    roi_avg(sns=sns, atlas_name='ROI', glm=glm)
    roi_avg(sns=sns, atlas_name='BA_handArea', glm=glm)
