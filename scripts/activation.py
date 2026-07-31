from EFC_learningfMRI.betas import roi_avg
from EFC_learningfMRI.surface import smooth_cifti_contrasts, average_smoothed_contrasts
import EFC_learningfMRI.globals as gl

if __name__ == "__main__":
    glm = 2
    sns = gl.participants
    cond_names = ('chordID', 'session', 'repetition', )
    roi_avg(sns=sns, atlas_name='ROI', glm=glm, cond_names=cond_names)
    
    # for sn in sns: smooth_cifti_contrasts(sn=sn, glm=glm, stat='con')
    # average_smoothed_contrasts(sns=sns, glm=glm, stat='con')
    
