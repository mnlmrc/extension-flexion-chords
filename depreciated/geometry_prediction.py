# %%
from EFC_learningfMRI.geometry import dissimilarity_prediction
import EFC_learningfMRI.globals as gl 

if __name__=='__main__':
    glm = 3
    atlas = 'ROI'
    dissimilarity_prediction(sns=gl.participants,
                             glm=glm,
                             atlas_name='ROI')
# %%