import numpy as np
import os
from EFC_learningfMRI.betas import make_cifti_cortex
import EFC_learningfMRI.globals as gl

if __name__=='__main__':
    glm = 1
    for p in gl.participants:
        for s in gl.sessions:
            make_cifti_cortex(p, glm=glm, type='beta')