import os
from collections import defaultdict
import numpy as np
import PcmPy as pcm
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.betas import BetasPrewithenedLoader
from EFC_learningfMRI.G_matrix import calc_G_rois


if __name__=='__main__':
    sns    = gl.participants
    glm    = 3
    loader = BetasPrewithenedLoader(sns=sns, glm=glm)
    calc_G_rois(loader)
