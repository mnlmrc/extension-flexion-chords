import os
from EFC_learningfMRI.betas import BetasPrewithenedLoader
from EFC_learningfMRI.pcm import fit_component_model
import EFC_learningfMRI.globals as gl


if __name__=='__main__':
    sns    = [117] #gl.participants
    glm    = 3
    loader = BetasPrewithenedLoader(sns=sns, glm=glm, residual_fname='residual.dtseries.nii')

    fit_component_model(loader)
