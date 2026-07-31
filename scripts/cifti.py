from EFC_learningfMRI.betas import CiftiCortex
import EFC_learningfMRI.globals as gl

if __name__=='__main__':
    sns = gl.participants
    glm = 6

    for sn in sns:
        cifti = CiftiCortex(sn, glm)
        cifti.beta()
        cifti.contrast()
        cifti.residual()
