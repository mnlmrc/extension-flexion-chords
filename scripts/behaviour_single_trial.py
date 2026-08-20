import EFC_learningfMRI.behaviour as behav
import EFC_learningfMRI.globals as gl

if __name__=='__main__':
    sns      = gl.participants[10:]
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    for sn in sns:
        for sess in sessions:
            behav.single_trial_behaviour(sn=sn, 
                                         session=sess)
