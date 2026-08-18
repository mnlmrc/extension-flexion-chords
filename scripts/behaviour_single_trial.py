from EFC_learningfMRI.force import single_trial_behaviour

if __name__=='__main__':
    sn       = 117
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    sessions = [24]

    for sess in sessions:
        single_trial_behaviour(sn=sn, session=sess)
