from EFC_learningfMRI.searchlight import make_searchlight


if __name__=='__main__':
    sns = [114] # gl.participants
    for sn in sns:
        make_searchlight(sn=sn)