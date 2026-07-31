import nibabel as nb 
import nitools as nt
import numpy as np
import os
import EFC_learningfMRI.globals as gl
import nitools as nt

if __name__=='__main__':
    sns         = gl.participants
    glm         = 3
    struct_dict = {'L': 'CortexLeft', 'R': 'CortexRight'}
    for H in gl.Hem:
        struct = struct_dict[H]
        for session in gl.sessions:
            data, data_diff = [], []
            for sn in sns:
                fname      = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'searchlight_crossnobis.{session}.glm{glm}.{H}.func.gii')
                gifti      = nb.load(fname)
                cols       = nt.get_gifti_column_names(gifti)
                tr_col     = cols.index('encoding-trained')
                untr_col   = cols.index('encoding-untrained')
                data_      = nt.get_gifti_data_matrix(gifti)
                data_diff_ = data_[:, tr_col] - data_[:, untr_col]
                data.append(data_); data_diff.append(data_diff_)   

            group = np.nanmean(np.stack(data, axis=0), axis=0)
            gifti = nt.make_func_gifti(group, anatomical_struct=struct, column_names=cols)
            nb.save(gifti, os.path.join(gl.baseDir, gl.surfDir, f'searchlight_crossnobis.{session}.glm{glm}.{H}.func.gii'))

            group_diff = np.nanmean(np.stack(data_diff, axis=0), axis=0)
            gifti      = nt.make_func_gifti(group_diff, anatomical_struct=struct, column_names=['trained-untrained'])
            nb.save(gifti, os.path.join(gl.baseDir, gl.surfDir, f'searchlight_crossnobis_diff.{session}.glm{glm}.{H}.func.gii'))
