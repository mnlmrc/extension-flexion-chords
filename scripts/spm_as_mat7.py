import os
import shutil
import subprocess
import EFC_learningfMRI.globals as gl

def save_spm_as_mat7(sn, glm):
    path_glm    = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}')
    path_spm    = os.path.join(path_glm, 'SPM.mat')
    path_backup = path_spm + ".backup"

    if os.path.exists(path_backup):
        pass
    else:
        # Step 1: Backup the original file
        shutil.copy(path_spm, path_backup)
        print(f"Backed up {path_spm} to {path_backup}")

    # Step 2: Run MATLAB command
    matlab_cmd = (
        f"matlab -nodesktop -nosplash -r "
        f"\"load('{path_spm}'); save('{path_spm}', '-struct', 'SPM', '-v7'); exit\""
    )

    subprocess.run(matlab_cmd, shell=True, check=True)
    print(f"Processed {path_spm} with MATLAB")


if __name__=='__main__':
    sns = [116]
    glm = 3

    for sn in sns:
        save_spm_as_mat7(sn, glm)