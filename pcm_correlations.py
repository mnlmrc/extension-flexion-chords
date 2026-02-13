import numpy as np
import PcmPy as pcm
import pickle
import os
import nibabel as nb
import globals as gl
import nitools as nt
import time
import pandas as pd
import argparse
from util import get_trained_and_untrained
from imaging_pipelines.util import extract_mle_corr
import imaging_pipelines.model as md

def main(args):
    if args.what == 'correlation_across_sessions':
        glm_path = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}')
        roi_path = os.path.join(gl.baseDir, gl.roiDir)
        pcm_path = os.path.join(gl.baseDir, gl.pcmDir)
        Hem = ['L', 'R']
        rois = gl.rois['ROI']
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr.p'), "rb")
        Mflex = pickle.load(f)
        chords = ['trained', 'untrained']
        corrs = [['03', '09'], ['03', '23'], ['09', '23']]
        cond_vec_mapping = {
            '03,0.0': 0,
            '03,1.0': 1,
            '03,2.0': 2,
            '03,3.0': 3,
            '09,0.0': 4,
            '09,1.0': 5,
            '09,2.0': 6,
            '09,3.0': 7,
            '23,0.0': 8,
            '23,1.0': 9,
            '23,2.0': 10,
            '23,3.0': 11,
        }
        df = pd.DataFrame()
        for corr in corrs:
            for chord in chords:
                for H in Hem:
                    for roi in rois:
                        N = len(args.sns)
                        Y = list()
                        G_obs = np.zeros((N, 8, 8))
                        for s, sn in enumerate(args.sns):
                            # load data + masks
                            print(f'doing...participant {sn}, {H}, {roi}, sess {int(corr[0])} vs. {int(corr[1])}, {chord} chords')
                            betas = nb.load(os.path.join(glm_path, f'subj{sn}', 'beta.dscalar.nii'))
                            residuals = nb.load(os.path.join(glm_path, f'subj{sn}', 'ResMS.nii'))
                            mask = nb.load(os.path.join(roi_path, f'subj{sn}', f'ROI.{H}.{roi}.nii'))

                            # covert cifti betas to volums
                            betas = nt.volume_from_cifti(betas, struct_names=['CortexLeft', 'CortexRight'])

                            # do prewhitening
                            betas_prewhitened = md.calc_prewhitened_betas(betas, residuals, mask)

                            # get trained and untrained chords
                            trained_and_untrained = get_trained_and_untrained(sn)
                            trained, untrained = trained_and_untrained[:4], trained_and_untrained[4:]

                            # get regressor and partition info
                            reginfo = pd.read_csv(os.path.join(glm_path, f'subj{sn}', 'reginfo.tsv'), sep='\t')
                            sess = reginfo.name.str.split(',', n=1, expand=True)[0]
                            chordID = reginfo.name.str.split(',', n=1, expand=True)[1]
                            part_vec = (reginfo.run % 10).to_numpy()

                            # make booleans for chordID and session
                            chord_bool = chordID.isin(trained if chord=='trained' else untrained).to_numpy()
                            sess_bool = sess.isin(corr).to_numpy()

                            # map chordID to number for better control
                            chord_mapping = {val: i for i, val in enumerate(trained if chord=='trained' else untrained)}
                            chordID = chordID.map(chord_mapping) #.astype(int)

                            # apply bool to chordID, part_vec and betas
                            betas_prewhitened = betas_prewhitened[chord_bool & sess_bool]
                            chordID = chordID[chord_bool & sess_bool]
                            sess = sess[chord_bool & sess_bool]
                            part_vec = part_vec[chord_bool & sess_bool]

                            # make obs_vec
                            cond_vec = sess + ',' + chordID.astype(str)
                            cond_vec = cond_vec.map(cond_vec_mapping).to_numpy()
                            obs_des = {'cond_vec': cond_vec, 'part_vec': part_vec}

                            # remove mean
                            betas_prewhitened = betas_prewhitened - betas_prewhitened.mean(axis=1, keepdims=True)

                            # make Y and est G
                            Y.append(pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des))
                            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements,
                                                             Y[s].obs_descriptors['cond_vec'],
                                                             Y[s].obs_descriptors['part_vec'],
                                                             X=pcm.indicator(part_vec),)

                        # save G
                        np.save(os.path.join(pcm_path, f'G_obs.corr_across_sess.glm{args.glm}.{int(corr[0])}-{int(corr[0])}.{chord}.glm{args.glm}.{H}.{roi}.npy'), G_obs)

                        # estimate correlation
                        _, theta = pcm.fit_model_individ(Y, Mflex, fixed_effect='block', fit_scale=False, verbose=False)
                        _, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect='block', fit_scale=True, verbose=False)
                        r_indiv, r_group, SNR = extract_mle_corr(Mflex, theta[0], theta_gr[0], cond_effect=True)
                        df_tmp = pd.DataFrame()
                        df_tmp['participant_id'] = args.sns
                        df_tmp['r_indiv'] = r_indiv
                        df_tmp['r_group'] = r_group
                        df_tmp['SNR'] = SNR
                        df_tmp['chord'] = chord
                        df_tmp['corr'] = f'sess {int(corr[0])} vs. {int(corr[1])}'
                        df_tmp['roi'] = roi
                        df_tmp['Hem'] = H
                        df = pd.concat([df, df_tmp])

        # save dataframe
        df.to_csv(os.path.join(pcm_path, f'correlation_across_sessions.glm{args.glm}.BOLD.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int,
                        default=[101, 102, 103, 104, 105, 106, 107])
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=16)

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')