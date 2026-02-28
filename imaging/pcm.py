import numpy as np
import PcmPy as pcm
import pickle
import os
import nibabel as nb
import globals.path as pth
import nitools as nt
import time
import pandas as pd
import argparse
from util.util import get_trained_and_untrained
import globals.imaging as im
from imaging_pipelines.util import extract_mle_corr
import imaging_pipelines.model as md

def correlation(sns, glm, rois):
        glm_path = os.path.join(pth.baseDir, f'{pth.glmDir}{glm}')
        roi_path = os.path.join(pth.baseDir, pth.roiDir)
        pcm_path = os.path.join(pth.baseDir, pth.pcmDir)
        f = open(os.path.join(pth.baseDir, pth.pcmDir, f'M.corr.p'), "rb")
        Mflex = pickle.load(f)
        chords = ['trained', 'untrained']
        corrs = [['sess03', 'sess09'], ['sess03', 'sess23'], ['sess09', 'sess23']]
        cond_vec_mapping = {
            'sess03,0.0': 0,
            'sess03,1.0': 1,
            'sess03,2.0': 2,
            'sess03,3.0': 3,
            'sess09,0.0': 4,
            'sess09,1.0': 5,
            'sess09,2.0': 6,
            'sess09,3.0': 7,
            'sess23,0.0': 8,
            'sess23,1.0': 9,
            'sess23,2.0': 10,
            'sess23,3.0': 11,
        }
        df = pd.DataFrame()
        for corr in corrs:
            for chord in chords:
                for H in im.Hem:
                    for roi in rois:
                        N = len(sns)
                        Y = list()
                        G_obs = np.zeros((N, 8, 8))
                        for s, sn in enumerate(sns):
                            # load data + masks
                            print(f'doing...participant {sn}, {H}, {roi}, sess {int(corr[0][-2:])} vs. '
                                  f'{int(corr[1][-2:])}, {chord} chords')
                            betas = nb.load(os.path.join(glm_path, f'subj{sn}', 'beta.dscalar.nii'))
                            residuals = nb.load(os.path.join(glm_path, f'subj{sn}', 'residual.dtseries.nii'))
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
                            sess = reginfo.name.str.split(',', n=1, expand=True).loc[:, 1]
                            chordID = reginfo.name.str.split(',', n=1, expand=True).loc[:, 0]
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
                        np.save(os.path.join(pcm_path, f'G_obs.corr_across_sess.glm{glm}.{int(corr[0][-2:])}-'
                                                       f'{int(corr[1][-2:])}.{chord}.glm{glm}.{H}.{roi}.npy'), G_obs)

                        # estimate correlation
                        _, theta = pcm.fit_model_individ(Y, Mflex, fixed_effect='block', fit_scale=False, verbose=False)
                        _, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect='block', fit_scale=True, verbose=False)
                        r_indiv, r_group, SNR = extract_mle_corr(Mflex, theta[0], theta_gr[0], cond_effect=True)
                        df_tmp = pd.DataFrame()
                        df_tmp['participant_id'] = sns
                        df_tmp['r_indiv'] = r_indiv
                        df_tmp['r_group'] = r_group
                        df_tmp['SNR'] = SNR
                        df_tmp['chord'] = chord
                        df_tmp['corr'] = f'sess {int(corr[0][-2:])} vs. {int(corr[1][-2:])}'
                        df_tmp['roi'] = roi
                        df_tmp['Hem'] = H
                        df = pd.concat([df, df_tmp])

        # save dataframe
        df.to_csv(os.path.join(pcm_path, f'correlation_across_sessions.glm{glm}.BOLD.tsv'), sep='\t', index=False)
