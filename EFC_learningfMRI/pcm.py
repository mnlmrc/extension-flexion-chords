import numpy as np
import PcmPy as pcm
import pickle
import os
import nibabel as nb
import EFC_learningfMRI.globals as gl
import nitools as nt
import time
import pandas as pd
import argparse
from EFC_learningfMRI.util import get_trained_and_untrained
from imaging_pipelines.util import extract_mle_corr
import imaging_pipelines.model as md

def correlation(sns, glm, rois, atlas_name='ROI'):
        glm_path = os.path.join(gl.baseDir, f'glm{glm}')
        roi_path = os.path.join(gl.baseDir, gl.roiDir)
        pcm_path = os.path.join(gl.baseDir, gl.pcmDir)
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr.p'), "rb")
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
                for H in gl.Hem:
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
                            mask = nb.load(os.path.join(roi_path, f'subj{sn}', f'{atlas_name}.{H}.{roi}.nii'))

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
                                                             Y[s].obs_descriptors['part_vec'],)

                        # save G
                        np.save(os.path.join(pcm_path, f'G_obs.corr_across_sess.glm{glm}.{int(corr[0][-2:])}-'
                                                       f'{int(corr[1][-2:])}.{chord}.{H}.{roi}.npy'), G_obs)

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
        df.to_csv(os.path.join(pcm_path, f'MLE_correlation.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def correlation_rep_suppr():
    glm_path = os.path.join(gl.baseDir, f'glm{glm}')
    roi_path = os.path.join(gl.baseDir, gl.roiDir)
    pcm_path = os.path.join(gl.baseDir, gl.pcmDir)
    f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr.p'), "rb")
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
            for H in gl.Hem:
                for roi in rois:
                    N = len(sns)
                    Y = list()
                    G_obs = np.zeros((N, 8, 8))
                    for s, sn in enumerate(sns):
                        # load data + masks
                        print(f'doing...participant {sn}, {H}, {roi}, sess {int(corr[0][-2:])} vs. '
                                f'{int(corr[1][-2:])}, {chord} chords')
                        betas_cifti = nb.load(os.path.join(glm_path, f'subj{sn}', 'beta.dscalar.nii'))
                        mask = nb.load(os.path.join(roi_path, f'subj{sn}', f'{atlas_name}.{H}.{roi}.nii'))

                        # covert cifti betas to volums
                        betas = nt.volume_from_cifti(betas, struct_names=['CortexLeft', 'CortexRight'])

                        # get trained and untrained chords
                        trained_and_untrained = get_trained_and_untrained(sn)
                        trained, untrained = trained_and_untrained[:4], trained_and_untrained[4:]

                        # get regressor and partition info
                        reginfo = betas_cifti.header.get_axis(0)
                        regr_run = reginfo.name.str.split('.', n=1, expand=True)
                        regressor, run = regr_run.loc[:, 0], regr_run.loc[:, 1]
                        sess = regressor.str.split(',', n=1, expand=True).loc[:, 1]
                        chordID = regressor.str.split(',', n=1, expand=True).loc[:, 0]
                        part_vec = (run % 10).to_numpy()

                        # make booleans for chordID and session
                        chord_bool = chordID.isin(trained if chord=='trained' else untrained).to_numpy()
                        sess_bool = sess.isin(corr).to_numpy()

                        # map chordID to number for better control
                        chord_mapping = {val: i for i, val in enumerate(trained if chord=='trained' else untrained)}
                        chordID = chordID.map(chord_mapping) #.astype(int)

                        # apply bool to chordID, part_vec and betas
                        betas = betas[chord_bool & sess_bool]
                        chordID = chordID[chord_bool & sess_bool]
                        sess = sess[chord_bool & sess_bool]
                        part_vec = part_vec[chord_bool & sess_bool]

                        # make obs_vec
                        cond_vec = sess + ',' + chordID.astype(str)
                        cond_vec = cond_vec.map(cond_vec_mapping).to_numpy()
                        obs_des = {'cond_vec': cond_vec, 'part_vec': part_vec}

                        # remove mean
                        betas_centred = betas - betas.mean(axis=1, keepdims=True)

                        # make Y and est G
                        Y.append(pcm.dataset.Dataset(betas_centred, obs_descriptors=obs_des))
                        G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements,
                                                            Y[s].obs_descriptors['cond_vec'],
                                                            Y[s].obs_descriptors['part_vec'],)

                    # save G
                    np.save(os.path.join(pcm_path, f'G_obs.corr_across_sess_rep.glm{glm}.{int(corr[0][-2:])}-'
                                                    f'{int(corr[1][-2:])}.{chord}.{H}.{roi}.npy'), G_obs)

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
    df.to_csv(os.path.join(pcm_path, f'MLE_correlation.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def pcm_searchlight_sess(sns, glm, n_jobs=8):
    #f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.trained_untrained.p'), "rb")
    #M = pickle.load(f)
    structnames = ['CortexLeft', 'CortexRight']
    glm_path = os.path.join(gl.baseDir, f'glm{glm}')
    cifti_img_name = 'beta.dscalar.nii'
    res_img_name = 'ResMS.nii'
    searchlight_path = os.path.join(gl.baseDir, gl.roiDir)
    surf_path = os.path.join(gl.baseDir, gl.surfDir)
    for n_sess in range(3):
        regr_interest = np.arange(8) + 8 * n_sess
        session = list(gl.sess_mapping.values())[n_sess]
        regressor_mapping = []
        for sn in sns:
            trained_untrained = get_trained_and_untrained(sn)
            regressor_mapping.append({
                f"{chordID},sess{sess:02d}": i
                for i, (chordID, sess) in enumerate(
                    ((c, s) for s in gl.sessions for c in trained_untrained))})
        print(f'Using {n_jobs} CPUs')
        for h, H in enumerate(['L']):
            SL = md.PcmSearchlight(
                #M=M,
                cifti_img=[os.path.join(glm_path, f'subj{sn}', cifti_img_name) for sn in sns],
                res_img=[os.path.join(glm_path, f'subj{sn}', res_img_name) for sn in sns],
                searchlight_list=[os.path.join(searchlight_path, f'subj{sn}', f'searchlight.{H}.h5') for sn in sns],
                structnames=structnames[h],
                regressor_mapping=regressor_mapping,
                regr_interest=regr_interest,
                n_jobs=n_jobs)
            n_centre = SL.n_centre
            #Mc = M[0]
            encoding = np.full((n_centre, SL.N), np.nan)
            D_tr = np.full((n_centre, SL.N), np.nan)
            D_untr = np.full((n_centre, SL.N), np.nan)
            #param_c = np.full((n_centre, Mc.n_param), np.nan)
            #var_tot = np.full((n_centre, SL.N), np.nan)
            #SL._run_searchlight(0)
            #SL.n_centre = 20
            #G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr, good = SL.run_seachlight_parallel()
            G_obs = SL.run_seachlight_parallel()
            for c in range(SL.n_centre):
                G = G_obs[c]
                #var_tot[c] = np.trace(G, axis1=1, axis2=2)
                D = pcm.G_to_dist(G) #np.array([pcm.G_to_dist(G[s]).mean() for s in range(SL.N)])
                encoding[c] = D.mean(axis=(1, 2))
                D_tr[c] = D[:, :4, :4].mean(axis=(1, 2))
                D_untr[c] = D[:, 4:, 4:].mean(axis=(1, 2))
                # if good[c]:
                #     param_c[c] = theta_gr[c][0][:Mc.n_param]

            # # trace to gifti
            # gifti = nt.make_func_gifti(var_tot, anatomical_struct=structnames[h], column_names=sns)
            # nb.save(gifti, os.path.join(surf_path, f'searchlight.var_tot.session{session}.{H}.func.gii'))

            # distance trained to gifti
            gifti = nt.make_func_gifti(encoding, anatomical_struct=structnames[h], column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding.session{gl.sessions[session]}.{H}.func.gii'))

            # distance trained to gifti
            gifti = nt.make_func_gifti(D_tr, anatomical_struct=structnames[h], column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding-trained.session{gl.sessions[session]}.{H}.func.gii'))

            # distance untrained to gifti
            gifti = nt.make_func_gifti(D_untr, anatomical_struct=structnames[h], column_names=sns)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.encoding-untrained.session{gl.sessions[session]}.{H}.func.gii'))

            # # var_expl to gifti
            # var_expl = np.exp(param_c)
            # column_names = ['trained_untrained', 'chordID']
            # gifti = nt.make_func_gifti(var_expl, anatomical_struct=structnames[h], column_names=column_names)
            # nb.save(gifti, os.path.join(surf_path, f'searchlight.var_expl.session{session}.{H}.func.gii'))
