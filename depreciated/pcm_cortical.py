from imaging_pipelines.modeling import PcmRois, calc_prewhitened_betas
import globals as gl
import numpy as np
import os
import pickle
import time
import argparse
import PcmPy as pcm
import pandas as pd
from util import get_trained_and_untrained
import nibabel as nb


def get_diag_block(M, m, block_index):
    """
    Extract the m x m block at `block_index` along the diagonal of square matrix M.

    Args:
        M (ndarray): N x N matrix
        m (int): size of the square diagonal blocks
        block_index (int): which diagonal block to extract (0-based)

    Returns:
        ndarray: m x m submatrix
    """
    start = block_index * m
    end = start + m
    return M[start:end, start:end]


def get_block(M, m, i, j):
    """
    Extract the m x m block at block-row i and block-column j from M.

    Args:
        M (ndarray): N x N matrix
        m (int): size of each block
        i (int): block row index
        j (int): block column index

    Returns:
        ndarray: m x m block at position (i, j)
    """
    row_start = i * m
    col_start = j * m
    return M[row_start:row_start + m, col_start:col_start + m]


def make_chord_mapping_for_models():
    finger = {
        '11911': np.array([1, 1, 0, 1, 1]),
        '12129': np.array([1, 1, 1, 1, 0]),
        '12291': np.array([1, 1, 1, 0, 1]),
        '21291': np.array([1, 1, 1, 0, 1]),
        '21911': np.array([1, 1, 0, 1, 1]),
        '22911': np.array([1, 1, 0, 1, 1]),
        '91211': np.array([0, 1, 1, 1, 1]),
        '92122': np.array([0, 1, 1, 1, 1]),
    }

    config = {
        '11911': np.array([1, 1, 0, 1, 1]),
        '12129': np.array([1, -1, 1, -1, 0]),
        '12291': np.array([1, -1, -1, 0, 1]),
        '21291': np.array([-1, 1, -1, 0, 1]),
        '21911': np.array([-1, 1, 0, 1, 1]),
        '22911': np.array([-1, -1, 0, 1, 1]),
        '91211': np.array([0, 1, -1, 1, 1]),
        '92122': np.array([0, -1, 1, -1, -1]),
    }

    return finger, config


def make_models():
    C = pcm.centering(8)

    # models
    finger = np.zeros((8, 5))
    finger[0] = np.array([1, 1, 0, 1, 1]) # 11911
    finger[1] = np.array([1, 1, 1, 1, 0]) # 12129
    finger[2] = np.array([1, 1, 1, 0, 1]) # 12291
    finger[3] = np.array([1, 1, 1, 0, 1]) # 21291
    finger[4] = np.array([1, 1, 0, 1, 1]) # 21911
    finger[5] = np.array([1, 1, 0, 1, 1]) # 22911
    finger[6] = np.array([0, 1, 1, 1, 1]) # 91211
    finger[7] = np.array([0, 1, 1, 1, 1]) # 92122

    config = np.zeros((8, 5))
    config[0] = np.array([1, 1, 0, 1, 1])  # 11911
    config[1] = np.array([1, -1, 1, -1, 0])  # 12129
    config[2] = np.array([1, -1, -1, 0, 1])  # 12291
    config[3] = np.array([-1, 1, -1, 0, 1])  # 21291
    config[4] = np.array([-1, 1, 0, 1, 1])  # 21911
    config[5] = np.array([-1, -1, 0, 1, 1])  # 22911
    config[6] = np.array([0, 1, -1, 1, 1])  # 91211
    config[7] = np.array([0, -1, 1, -1, -1])  # 92122

    chord = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # centering
    finger = C @ finger
    config = C @ config

    # second moment
    G_finger = finger @ finger.T
    G_config = config @ config.T
    G_component = np.array([G_finger / np.trace(G_finger),
                            G_config / np.trace(G_config),
                            ])

    M = []
    M.append(pcm.FixedModel('null', np.eye(8)))
    M.append(pcm.FixedModel('finger', G_finger))
    M.append(pcm.FixedModel('config', G_config))
    M.append(pcm.ComponentModel('component', G_config))
    M.append(pcm.FreeModel('ceil', 8))

    return M


def make_models_individ(sn):
    """
    Make models to be fit in individual participants for one day of testing.
    Args:
        sn: participant number

    Returns:
        list of models
    """
    C = pcm.centering(8)

    chords = get_trained_and_untrained(sn)

    map_finger, map_config = make_chord_mapping_for_models()

    # make models
    chord = C @ np.array([-1, -1, -1, -1, 1, 1, 1, 1])  # trained vs untrained
    finger = C @ np.array([map_finger[c] for c in chords])  # finger
    config = C @ np.array([map_config[c] for c in chords])

    # make Gs
    G_chord = np.outer(chord, chord)
    G_finger = finger @ finger.T
    G_config = config @ config.T
    G_component = np.array([
                            G_chord / np.trace(G_chord),
                            G_finger / np.trace(G_finger),
                            G_config / np.trace(G_config),
                            ])

    M = []
    M.append(pcm.FixedModel('null', np.eye(8)))
    M.append(pcm.FixedModel('chord', G_chord))
    M.append(pcm.FixedModel('finger', G_finger))
    M.append(pcm.FixedModel('config', G_config))
    M.append(pcm.ComponentModel('component', G_component))
    M.append(pcm.FreeModel('ceil', 8))

    return M


def main(args):
    Hem = ['L', 'R']
    experiment = 'EFC_learningfMRI'
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
    glm_path = os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{args.glm}')
    cifti_img = 'beta.dscalar.nii'
    roi_path = os.path.join(gl.baseDir, experiment, gl.roiDir)
    pcm_path = os.path.join(gl.baseDir, experiment, gl.pcmDir)
    days = [3, 9, 23]
    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    if args.what == 'fit_avg':

        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'models.p'), "rb")
        M  = pickle.load(f)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
        glm_path = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}')
        cifti_img = 'beta.dscalar.nii'
        roi_path = os.path.join(gl.baseDir, gl.roiDir)

        days = [3, 9, 23]
        chords = (gl.chordID)
        chord_mapping = {chord: i for i, chord in enumerate(chords)}
        regressor_mapping = {
            (f'{day:02d},{chord}'): chord_mapping[chord]  # or some initial value instead of None
            for day in days
            for chord in chords
        }

        R = PcmRois(args.snS, M, glm_path, cifti_img, roi_path, roi_imgs, regressor_mapping=regressor_mapping,
                 regr_of_interest=[0, 1, 2, 3, 4, 5, 6, 7])
        # res = R.run_pcm_in_roi(roi_imgs[0])
        res = R.run_parallel_pcm_across_rois()

        for H in Hem:
            for roi in rois:
                r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')

                path = os.path.join(gl.baseDir, gl.pcmDir)
                os.makedirs(path, exist_ok=True)

                res['T_in'][r].to_pickle(os.path.join(path, f'T_in.glm{args.glm}.{H}.{roi}.p'))
                res['T_cv'][r].to_pickle(os.path.join(path, f'T_cv.glm{args.glm}.{H}.{roi}.p'))
                res['T_gr'][r].to_pickle(os.path.join(path, f'T_gr.glm{args.glm}.{H}.{roi}.p'))

                np.save(os.path.join(path, f'G_obs.glm{args.glm}.{H}.{roi}.npy'), res['G_obs'][r])

                f = open(os.path.join(path, f'theta_in.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_in'][r], f)
                f = open(os.path.join(path, f'theta_cv.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_cv'][r], f)
                f = open(os.path.join(path, f'theta_gr.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_gr'][r], f)
    if args.what == 'fit_individ':
        R = PcmRois(glm_path=glm_path, cifti_img=cifti_img, roi_path=roi_path, roi_imgs=roi_imgs)
        for sn in args.sns:
            path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
            os.makedirs(path, exist_ok=True)
            M = make_models_individ(sn)
            chords = get_trained_and_untrained(sn)
            regressor_mapping = {
                f'{day:02d},{chord}': i
                for i, (day, chord) in enumerate(
                    (d, c) for d in days for c in chords
                )
            }
            R.regressor_mapping = regressor_mapping
            for day in days:
                for roi_img in roi_imgs:
                    atlas, H, roi, _ = roi_img.split('.')
                    print(f'fitting individual...{H}, {roi}, day{day}')
                    regr_of_interest=np.array([regressor_mapping[regr] for regr in regressor_mapping.keys()
                                               if day==int(regr.split(',')[0])])
                    R.regr_of_interest = regr_of_interest
                    Y, _ = R._make_roi_dataset_within(roi_img, sn)
                    T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=False, verbose=True, fixed_effect='block')
                    T_in.to_pickle(os.path.join(path, f'T_in.day{day}.{H}.{roi}.p'))
                    f = open(os.path.join(path, f'theta_in.day{day}.{H}.{roi}.p'), 'wb')
                    pickle.dump(theta_in, f)
    if args.what == 'fit_across_sessions':
        path = os.path.join(gl.baseDir, gl.pcmDir)
        f = open(os.path.join(path, f'M.across_sessions.p'), "rb")
        M = pickle.load(f)
        M = M[:-1]
        R = PcmRois(glm_path=glm_path, cifti_img=cifti_img, roi_path=roi_path, roi_imgs=roi_imgs, M=M)
        for roi_img in roi_imgs:
            atlas, H, roi, _ = roi_img.split('.')
            print(f'saving G_obs...{H},{roi}')
            Y = []
            for sn in args.sns:
                chords = get_trained_and_untrained(sn)
                regressor_mapping = {
                    f'{day:02d},{chord}': i
                    for i, (day, chord) in enumerate(
                        (d, c) for d in days for c in chords
                    )
                }
                R.regressor_mapping = regressor_mapping
                Yi, G_obs = R._make_roi_dataset_within(roi_img, sn)
                Y.append(Yi)
            T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr = R._fit_model_to_dataset(Y)

            T_in.to_pickle(os.path.join(path, f'T_in.across_sessions.{H}.{roi}.p'))
            f = open(os.path.join(path, f'theta_in.across_sessions.{H}.{roi}.p'), 'wb')
            pickle.dump(theta_in, f)

            T_cv.to_pickle(os.path.join(path, f'T_cv.across_sessions.{H}.{roi}.p'))
            f = open(os.path.join(path, f'theta_cv.across_sessions.{H}.{roi}.p'), 'wb')
            pickle.dump(theta_cv, f)

            T_gr.to_pickle(os.path.join(path, f'T_gr.across_sessions.{H}.{roi}.p'))
            f = open(os.path.join(path, f'theta_gr.across_sessions.{H}.{roi}.p'), 'wb')
            pickle.dump(theta_gr, f)
    if args.what == 'calc_G':
        R = PcmRois(glm_path=glm_path, cifti_img=cifti_img, roi_path=roi_path, roi_imgs=roi_imgs)
        for roi_img in roi_imgs:
            atlas, H, roi, _ = roi_img.split('.')
            print(f'saving G_obs...{H},{roi}')
            for sn in args.sns:
                chords = get_trained_and_untrained(sn)
                regressor_mapping = {
                    f'{day:02d},{chord}': i
                    for i, (day, chord) in enumerate(
                        (d, c) for d in days for c in chords
                    )
                }
                R.regressor_mapping = regressor_mapping
                _, G_obs = R._make_roi_dataset_within(roi_img, sn)
                filename = f'G_obs.{sn}.{H}.{roi}.npz' #if args.demean==False #else f'G_obs.demean.{sn}.{H}.{roi}.npz'
                np.savez(os.path.join(pcm_path, filename),
                    G_obs=G_obs,
                    conds=[f'{int(regr.split(",")[0])},{regr.split(",")[1]}' for regr in regressor_mapping.keys()])

    if args.what == 'corr':
        for H in Hem:
            for roi in rois:
                for s, sn in enumerate(args.sns):
                    betas = nb.load(os.path.join(glm_path, f'subj{sn}', cifti_img))
                    row_names = pd.Index(betas.header.get_axis(0).name)
                    regr, part_vec = zip(*row_names.str.split('.'))
                    part_vec = np.array(part_vec).astype(int)
                    day, chordID = zip(*pd.Index(regr).str.split(','))
                    day, chordID = np.array(day), np.array(chordID)

                    meta = pd.DataFrame({
                        "row": np.arange(len(row_names)),
                        "day": day,
                        "chordID": chordID,
                    })

                    chords = get_trained_and_untrained(experiment, sn)
                    trained = chords[:4]
                    untrained = chords[4:]

                    betas = betas.get_fdata()
                    regr, part_vec = zip(*pd.Index(betas.header.get_axis(0).name).str.split('.'))
                    day, chordID = zip(*pd.Index(regr).str.split(','))


                    pass

    if args.what == 'pool_G':
        for H in Hem:
            for roi in rois:
                G_obs = []
                for sn in args.sns:
                    filename = f'G_obs.{sn}.{H}.{roi}.npz' if args.demean == False else f'G_obs.demean.{sn}.{H}.{roi}.npz'
                    npz = np.load(os.path.join(pcm_path, filename))
                    G_obs.append(npz['G_obs'])
                G_obs = np.array(G_obs)
                filename = f'G_obs.{H}.{roi}.npy' if args.demean == False else f'G_obs.demean.{H}.{roi}.npy'
                np.save(os.path.join(pcm_path, filename), G_obs)

    if args.what == 'calc_dissimilarities':
        dissimilarity = {
            'day': [],
            'roi': [],
            'Hem': [],
            'sn': [],
            'crossnobis_trained': [],
            'crossnobis_untrained': [],
            'crossnobis_between': [],
            'cosine_between': [],
        }
        for s, sn in enumerate(args.sns):
            for H in Hem:
                for roi in rois:
                    G_obs = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.{H}.{roi}.npy'))
                    G = G_obs[s]
                    D = pcm.G_to_dist(G)
                    cosine = pcm.G_to_cosine(G)
                    for d, day in enumerate(days):
                        block_diag_D = get_diag_block(D, 8, d)
                        block_D = get_block(block_diag_D, 4, 0, 1)
                        block_diag_cos = get_diag_block(cosine, 8, d)
                        block_cos = get_block(block_diag_cos, 4, 0, 1)
                        mask = np.tri(4, k=-1, dtype=bool)
                        crossnobis_trained = block_diag_D[:4, :4][mask].mean()
                        crossnobis_untrained = block_diag_D[4:, 4:][mask].mean()
                        dissimilarity['crossnobis_trained'].append(crossnobis_trained)
                        dissimilarity['crossnobis_untrained'].append(crossnobis_untrained)
                        dissimilarity['crossnobis_between'].append(block_D.mean())
                        dissimilarity['cosine_between'].append(np.arccos(block_cos.mean()))
                        dissimilarity['day'].append(day)
                        dissimilarity['roi'].append(roi)
                        dissimilarity['sn'].append(sn)
                        dissimilarity['Hem'].append(H)
        dissimilarity = pd.DataFrame(dissimilarity)
        dissimilarity.to_csv(os.path.join(pcm_path, f'dissimilarity.tsv'), sep='\t', index=False)

    # if args.what == 'corr':
    #     rng = np.random.default_rng(0)  # seed for reprodocibility
    #     # f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "rb")
    #     # Mflex = pickle.load(f)
    #     for H in Hem:
    #         for roi in rois:
    #             N = len(args.sns)
    #             Y = list()
    #             r = roi_imgs.index(f'ROI.{H}.{roi}.nii')
    #             print(f'doing...ROI.{H}.{roi}')
    #             for s, sn in enumerate(args.sns):
    #                 betas_prewhitened, obs_des = calc_prewhitened_betas(glm_path=os.path.join(glm_path, f'subj{sn}'),
    #                                                                     cifti_img='beta.dscalar.nii',
    #                                                                     res_img='ResMS.nii',
    #                                                                     roi_path=os.path.join(roi_path, f'subj{sn}'),
    #                                                                     roi_img=roi_imgs[r])
    #                 trained = pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.')
    #                 cond_vec, part_vec = obs_des['cond_vec'], obs_des['part_vec']
    #                 betas_reduced, cond_vec_reduced, part_vec_reduced = [], [], []
    #                 for part in np.unique(part_vec):
    #                     mask = part_vec == part
    #                     beta_tmp = betas_prewhitened[mask]
    #                     cond_tmp = cond_vec[mask]
    #
    #                     day, chordID = np.array([item.split(',') for item in cond_tmp]).T
    #
    #                     pre = beta_tmp[(day == day1) | (chordID in trained)]
    #                     post = beta_tmp[(day == day2) | (chordID in trained)]
    #
    #                     betas_reduced.extend([pre, post])
    #                     cond_vec_reduced.extend(['pre', 'pre', 'pre', 'pre', 'exec','exec','exec','exec',])
    #                     part_vec_reduced.extend([part] * 8)
    #                 pass

    # if args.what == 'corr':
    #
    #     f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr.p'), "rb")
    #     M = pickle.load(f)
    #
    #     chordID = gl.chordID
    #     regressor_mapping = {
    #         f'{day:02d},{chord}': i
    #         for i, (day, chord) in enumerate(
    #             (d, c) for d in days for c in chordID
    #         )
    #     }
    #
    #     corrs = [3, 9], [9, 23]
    #
    #     path = os.path.join(gl.baseDir, gl.pcmDir)
    #     os.makedirs(path, exist_ok=True)
    #
    #     R = PcmRois(glm_path=glm_path, cifti_img=cifti_img, roi_path=roi_path, roi_imgs=roi_imgs,
    #                 regressor_mapping=regressor_mapping)
    #
    #     for chord in ['trained', 'untrained']:
    #         for corr in corrs:
    #             for roi_img in roi_imgs:
    #                 atlas, H, roi, _ = roi_img.split('.')
    #                 print(f'correlation...{H},{roi}')
    #                 Y = []
    #                 for sn in args.sns:
    #                     chords = pinfo[pinfo.sn == sn].reset_index()[chord][0].split('.')
    #                     regr_of_interest = [
    #                         idx for key, idx in regressor_mapping.items()
    #                         if key.split(',')[1] in chords and int(key.split(',')[0]) in corr
    #                     ]
    #                     R.regr_of_interest = regr_of_interest
    #                     Y_tmp,_ = R._make_roi_dataset_within(roi_img, sn)
    #                     Y.append(Y_tmp)
    #
    #                 T_in, theta_in = pcm.fit_model_individ(Y, M, fixed_effect='block', fit_scale=False)
    #                 T_gr, theta_gr = pcm.fit_model_group(Y, M, fixed_effect='block', fit_scale=True)
    #
    #                 T_in.to_pickle(os.path.join(path, f'T_in.corr.{chord}.glm{args.glm}.{corr[0]}-{corr[1]}.{H}.{roi}.p'))
    #                 T_gr.to_pickle(os.path.join(path, f'T_gr.corr.{chord}.glm{args.glm}.{corr[0]}-{corr[1]}.{H}.{roi}.p'))
    #
    #                 f = open(os.path.join(path, f'theta_in.corr.{chord}.glm{args.glm}.{corr[0]}-{corr[1]}.{H}.{roi}.p'), 'wb')
    #                 pickle.dump(theta_in, f)
    #
    #                 f = open(os.path.join(path, f'theta_gr.corr.{chord}.glm{args.glm}.{corr[0]}-{corr[1]}.{H}.{roi}.p'), 'wb')
    #                 pickle.dump(theta_gr, f)

    # if args.what == 'corr_across_days_chord':
    #
    #     f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr_chord.p'), "rb")
    #     M = pickle.load(f)
    #
    #     pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    #
    #     chordID = gl.chordID
    #     regressor_mapping = {
    #         f'{day:02d},{chord}': i
    #         for i, (day, chord) in enumerate(
    #             (d, c) for d in days for c in chordID
    #         )
    #     }
    #
    #     corrs = [3, 9], [9, 23]
    #
    #     path = os.path.join(gl.baseDir, gl.pcmDir)
    #     os.makedirs(path, exist_ok=True)
    #
    #     R = PcmRois(glm_path=glm_path, cifti_img=cifti_img, roi_path=roi_path, roi_imgs=roi_imgs,
    #                 regressor_mapping=regressor_mapping)
    #
    #     for corr in corrs:
    #         for roi_img in roi_imgs:
    #             atlas, H, roi, _ = roi_img.split('.')
    #             print(f'correlation...{H},{roi}')
    #             for ch in chordID:
    #                 Y = []
    #                 for sn in args.sns:
    #                     regr_of_interest = [idx for key, idx in regressor_mapping.items()
    #                         if key.split(',')[1]==ch and int(key.split(',')[0]) in corr]
    #                     R.regr_of_interest = regr_of_interest
    #                     Y_tmp,_ = R._make_roi_dataset_within(roi_img, sn)
    #                     Y.append(Y_tmp)
    #
    #                 T_in, theta_in = pcm.fit_model_individ(Y, M, fixed_effect=None, fit_scale=False)
    #                 T_gr, theta_gr = pcm.fit_model_group(Y, M, fixed_effect=None, fit_scale=True)
    #
    #                 T_in.to_pickle(os.path.join(path, f'T_in.corr.glm{args.glm}.{corr[0]}-{corr[1]}.{ch}.{H}.{roi}.p'))
    #                 T_gr.to_pickle(os.path.join(path, f'T_gr.corr.glm{args.glm}.{corr[0]}-{corr[1]}.{ch}.{H}.{roi}.p'))
    #
    #                 f = open(os.path.join(path, f'theta_in.corr.glm{args.glm}.{corr[0]}-{corr[1]}.{ch}.{H}.{roi}.p'), 'wb')
    #                 pickle.dump(theta_in, f)
    #
    #                 f = open(os.path.join(path, f'theta_gr.corr.glm{args.glm}.{corr[0]}-{corr[1]}.{ch}.{H}.{roi}.p'), 'wb')
    #                 pickle.dump(theta_gr, f)
    #
    # if args.what == 'pool_corr_likelihood':
    #
    #     chordID = gl.chordID
    #     corrs = [3, 9], [9, 23]
    #     path = os.path.join(gl.baseDir, gl.pcmDir)
    #
    #     df = pd.DataFrame()
    #     for corr in corrs:
    #         for H in Hem:
    #             for roi in rois:
    #                 for ch in chordID:
    #                     f = open(os.path.join(path, f'T_in.corr.glm{args.glm}.{corr[0]}-{corr[1]}.{ch}.{H}.{roi}.p'), 'rb')
    #                     T = pickle.load(f)
    #                     L = T.likelihood
    #                     r = L.columns
    #                     L = L.to_numpy()
    #                     L = L - L.mean(axis=1).reshape(-1, 1)
    #                     for i, l in enumerate(L):
    #                         sn = args.sns[i]
    #                         trained = trained = pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.')
    #                         chord = 'trained' if ch in trained else 'untrained'
    #                         df_tmp = pd.DataFrame(data={r[j]: [l[j]] for j in range(len(r))})
    #                         df_tmp['chord'] = chord
    #                         df_tmp['chordID'] = ch
    #                         df_tmp['roi'] = roi
    #                         df_tmp['Hem'] = H
    #                         df_tmp['sn'] = sn
    #                         df_tmp['corr'] = f'{corr[0]}-{corr[1]}'
    #                         df_tmp_melt = df_tmp.melt(id_vars=['chordID', 'sn', 'chord', 'roi', 'Hem', 'corr'],
    #                                                   value_vars=r, var_name='r', value_name='likelihood')
    #                         df = pd.concat([df, df_tmp_melt])
    #
    #     df.to_csv(os.path.join(path, 'correlation.likelihood.tsv'), sep='\t', index=False)
    #
    # if args.what == 'pool_corr':
    #
    #     f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr_chord.p'), "rb")
    #     M = pickle.load(f)
    #
    #     chordID = gl.chordID
    #     corrs = [3, 9], [9, 23]
    #     path = os.path.join(gl.baseDir, gl.pcmDir)
    #
    #     dict_corr = {'r': [], 'Hem': [], 'roi': [], 'corr': [], 'sn': [], 'chordID': [], 'chord': []}
    #     for corr in corrs:
    #         for H in Hem:
    #             for roi in rois:
    #                 for ch in chordID:
    #                     f = open(os.path.join(path, f'theta_in.corr.glm{args.glm}.{corr[0]}-{corr[1]}.{ch}.{H}.{roi}.p'),
    #                              'rb')
    #                     theta = pickle.load(f)[-1]
    #                     maxr = M[-1].get_correlation(theta)
    #                     for i, r in enumerate(maxr):
    #                         sn = args.sns[i]
    #                         trained = trained = pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.')
    #                         chord = 'trained' if ch in trained else 'untrained'
    #                         dict_corr['r'].append(r)
    #                         dict_corr['Hem'].append(H)
    #                         dict_corr['roi'].append(roi)
    #                         dict_corr['corr'].append(f'{corr[0]}-{corr[1]}')
    #                         dict_corr['sn'].append(sn)
    #                         dict_corr['chord'].append(chord)
    #                         dict_corr['chordID'].append(ch)
    #
    #     df = pd.DataFrame(dict_corr)
    #
    #     df.to_csv(df.to_csv(os.path.join(path, 'correlation.pearsonr.tsv'), sep='\t', index=False))



if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument('--atlas', type=str, default='ROI')
    # parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=1)
    parser.add_argument('--demean', action='store_true',)
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--n_tessels', type=int, default=362, choices=[42, 162, 362, 642, 1002, 1442])

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')