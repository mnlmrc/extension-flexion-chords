from imaging_pipelines.modeling import PcmRois
import globals as gl
import numpy as np
import os
import pickle
import time
import argparse
import PcmPy as pcm

def make_models():
    C = pcm.centering(8)

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

def main(args):

    if args.what == 'make_models':
        M = make_models()
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'models.p'), "wb")
        pickle.dump(M, f)

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


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[101])
    parser.add_argument('--atlas', type=str, default='ROI')
    # parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--n_tessels', type=int, default=362, choices=[42, 162, 362, 642, 1002, 1442])

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')