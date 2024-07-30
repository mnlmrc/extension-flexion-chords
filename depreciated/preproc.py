# import argparse
# import os
#
# import numpy as np
#
# import globals as gl
# import pandas as pd
# from fetch import load_mov_block
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
#
# def calc_md(X):
#     N, m = X.shape
#     F1 = X[0]
#     FN = X[-1] - F1  # Shift the end point
#
#     shifted_matrix = X - F1  # Shift all points
#
#     d = list()
#
#     for t in range(1, N - 1):
#         Ft = shifted_matrix[t]
#
#         # Project Ft onto the ideal straight line
#         proj = np.dot(Ft, FN) / np.dot(FN, FN) * FN
#
#         # Calculate the Euclidean distance
#         d.append(np.linalg.norm(Ft - proj))
#
#     d = np.array(d)
#     MD = d.mean()
#
#     return MD
#
#
# def calc_pca(X):
#     scaler = StandardScaler()
#
#     X_scaled = scaler.fit_transform(X)
#     pca = PCA()
#     pca = pca.fit(X_scaled)
#
#     loadings = pca.components_
#     explained = pca.explained_variance_ratio_
#     principal_components = pca.fit_transform(X_scaled)
#
#     return explained, loadings, principal_components
#
#
# def calc_jerk(X, fsample=gl.fsample):
#     # Ensure the positions and times are numpy arrays
#     pos = np.array(X)
#
#     vel = np.gradient(pos, 1 / fsample, axis=0)
#     acc = np.gradient(vel, 1 / fsample, axis=0)
#     jerk = np.gradient(acc, 1 / fsample, axis=0)
#
#     av_squared_jerk = np.linalg.norm(jerk).mean()
#
#     return av_squared_jerk
#
#
# def calc_dist(X, latency=.05, fsample=gl.fsample):
#     index = int(latency * fsample)
#     x = X[:index].mean(axis=0)
#
#     L = x[-1] - x[0]
#     L_norm = L / np.linalg.norm(L)
#     x_norm = x / np.linalg.norm(x)
#
#     d = x_norm - L_norm
#
#     return d
#
#
# def calc_metrics(mov, tr):
#     force = mov[tr][:, gl.diffCols]
#     c = np.any(np.abs(force) > gl.fthresh, axis=1)
#
#     # extrapolate force from RT to start of hold time
#     force = force[c][:int(-gl.hold_time * gl.fsample)]
#
#     rt = np.argmax(c) / gl.fsample
#     md = calc_md(force)
#     exp, loadings, _ = calc_pca(force)
#
#     L = force[-1] - force[0]
#     L_norm = L / np.linalg.norm(L)
#     pc1 = loadings[0]
#     pc1_norm = pc1 / np.linalg.norm(pc1)
#     proj = np.dot(pc1_norm, L_norm)
#     angle = np.arccos(proj)
#     sine = np.sin(angle)
#     jerk = calc_jerk(force)
#     dist = calc_dist(force)
#
#     metrics = {
#         'RT': rt,
#         'MD': md,
#         'PC': exp,
#         'angle': angle,
#         'sine': sine,
#         'jerk': jerk,
#         'dist': dist
#     }
#
#     return metrics
#
#
# def preprocessing(experiment="efc2", participant_id="subj100", session="testing", day="1"):
#     # define path
#     path = os.path.join(gl.baseDir, experiment, session, f"day{day}")
#
#     # extract subject number
#     sn = int(''.join([c for c in participant_id if c.isdigit()]))
#
#     # load dat file
#     dat = pd.read_csv(os.path.join(path, f"{experiment}_{sn}.dat"), sep="\t")
#
#     # init dict
#     results = {
#         'MD': [], 'RT': [], 'PC': [], 'angle': [],
#         'jerk': [], 'sine': [], 'dist': [], 'repetition': []
#     }
#
#     for bl in range(gl.nblocks):
#         block = f'{bl + 1:02d}'
#         print(f"experiment:{experiment}, "
#               f"participant_id:{participant_id}, "
#               f"session:{session}, "
#               f"day:{day}, "
#               f"block:{block}")
#         filename = os.path.join(path, f'{experiment}_{sn}_{block}.mov')
#         mov = load_mov_block(filename)
#
#         dat_tmp = dat[dat.BN == bl + 1].reset_index()
#         rep = 1
#
#         ntrial = len(mov)
#
#         for tr in range(ntrial):
#             if tr == 0 or dat_tmp.iloc[tr].chordID != dat_tmp.iloc[tr - 1].chordID:
#                 rep = 1
#             else:
#                 rep += 1
#             results['repetition'].append(rep)
#
#             if dat_tmp.iloc[tr].trialPoint:
#
#                 metrics = calc_metrics(mov, tr)
#
#                 results['RT'].append(metrics['RT'])
#                 results['MD'].append(metrics['MD'])
#                 results['PC'].append(metrics['PC'])
#                 results['angle'].append(metrics['angle'])
#                 results['sine'].append(metrics['sine'])
#                 results['jerk'].append(metrics['jerk'])
#                 results['dist'].append(metrics['dist'])
#
#             else:
#
#                 results['RT'].append(None)
#                 results['MD'].append(None)
#                 results['PC'].append(None)
#                 results['angle'].append(None)
#                 results['sine'].append(None)
#                 results['jerk'].append(None)
#                 results['dist'].append(None)
#
#     df = dat[['BN', 'TN', 'subNum', 'chordID', 'trialPoint']].copy()
#     for key, values in results.items():
#         if key in ['PC', 'dist']:
#             for i in range(5):
#                 df[f'{key}{i}'] = [val[i] if val is not None else None for val in values]
#         else:
#             df[key] = values
#
#     return df
#
#
# if __name__ == "__main__":
#
#     experiment = 'efc2'
#     participants = [
#         'subj100',
#         'subj101',
#         'subj102',
#         'subj103',
#         'subj104',
#         # 'subj105',
#         'subj106',
#         'subj107'
#     ]
#     sessions = ['testing',
#                 'training']
#     days = ['1', '2', '3', '4', '5']
#
#     df = pd.DataFrame()
#
#     for participant_id in participants:
#         for day in days:
#             if day == '1' or day == '5':
#                 session = 'testing'
#             else:
#                 session = 'training'
#
#             df_tmp = preprocessing(experiment=experiment,
#                                    participant_id=participant_id,
#                                    session=session,
#                                    day=day)
#             df_tmp['session'] = session
#             df_tmp['participant_id'] = participant_id
#             df_tmp['day'] = day
#             df_tmp.loc[df_tmp['chordID'].isin(gl.trained), 'chord'] = 'trained'
#             df_tmp.loc[df_tmp['chordID'].isin(gl.untrained), 'chord'] = 'untrained'
#             df = pd.concat([df, df_tmp])
#
#     df.to_csv(os.path.join(gl.baseDir, experiment, 'results.csv'))


# case 'EMG:recon_chord2nat':
#
#             path = os.path.join(gl.baseDir, experiment, gl.natDir)
#
#             scaler = MinMaxScaler()
#
#             Hp_dict = {key: [] for key in participant_id}
#             H_nat_dict = {key: [] for key in participant_id}
#             reconerr_dict = {key: [] for key in participant_id}
#
#             for p in participant_id:
#
#                 sn = int(''.join([c for c in p if c.isdigit()]))
#                 filename = f'natChord_subj{"{:02}".format(sn)}_emg_natural_whole_sampled.mat'
#                 mat = scipy.io.loadmat(os.path.join(path, filename))
#                 emg_nat = [np.array(matrix[0]) for matrix in mat['emg_natural_dist']['dist'][0][0]]
#
#                 M_chords = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.chordDir, 'natChord_chord.tsv'),
#                                        sep='\t')
#                 M_chords = M_chords[M_chords['sn'] == sn]
#                 chords = list(M_chords['chordID'])
#                 M_chords = M_chords[[f'emg_hold_avg_e{e + 1}' for e in range(5)] +
#                                     [f'emg_hold_avg_f{f + 1}' for f in range(5)]].to_numpy()
#                 M_chords = scaler.fit_transform(M_chords)
#
#                 for M_nat in emg_nat:
#                     M_nat = scaler.fit_transform(M_nat)
#                     W_nat, H_nat, r2_nat, _, _ = iterative_nnmf(M_nat)
#
#                     Hp = optimize_H(W_nat, M_nat, M_chords)
#
#                     assert_selected_rows_belong(M_chords, Hp)
#
#                     reconerr = calc_reconerr(W_nat, Hp, M_nat)
#
#                     Hp_dict[p].append(Hp)
#                     H_nat_dict[p].append(H_nat)
#                     reconerr_dict[p].append(reconerr)
#
#             return Hp_dict, H_nat_dict, reconerr_dict
