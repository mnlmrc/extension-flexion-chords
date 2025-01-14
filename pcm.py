import PcmPy as pcm

def calc_G(experiment, glm, day, sn):



def pcm():
    scaler = MinMaxScaler()

    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")
    reginfo[['cue', 'stimFinger']] = reginfo['name'].str.split(',', expand=True)
    reginfo.cue = reginfo.cue.str.replace(" ", "")
    reginfo.stimFinger = reginfo.stimFinger.fillna('nogo')
    reginfo1 = reginfo[reginfo['run'] == 1]

    Z_all = pcm.matrix.indicator(reginfo1.name)
    Z_cue = pcm.matrix.indicator(reginfo1.cue)
    Z_stimFinger = pcm.matrix.indicator(reginfo1.stimFinger)

    M_all, G_all = FixedModel('all', Z_all)
    M_cue, G_cue = FixedModel('cue', Z_cue)
    M_stimFinger, G_stimFinger = FixedModel('stimFinger', Z_stimFinger)
    I = np.eye(G_cue.shape[0])
    X = np.c_[Z_cue, Z_stimFinger]
    Io = I - X @ np.linalg.pinv(X) @ I

    Gc = np.stack([G_cue, G_stimFinger, Io @ Io.T], axis=0)

    # MC = pcm.model.ComponentModel('cue+stimFinger', [G_cue, G_stimFinger, Io@Io.T])
    MF = pcm.model.ModelFamily(Gc, comp_names=['cue', 'stimFinger', 'interaction'])

    betas = np.load(
        os.path.join(gl.baseDir, experiment, gl.glmDir + str(glm), f'subj{sn}', 'ROI.L.M1.beta.npy'))
    res = np.load(
        os.path.join(gl.baseDir, experiment, gl.glmDir + str(glm), f'subj{sn}', 'ROI.L.M1.res.npy'))
    betas_prewhitened = betas / np.sqrt(res)

    # betas_prewhitened_scaled = scaler.fit_transform(betas_prewhitened)

    noise_cov = np.diag(res)

    dataset = pcm.dataset.Dataset(
        betas_prewhitened,
        obs_descriptors={'cond_vec': reginfo.name,
                         'part_vec': reginfo.run})

    T, theta = pcm.inference.fit_model_individ(
        dataset, MF, fixed_effect='block', fit_scale=False, noise_cov='block', verbose=True)

    print(T)

    pcm.vis.model_plot(T.likelihood - MF.num_comp_per_m)