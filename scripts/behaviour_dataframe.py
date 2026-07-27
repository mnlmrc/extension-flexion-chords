import EFC_learningfMRI.globals as gl
import pandas as pd
import os

if __name__ == '__main__':
    sns       = gl.participants
    df_pooled = pd.DataFrame()
    for sn in sns:
        for sess in range(24):
            print(f'doing participant {sn}, day {sess + 1}')
            df = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, f'day{sess+1}', f'efc4_{sn}_single_trial.tsv'), sep = '\t',)
            df_pooled = pd.concat([df_pooled, df])

    df_pooled.to_csv(os.path.join(gl.baseDir, gl.behavDir, f'behaviour.trial.tsv'), sep='\t', index=False)

    df_sess = df_pooled.groupby(['subNum', 'session', 'chord']).mean(numeric_only=True).reset_index()
    df_sess.to_csv(os.path.join(gl.baseDir, gl.behavDir, f'behaviour.session.tsv'), sep='\t', index=False)

    df_corr = df_pooled[df_pooled.trialPoint == 1].reset_index()
    df_sess_corr = df_corr.groupby(['subNum', 'session', 'chord']).mean(numeric_only=True).reset_index()
    df_sess_corr.to_csv(os.path.join(gl.baseDir, gl.behavDir, f'behaviour.session.success.tsv'), sep='\t', index=False)

    df_sess_corr_rep = df_corr.groupby(['subNum', 'session', 'chord', 'Repetition'], observed=True).mean(
        numeric_only=True).reset_index()
    df_sess_corr_rep.to_csv(os.path.join(gl.baseDir, gl.behavDir, f'behaviour.session.success.repetition.tsv'),
                             sep='\t', index=False)

    df_fabs = df_corr.melt(id_vars=['subNum', 'chordID', 'chord', 'TN', 'BN', 'session', 'Repetition', 'MD', 'ET'],
                             value_vars=['thumb_abs', 'index_abs', 'middle_abs', 'ring_abs', 'pinkie_abs'],
                             var_name='finger', value_name='force_abs')
    df_fder = df_corr.melt(id_vars=['subNum', 'chordID', 'chord', 'TN', 'BN', 'session', 'Repetition'],
                             value_vars=['thumb_der', 'index_der', 'middle_der', 'ring_der', 'pinkie_der'],
                             var_name='finger', value_name='force_der')
    df_fder_peak = df_corr.melt(id_vars=['subNum', 'chordID', 'chord', 'TN', 'BN', 'session', 'Repetition'],
                           value_vars=['thumb_der_peak', 'index_der_peak', 'middle_der_peak', 'ring_der_peak',
                                       'pinkie_der_peak'], var_name='finger', value_name='force_der_peak')
    df_fder_t2peak = df_corr.melt(id_vars=['subNum', 'chordID', 'chord', 'TN', 'BN', 'session', 'Repetition'],
                           value_vars=['thumb_der_t2peak', 'index_der_t2peak', 'middle_der_t2peak', 'ring_der_t2peak',
                                       'pinkie_der_t2peak'], var_name='finger', value_name='force_der_t2peak')

    df_force = df_fabs.copy()
    df_force['force_der'] = df_fder.force_der.to_numpy()
    df_force['force_der_peak'] = df_fder_peak.force_der_peak.to_numpy()
    df_force['force_der_t2peak'] = df_fder_t2peak.force_der_t2peak.to_numpy()
    df_force = df_force.groupby(['subNum', 'chordID', 'chord', 'TN', 'BN', 'session', 'Repetition']).mean(numeric_only=True).reset_index()
    df_force.to_csv(os.path.join(gl.baseDir, gl.behavDir, f'force.success.tsv'), sep='\t', index=False)

    df_sess_force = df_force.groupby(['subNum', 'session', 'chord']).mean(numeric_only=True).reset_index()
    df_sess_force.to_csv(os.path.join(gl.baseDir, gl.behavDir, f'force.session.success.tsv'), sep='\t', index=False)

    df_rep_fabs = df_fabs.groupby(['subNum', 'session', 'chord', 'Repetition'], observed=True).mean(
        numeric_only=True).reset_index()
    df_rep_fder = df_fder.groupby(['subNum', 'session', 'chord', 'Repetition'], observed=True).mean(
        numeric_only=True).reset_index()
    df_rep_fder_peak = df_fder_peak.groupby(['subNum', 'session', 'chord', 'Repetition'], observed=True).mean(
        numeric_only=True).reset_index()
    df_rep_force = df_rep_fabs.copy()
    df_rep_force['force_der'] = df_rep_fder.force_der.to_numpy()
    df_rep_force['force_der_peak'] = df_rep_fder_peak.force_der_peak.to_numpy()
    df_rep_force.to_csv(os.path.join(gl.baseDir, gl.behavDir, 'force.session.success.repetition.tsv'), sep='\t',
                        index=False)

