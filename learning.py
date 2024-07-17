import globals as gl
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def reliability_var(Y, subj_vec, part_vec, cond_vec=None, centered=True):
    if cond_vec is None:
        cond_vec = np.ones_like(subj_vec)

    subjects = np.unique(subj_vec)
    partitions = np.unique(part_vec)
    conds = np.unique(cond_vec)

    v_gs = np.zeros(len(conds))
    v_gse = np.zeros(len(conds))

    subj_data = {}

    for k, cond in enumerate(conds):
        for i, subj in enumerate(subjects):
            A = np.array([])
            for j, part in enumerate(partitions):
                part_data = Y[(subj_vec == subj) & (part_vec == part) & (cond_vec == cond)]

                if centered:
                    part_data -= part_data.mean(axis=0)

                A = np.concatenate((A, part_data))

            subj_data[(i, k)] = np.hstack(A)

            B = np.dot(A, A.T)
            N = len(A)

            v_gse[k] += np.trace(B) / N / len(subjects)
            v_gs[k] += (np.sum(B) - np.trace(B)) / N / (N - 1) / len(subjects)

    v_g = np.zeros(len(conds))

    N = len(subjects)
    for k in range(len(conds)):
        for i in range(N - 1):
            for j in range(i + 1, N):
                B = np.dot(subj_data[(i, k)].T, subj_data[(j, k)])
                v_g[k] += np.sum(B) / B.shape[0] ** 2 / (N * (N - 1) / 2)

    return v_g, v_gs, v_gse


# Example data (replace with your actual data)
data = pd.DataFrame({
    'd0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'd1': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'd2': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'd3': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'd4': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'chord': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
    'participants_id': ['P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P4', 'P4', 'P5', 'P5']
})

measurements = data[['d0', 'd1', 'd2', 'd3', 'd4']].values
subj_vec = pd.Categorical(data['participants_id']).codes
part_vec = np.tile([1, 2], len(subj_vec) // 2)
cond_vec = pd.Categorical(data['chord']).codes

v_g, v_gs, v_gse = reliability_var(measurements, subj_vec, part_vec, cond_vec)

# Display the variance components
print("Variance Components:")
for k in range(len(np.unique(cond_vec))):
    print(f"Condition {k}:")
    print(f"  v_g: {v_g[k]:.4f}")
    print(f"  v_gs: {v_gs[k]:.4f}")
    print(f"  v_gse: {v_gse[k]:.4f}")
    print("\n")


def calc_avg(filename, columns=None):
    """
        Computes the average MD for each chordID in the given dataframe.

        Parameters:
        data (pd.DataFrame): The input dataframe containing 'chordID' and 'MD' columns.

        Returns:
        pd.DataFrame: A dataframe with 'chordID' and the corresponding average 'MD'.
        """
    # Group by 'chordID' and compute the mean of 'MD'
    data = pd.read_csv(filename)
    avg = data.groupby('chordID')[columns].mean().reset_index()
    # md.rename(columns={'MD': 'average_MD'}, inplace=True)

    return avg


def make_tab(
        experiment='efc2',
        participant_id='subj100',
        session='testing',
        day='1'):
    print(f"experiment:{experiment}, "
          f"participant_id:{participant_id}, "
          f"session:{session}, "
          f"day:{day}, ")

    path = os.path.join(gl.baseDir, experiment, session, f"day{day}")

    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    filename = os.path.join(path, f"{experiment}_{sn}.csv")

    # Calculate average values
    df = calc_avg(filename, columns=['MD', 'RT', 'angle', 'jerk', 'sine']
                                    + [f'PC{i}' for i in range(5)]
                                    + [f'd{i}' for i in range(5)])

    # Add an empty 'set' column
    df['chord'] = None

    # Update the 'set' column for trained and untrained chordIDs
    df.loc[df['chordID'].isin(gl.trained), 'chord'] = 'trained'
    df.loc[df['chordID'].isin(gl.untrained), 'chord'] = 'untrained'

    df['session'] = session
    df['participant_id'] = participant_id
    df['day'] = day

    return df


if __name__ == "__main__":

    experiment = 'efc2'
    participants = [
        'subj100',
        'subj101',
        'subj102',
        'subj103',
        'subj104',
        # 'subj105',
        'subj106',
        'subj107'
    ]
    sessions = ['testing',
                'training']
    days = ['1', '2', '3', '4', '5']

    df = pd.DataFrame()

    for participant_id in participants:
        for day in days:
            if day == '1' or day == '5':
                session = 'testing'
            else:
                session = 'training'

            df = pd.concat([df, make_tab(experiment,
                                         participant_id,
                                         session,
                                         day)])

    df.to_csv(os.path.join(gl.baseDir, experiment, 'results.csv'))

    _, axs = plt.subplots()
    sns.pointplot(df, ax=axs, x='day', y='MD', hue='chord', dodge=True, linestyle='none', errorbar='se',
                  palette=['blue', 'red'])

    _, axs = plt.subplots()
    sns.pointplot(df, ax=axs, x='day', y='RT', hue='chord', dodge=True, linestyle='none', errorbar='se',
                  palette=['blue', 'red'])

    _, axs = plt.subplots()
    sns.pointplot(df, ax=axs, x='day', y='jerk', hue='chord', dodge=True, linestyle='none', errorbar='se',
                  palette=['blue', 'red'])

    _, axs = plt.subplots()
    sns.pointplot(df, ax=axs, x='day', y='sine', hue='chord', dodge=True, linestyle='none', errorbar='se',
                  palette=['blue', 'red'])

    _, axs = plt.subplots()
    df_melted = df.melt(id_vars=['day', 'chord'],
                        value_vars=[f'PC{i}' for i in range(5)],
                        var_name='PCs', value_name='Explained')
    sns.pointplot(df_melted[df_melted['chord'] == 'untrained'], x='PCs', y='Explained', hue='day',
                  errorbar='se', palette='Blues')
    sns.pointplot(df_melted[df_melted['chord'] == 'trained'], x='PCs', y='Explained', hue='day',
                  errorbar='se', palette='Reds')
    # axs.set_yscale('log')

    plt.show()
