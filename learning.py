import globals as gl
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    df = calc_avg(filename, columns=['MD', 'RT', 'angle', 'jerk'])

    # Add an empty 'set' column
    df['set'] = None

    # Update the 'set' column for trained and untrained chordIDs
    df.loc[df['chordID'].isin(gl.trained), 'set'] = 'trained'
    df.loc[df['chordID'].isin(gl.untrained), 'set'] = 'untrained'

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

    sns.lineplot(df, x='day', y='jerk', hue='set')

    plt.show()


