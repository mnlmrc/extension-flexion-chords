import pandas as pd
import scipy


def load_nat_emg(file_path):
    # Load the .mat file
    mat = scipy.io.loadmat(file_path)

    # Extract the 'dist' cell array from 'emg_natural_dist'
    emg_nat = mat['emg_natural_dist']
    emg_nat = emg_nat['dist'][0, 0]

    emg_nat_list = []
    for e in emg_nat:
        emg_nat_list.append(e[0])

    return emg_nat_list

def calc_success(X):

    success = X.groupby(['subNum', 'chordID', 'day', 'chord'])['trialPoint'].mean().reset_index()
    success.rename(columns={'trialPoint': 'success'}, inplace=True)

    success.sort_values(by='chord', inplace=True)

    return success

def calc_avg(X, columns=None, by=None):
    """
        Computes the average value of variables in <columns> grouped by <by>.

        Parameters:
        data (pd.DataFrame): The input dataframe.

        Returns:
        pd.DataFrame: A dataframe with averaged values.
        """

    if isinstance(columns, str):
        columns = [columns]

    # Group by 'chordID' and compute the mean of 'MD'
    if X is str:
        data = pd.read_csv(X)
    elif isinstance(X, pd.DataFrame):
        data = X
    else:
        data = None

    columns = {col: 'mean' for col in columns}
    avg = data.groupby(by).agg(columns).reset_index()
    # md.rename(columns={'MD': 'average_MD'}, inplace=True)

    return avg
