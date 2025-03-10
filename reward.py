import argparse

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import globals as gl
import os
import matplotlib

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def save_reward_data_obs():
    data = pd.read_csv('/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc2/metrics.tsv', sep='\t')

    # Compute RT+ET
    data['RT+ET'] = data['RT'] + data['ET']

    # Filter trained chords and exclude day 5
    data = data[(data['chord']=='trained') & (data['day']!=5)]
    data_avg = data.groupby(['day', 'subNum']).mean(numeric_only=True).reset_index()

    data_avg.to_csv('reward_data_obs.tsv', index=False, sep='\t')


def save_reward_data_pred():

    data_avg = pd.read_csv('reward_data_obs.tsv', sep='\t')

    x = data_avg['day'].values
    y = data_avg['RT+ET'].values

    popt, _ = curve_fit(exp_func, x, y, p0=(max(y), -0.1, min(y)))

    # Generate predicted curve
    xhat = np.arange(24) + 1
    yhat = exp_func(xhat, *popt)

    data_pred = pd.DataFrame({'day': xhat, 'RT+ET': yhat})

    data_pred.to_csv('reward_data_pred.tsv', index=False, sep='\t')

def plot_ref():

    data_pred = pd.read_csv('reward_data_pred.tsv', sep='\t')

    plt.plot(data_pred['day'], data_pred['RT+ET'] / max_performance,
             label="Predicted Performance", linestyle="--", color="black")

    # Add level lines
    for level, threshold in levels.items():
        plt.axhline(y=threshold[0], lw=3, label=level, color=threshold[1])

    plt.legend()
    plt.xlabel("Day")
    plt.xticks([1, 5, 10, 20, 24])
    plt.ylabel("Execution time (% ref subject)")
    plt.title("Performance Progression and Reward Levels")


def plot_current(path, day):

    data = pd.read_csv(path, sep='\t')
    data = data[data.trialPoint == 1]

    ET = (data['ET'] + data['RT']).mean()
    ET_norm = ET / (max_performance * 1000)

    if ET_norm > thresholds[0]:
        plt.title("Keep trying! Bronze level is not far!")
    elif (ET_norm > thresholds[1]) & (ET_norm < thresholds[0]):
        plt.title("Congrats! You reached BRONZE LEVEL!\n"
                  "Keep trying! Silver level is not far!")
    elif (ET_norm > thresholds[2]) & (ET_norm < thresholds[1]):
        plt.title("Congrats! You reached SILVER LEVEL!\n"
                  "Keep trying! Gold level is not far!")
    elif (ET_norm > thresholds[3]) & (ET_norm < thresholds[2]):
        plt.title("Congrats! You reached GOLD LEVEL!\n"
                  "Keep trying! Platinum level is not far!")
    elif (ET_norm > thresholds[4]) & (ET_norm < thresholds[3]):
        plt.title("Congrats! You reached PLATINUM LEVEL!\n"
                  "Keep trying! Diamond level is not far!")
    elif (ET_norm > thresholds[5]) & (ET_norm < thresholds[4]):
        plt.title("Congrats! You reached DIAMOND LEVEL!\n"
                  "A little more and you'll be an ALIEN!")
    elif ET_norm < thresholds[6]:
        plt.title("Honestly! We don't see this every day!\n"
                  "you are an ALIEN!!!")

    print(f'Execution time: {ET} ms')

    plt.scatter(day, ET_norm, color='red', s=50)


if __name__ == '__main__':

    num_levels = 6  # Number of levels
    start = 0.6  # * max_performance  # Initial threshold
    end = 0.2  # * max_performance  # Final threshold
    exponents = np.linspace(0, 1, num_levels)  # Exponential spacing
    decay_rate = 4

    hrate = np.linspace(15.5, 17.5, 6)

    thresholds = start - (start - end) * (1 - np.exp(-decay_rate * exponents)) / (1 - np.exp(-decay_rate))

    # Compute exponentially decreasing thresholds
    # thresholds = start * (end / start) ** exponents #[::-1]  # Reverse order

    levels = {
        f"Bronze (${hrate[0]}/hour)": (thresholds[0], 'tan'),
        f"Silver (${hrate[1]}/hour)": (thresholds[1], 'silver'),
        f"Gold (${hrate[2]}/hour)": (thresholds[2], 'gold'),
        f"Platinum (${hrate[3]}/hour)": (thresholds[3], 'gainsboro'),
        f"Diamond (${hrate[4]}/hour)": (thresholds[4], 'lightsteelblue'),
        f"Alien (${hrate[5]}/hour)": (thresholds[5], 'green')
    }

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--day', type=int, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--ref', type=float, default=3.0)

    args = parser.parse_args()

    max_performance = args.ref

    if args.what == 'save_obs':
        save_reward_data_obs()
    if args.what == 'save_pred':
        save_reward_data_pred()
    if args.what == 'plot_ref':
        plot_ref()
        # plt.show()
    if args.what == 'plot_current':
        plot_ref()

        path = os.path.join(gl.baseDir, 'efc4', 'pilot', f'day{args.day}', f'subj{args.sn}', f'efc4_{args.sn}.dat')

        plot_current(path, day=args.day)

        plt.show()

