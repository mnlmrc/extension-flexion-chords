import os
import globals as gl
import numpy as np
import matplotlib.pyplot as plt

experiment = 'efc1'

# Parameters
time = np.linspace(0, 10, 1000)  # Simulate a 10-second duration
num_muscles = 5
color = 'darkred'

# Generate random EMG signals for 5 muscles with very pronounced and irregular modulation
np.random.seed(42)
irregular_modulation = np.abs(np.cumsum(np.random.randn(1000) * 0.1) + 3)

# Adding a sustained activation period for some muscles
emg_signals = []
for i in range(num_muscles):
    signal = np.abs(np.random.randn(1000) * (np.random.randint(1, 5)) * irregular_modulation)
    if i < 2:  # Make the first two muscles stay active for a few seconds
        signal[300:600] += np.random.randn(300) * 10  # Increase the amplitude for a sustained period
    emg_signals.append(signal)

# Calculate the average activation for each muscle
avg_activation = [np.mean(signal) for signal in emg_signals]

# Randomly shuffle the average activation levels across muscles
np.random.shuffle(avg_activation)

# Create a figure with subplots
fig, axs = plt.subplots(num_muscles, figsize=(5, 5)) #, gridspec_kw={'width_ratios': [3, 1]})

# Define a common y-axis limit for all subplots
common_ylim = (0, np.max(emg_signals) * 1.1)

# Plot the EMG signals and the average activation
for i in range(num_muscles):
    # Plot the random EMG signal with very pronounced and irregular modulation over time
    axs[i].plot(time, emg_signals[i], color=color)
    axs[i].set_xlim(0, 10)
    axs[i].set_ylim(common_ylim)

    # Set the title with much larger font size and black color
    axs[i].set_title(f'Muscle {i + 1}', fontsize=20, color='black')

    # Remove x and y ticks, labels, and spines
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)

    axs[i].grid(False)

    # # Plot the average activation as a much narrower horizontal bar
    # axs[i, 1].barh(0, avg_activation[i], color=color, height=.05)
    # axs[i, 1].set_xlim(0, max(avg_activation) * 1.1)  # Extend limit slightly for visibility
    # axs[i, 1].set_yticks([])  # Remove y-axis ticks
    # axs[i, 1].set_xticks([])  # Remove x-axis ticks
    # axs[i, 1].set_xlim(common_ylim)  # Ensure the bars start at zero
    #
    # # Remove spines from the bar plots
    # axs[i, 1].spines['top'].set_visible(False)
    # axs[i, 1].spines['right'].set_visible(False)
    # axs[i, 1].spines['bottom'].set_visible(False)
    # axs[i, 1].spines['left'].set_visible(False)

fig.supylabel('EMG activity', fontsize=24)
fig.supxlabel('time', fontsize=24)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'efc1_emg_sim.svg'), dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
