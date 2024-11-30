"""
1) Load freq file once
2) Plot all freq data as scatter
3) Plot histogram of freq data
- Then loop:
    4) Load end of file
    5) Update plots
"""

from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

script_path = os.path.realpath(__file__)
base_dir = os.path.dirname(script_path)
base_dir = '/home/abuzarmahmood/projects/video_movement_frequency'
artifact_dir = os.path.join(base_dir, 'artifacts')

# Get all files in artifact_dir
freq_files = glob(os.path.join(artifact_dir, 'freq_data_device*.csv'))

##############################
# Load freq data

cols = ['frame_rate','freq','time','counter']
freq_data = [pd.read_csv(f, header=None,) for f in freq_files] 
# Update columns 
for i, freq in enumerate(freq_data):
    freq.columns = cols

# Create plots

scatter_list = []
line_list = []
hist_list = []
for i, freq in enumerate(freq_data):
    freq_vals = freq['freq'].values
    # Convert from Hz to RPM
    freq_vals = freq_vals * 60
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sc = ax[0].scatter(freq['time'], freq_vals) 
    ln = ax[0].plot(freq['time'], freq_vals)
    ax[0].set_title(f"Freq data for device {i}")
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (RPM)')
    # Rotate x-axis labels
    ax[0].tick_params(axis='x', rotation=45)
    hs = ax[1].hist(freq['freq'], bins=20, orientation='horizontal')
    ax[1].set_title(f"Histogram of freq data for device {i}")
    ax[1].set_xlabel('Count')
    scatter_list.append(sc)
    line_list.append(ln)
    hist_list.append(hs)
    plt.tight_layout()
    # plt.show()

# Load end of file
# Update plots
# Repeat

while True:
    freq_data = [pd.read_csv(f, header=None, skiprows=range(1, len(f))) for f in freq_files] 
    # Update columns 
    for i, freq in enumerate(freq_data):
        freq.columns = cols
    for i, freq in enumerate(freq_data):
        freq_vals = freq['freq'].values
        # Convert from Hz to RPM
        freq_vals = freq_vals * 60
        scatter_list[i].set_offsets(np.c_[freq['time'], freq_vals])
        line_list[i][0].set_ydata(freq_vals)
        # hist_list[i][0].set_height(freq['freq'])
        plt.draw()
