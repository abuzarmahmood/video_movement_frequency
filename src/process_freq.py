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
    # Format time values to be more readable
    time_vals = freq['time']
    sc = ax[0].scatter(time_vals, freq_vals) 
    ln = ax[0].plot(time_vals, freq_vals)
    ax[0].set_title(f"Freq data for device {i}")
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (RPM)')
    # Format x-axis ticks
    ax[0].tick_params(axis='x', rotation=45)
    # Only show some tick labels to avoid overcrowding
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
    # Format numbers with fewer decimal places
    ax[0].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax[1].hist(freq_vals, bins=20, orientation='horizontal')
    ax[1].set_title(f"Histogram of freq data for device {i}")
    ax[1].set_xlabel('Count')
    scatter_list.append(sc)
    line_list.append(ln)
    hist_list.append(ax[1])  # Store the axis instead of histogram
    plt.tight_layout()

# Load end of file and update plots continuously
while True:
    for i, f in enumerate(freq_files):
        # Read the entire file each time to get full history
        freq = pd.read_csv(f, header=None)
        freq.columns = cols
        
        freq_vals = freq['freq'].values * 60  # Convert to RPM
        
        # Update time series
        time_vals = freq['time']
        scatter_list[i].set_offsets(np.c_[time_vals, freq_vals])
        line_list[i][0].set_data(time_vals, freq_vals)
        # Keep x-axis formatting consistent
        line_list[i][0].axes.xaxis.set_major_locator(plt.MaxNLocator(6))
        line_list[i][0].axes.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        # Update histogram
        hist_list[i].clear()  # Clear previous histogram
        hist_list[i].hist(freq_vals, bins=20, orientation='horizontal')
        hist_list[i].set_title(f"Histogram of freq data for device {i}")
        hist_list[i].set_xlabel('Count')
        
        # Update axis limits for time series
        line_list[i][0].axes.relim()
        line_list[i][0].axes.autoscale_view()
        
    plt.pause(0.1)  # Add small delay and handle GUI events
