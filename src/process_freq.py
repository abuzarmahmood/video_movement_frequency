"""
Process and visualize frequency data with interactive y-axis control

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
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pprint import pprint as pp
from playsound import playsound
import os

# Create the main window
root = tk.Tk()
root.title("Frequency Visualization")

# Create frame for controls
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X)

# Create input fields for y-axis limits
tk.Label(control_frame, text="Y-min:").pack(side=tk.LEFT)
y_min_entry = tk.Entry(control_frame, width=10)
y_min_entry.pack(side=tk.LEFT)
y_min_entry.insert(0, "0")

tk.Label(control_frame, text="Y-max:").pack(side=tk.LEFT)
y_max_entry = tk.Entry(control_frame, width=10)
y_max_entry.pack(side=tk.LEFT)
y_max_entry.insert(0, "1000")

# Add time window control
tk.Label(control_frame, text="Time Window (min):").pack(side=tk.LEFT)
time_window_entry = tk.Entry(control_frame, width=10)
time_window_entry.pack(side=tk.LEFT)
time_window_entry.insert(0, "5")  # Default 5 minutes

# Add median filter length control
tk.Label(control_frame, text="Median Filter Length:").pack(side=tk.LEFT)
median_filter_entry = tk.Entry(control_frame, width=10)
median_filter_entry.pack(side=tk.LEFT)
median_filter_entry.insert(0, "5")  # Default 5 samples

# Add frequency bounds controls
tk.Label(control_frame, text="Min Freq (RPM):").pack(side=tk.LEFT)
min_freq_entry = tk.Entry(control_frame, width=10)
min_freq_entry.pack(side=tk.LEFT)
min_freq_entry.insert(0, "0")

tk.Label(control_frame, text="Max Freq (RPM):").pack(side=tk.LEFT)
max_freq_entry = tk.Entry(control_frame, width=10)
max_freq_entry.pack(side=tk.LEFT)
max_freq_entry.insert(0, "3000")

def play_warning():
    """Play warning sound"""
    sound_file = os.path.join(os.path.dirname(__file__), "warning.wav")
    playsound(sound_file, block=False)

def apply_median_filter(data, window_length):
    """Apply median filter to data"""
    from scipy.signal import medfilt
    return medfilt(data, kernel_size=int(window_length))

def validate_numeric_input(value, min_val=None, max_val=None, param_name="Parameter"):
    """Validate numeric input with optional range checking"""
    try:
        num_val = float(value)
        if min_val is not None and num_val < min_val:
            raise ValueError(f"{param_name} must be greater than {min_val}")
        if max_val is not None and num_val > max_val:
            raise ValueError(f"{param_name} must be less than {max_val}")
        return True, num_val
    except ValueError as e:
        print(f"Invalid {param_name}: {str(e)}")
        return False, None

def apply_parameters():
    """Apply all parameter changes"""
    # Validate y-axis limits
    valid_ymin, ymin = validate_numeric_input(y_min_entry.get(), param_name="Y-min")
    valid_ymax, ymax = validate_numeric_input(y_max_entry.get(), param_name="Y-max")
    valid_time, time_window = validate_numeric_input(
        time_window_entry.get(), min_val=0.1, max_val=60, param_name="Time window"
    )
    valid_filter, filter_length = validate_numeric_input(
        median_filter_entry.get(), min_val=1, max_val=1000, param_name="Filter length"
    )
    
    # Validate frequency bounds
    valid_min_freq, min_freq = validate_numeric_input(
        min_freq_entry.get(), min_val=0, param_name="Min frequency"
    )
    valid_max_freq, max_freq = validate_numeric_input(
        max_freq_entry.get(), min_val=0, param_name="Max frequency"
    )
        
    if valid_ymin and valid_ymax and valid_time and valid_filter and valid_min_freq and valid_max_freq:
        if ymin >= ymax:
            print("Y-min must be less than Y-max")
            return

        # Apply y-axis limits to line plots
        for ln in np.array(line_list).flatten():
            ln.axes.set_ylim(ymin, ymax)
        
        # Apply y-axis limits to histograms
        for hist_pair in hist_list:
            for hist_ax in hist_pair:
                hist_ax.set_ylim(ymin, ymax)
        
        print("Parameters applied successfully")
    else:
        print("Parameter validation failed")

# Create apply button
apply_button = tk.Button(control_frame, text="Apply Parameters", command=apply_parameters)
apply_button.pack(side=tk.LEFT, padx=5)

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
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # Format time values to be more readable
    time_vals = freq['time'].astype('datetime64[s]').values
    # Full time series plot
    sc = ax[0,0].scatter(time_vals, freq_vals) 
    ln = ax[0,0].plot(time_vals, freq_vals)
    ax[0,0].set_title(f"Full freq data for device {i}")
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('Frequency (RPM)')
    ax[0,0].tick_params(axis='x', rotation=45)
    ax[0,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0,0].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Recent time series plot
    sc_recent = ax[0,1].scatter(time_vals, freq_vals)
    ln_recent = ax[0,1].plot(time_vals, freq_vals)
    ax[0,1].set_title(f"Recent freq data for device {i}")
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('Frequency (RPM)')
    ax[0,1].tick_params(axis='x', rotation=45)
    ax[0,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0,1].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Full histogram
    ax[1,0].hist(freq_vals, bins=20, orientation='horizontal')
    ax[1,0].set_title(f"Full histogram for device {i}")
    ax[1,0].set_xlabel('Count')

    # Recent histogram
    ax[1,1].hist(freq_vals, bins=20, orientation='horizontal')
    ax[1,1].set_title(f"Recent histogram for device {i}")
    ax[1,1].set_xlabel('Count')
    scatter_list.append((sc, sc_recent))
    line_list.append([ln, ln_recent])
    hist_list.append((ax[1,0], ax[1,1]))  # Store both histogram axes
    plt.tight_layout()

# Load end of file and update plots continuously
while True:
    for i, f in enumerate(freq_files):
        # Read the entire file each time to get full history
        freq = pd.read_csv(f, header=None)
        freq.columns = cols
        
        time_vals = freq['time'].astype('datetime64[s]').values
        freq_vals = freq['freq'].values * 60  # Convert to RPM
        
        # Get recent data window - validate input
        _, time_val = validate_numeric_input(
            time_window_entry.get(), min_val=0.1, max_val=60, param_name="Time window"
        )
        if time_val is None:
            time_val = 5  # Default to 5 minutes if invalid
        time_window = time_val * 60  # Convert minutes to seconds
        # Convert time_window to datetime64 
        time_window = np.timedelta64(int(time_window), 's')
        recent_mask = (time_vals.max() - time_vals) <= time_window
        recent_times = time_vals[recent_mask]
        recent_freqs = freq_vals[recent_mask]

        # Update full time series
        # Convert time_vals to same dtype as freq_vals
        time_vals = np.float64(time_vals)
        scatter_list[i][0].set_offsets(np.c_[time_vals, freq_vals])
        
        # Apply median filter to full series
        _, filter_length = validate_numeric_input(
            median_filter_entry.get(), min_val=1, max_val=1000, param_name="Filter length"
        )
        if filter_length is None:
            filter_length = 5
        filtered_vals = apply_median_filter(freq_vals, filter_length)
        
        # Get frequency bounds
        _, min_freq = validate_numeric_input(min_freq_entry.get(), min_val=0, param_name="Min frequency")
        _, max_freq = validate_numeric_input(max_freq_entry.get(), min_val=0, param_name="Max frequency")
        
        # Check if filtered values exceed bounds and play warning if they do
        if min_freq is not None and max_freq is not None:
            if any((filtered_vals < min_freq) | (filtered_vals > max_freq)):
                play_warning()
        
        # Plot both raw and filtered data
        line_list[i][0][0].set_data(time_vals, freq_vals)
        
        # Add/update frequency bound lines
        ax = line_list[i][0][0].axes
        # Remove old lines if they exist
        lines = ax.get_lines()
        lines = [l for l in lines if not l.get_label().startswith('Bound')]
        # ax.lines = lines
        
        if min_freq is not None:
            ax.axhline(y=min_freq, color='r', linestyle='--', label='Bound Min')
        if max_freq is not None:
            ax.axhline(y=max_freq, color='r', linestyle='--', label='Bound Max')
        if len(line_list[i][0]) > 1:
            line_list[i][0][1].set_data(time_vals, filtered_vals)
        else:
            filtered_line = line_list[i][0][0].axes.plot(time_vals, filtered_vals, 'r-', label='Filtered')[0]
            line_list[i][0] = [line_list[i][0][0], filtered_line]
            line_list[i][0][0].axes.legend()
        line_list[i][0][0].axes.xaxis.set_major_locator(plt.MaxNLocator(6))
        line_list[i][0][0].axes.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        # Update recent time series
        recent_times = np.float64(recent_times)
        scatter_list[i][1].set_offsets(np.c_[recent_times, recent_freqs])
        
        # Apply median filter to recent series
        filtered_recent = apply_median_filter(recent_freqs, filter_length)
        
        # Plot both raw and filtered recent data
        line_list[i][1][0].set_data(recent_times, recent_freqs)
        
        # Add/update frequency bound lines for recent plot
        ax = line_list[i][1][0].axes
        # Remove old lines if they exist
        lines = ax.get_lines()
        lines = [l for l in lines if not l.get_label().startswith('Bound')]
        # ax.lines = lines
        
        if min_freq is not None:
            ax.axhline(y=min_freq, color='r', linestyle='--', label='Bound Min')
        if max_freq is not None:
            ax.axhline(y=max_freq, color='r', linestyle='--', label='Bound Max')
        if len(line_list[i][1]) > 1:
            line_list[i][1][1].set_data(recent_times, filtered_recent)
        else:
            filtered_line = line_list[i][1][0].axes.plot(recent_times, filtered_recent, 'r-', label='Filtered')[0]
            line_list[i][1] = [line_list[i][1][0], filtered_line]
            line_list[i][1][0].axes.legend()
        line_list[i][1][0].axes.xaxis.set_major_locator(plt.MaxNLocator(6))
        line_list[i][1][0].axes.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        # Update full histogram
        hist_list[i][0].clear()
        hist_list[i][0].hist(freq_vals, bins=20, orientation='horizontal')
        hist_list[i][0].set_title(f"Full histogram for device {i}")
        hist_list[i][0].set_xlabel('Count')

        # Update recent histogram
        hist_list[i][1].clear()
        hist_list[i][1].hist(recent_freqs, bins=20, orientation='horizontal')
        hist_list[i][1].set_title(f"Recent histogram for device {i}")
        hist_list[i][1].set_xlabel('Count')
        
        # Update axis limits
        for ln in line_list[i]:
            ln[0].axes.relim()
            ln[0].axes.autoscale_view()
        
    plt.pause(0.1)  # Add small delay and handle GUI events
    root.update()  # Update the tkinter window
