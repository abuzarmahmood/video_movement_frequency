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
from matplotlib.dates import DateFormatter
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pprint import pprint as pp
from pydub import AudioSegment
from pydub.playback import play
import os
import threading
import time

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
y_max_entry.insert(0, "100")

# Add time window control
tk.Label(control_frame, text="Time Window (min):").pack(side=tk.LEFT)
time_window_entry = tk.Entry(control_frame, width=10)
time_window_entry.pack(side=tk.LEFT)
time_window_entry.insert(0, "5")  # Default 5 minutes

# Add filter controls
tk.Label(control_frame, text="Filter Length:").pack(side=tk.LEFT)
median_filter_entry = tk.Entry(control_frame, width=10)
median_filter_entry.pack(side=tk.LEFT)
median_filter_entry.insert(0, "5")  # Default 5 samples

# Add checkbox for mean/median selection
use_mean_var = tk.BooleanVar()
use_mean_checkbox = tk.Checkbutton(control_frame, text="Use Mean Filter", variable=use_mean_var)
use_mean_checkbox.pack(side=tk.LEFT)
# Make default true
use_mean_var.set(True)

# Add frequency bounds controls
tk.Label(control_frame, text="Min Freq (RPM):").pack(side=tk.LEFT)
min_freq_entry = tk.Entry(control_frame, width=10)
min_freq_entry.pack(side=tk.LEFT)
min_freq_entry.insert(0, "35")

tk.Label(control_frame, text="Max Freq (RPM):").pack(side=tk.LEFT)
max_freq_entry = tk.Entry(control_frame, width=10)
max_freq_entry.pack(side=tk.LEFT)
max_freq_entry.insert(0, "75")

# Load sound file once at startup
_warning_sound = AudioSegment.from_wav(os.path.join(os.path.dirname(__file__), "warning.wav"))
_warning_thread = None
_is_warning = False

def play_warning():
    """Play warning sound in a separate thread"""
    global _warning_thread, _is_warning
    if not _is_warning:
        _is_warning = True
        def _play():
            global _is_warning
            while _is_warning:
                play(_warning_sound)
                time.sleep(1)
        _warning_thread = threading.Thread(target=_play, daemon=True)
        _warning_thread.start()

def stop_warning():
    """Stop the warning sound"""
    global _is_warning
    _is_warning = False

def apply_filter(data, window_length, use_mean=False):
    """Apply median or mean filter to data where current value is last in window"""
    import numpy as np
    window_length = int(window_length)
    filtered = np.zeros_like(data)
    for i in range(len(data)):
        start_idx = max(0, i - window_length + 1)
        window = data[start_idx:i + 1]
        filtered[i] = np.mean(window) if use_mean else np.median(window)
    return filtered

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

# Store bound lines globally
bound_lines = []

def apply_parameters():
    """Apply all parameter changes"""
    global bound_lines
    
    # Remove existing bound lines
    for line in bound_lines:
        if line.axes is not None:  # Check if line still exists
            line.remove()
    bound_lines = []
    
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
            
            # Add new bound lines
            if valid_min_freq:
                line = ln.axes.axhline(y=min_freq, color='r', linestyle='--', label='Bound Min')
                bound_lines.append(line)
            if valid_max_freq:
                line = ln.axes.axhline(y=max_freq, color='r', linestyle='--', label='Bound Max')
                bound_lines.append(line)
        
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
    fig, ax = plt.subplots(2, 2, figsize=(8, 4))
    # Format time values to be more readable
    time_vals = freq['time'].astype('datetime64[s]').values
    # Full time series plot
    ln = ax[0,0].plot(time_vals, freq_vals, 'b-', label='Raw')
    ax[0,0].set_title(f"Full freq data for device {i}")
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('Frequency (RPM)')
    ax[0,0].tick_params(axis='x', rotation=45)
    ax[0,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0,0].xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

    # Recent time series plot
    ln_recent = ax[0,1].plot(time_vals, freq_vals, 'b-', label='Raw')
    ax[0,1].set_title(f"Recent freq data for device {i}")
    ax[0,1].set_xlabel('Time')
    ax[0,1].set_ylabel('Frequency (RPM)')
    ax[0,1].tick_params(axis='x', rotation=45)
    ax[0,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0,1].xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

    # Full histogram
    ax[1,0].hist(freq_vals, bins=20, orientation='horizontal')
    ax[1,0].set_title(f"Full histogram for device {i}")
    ax[1,0].set_xlabel('Count')

    # Recent histogram
    ax[1,1].hist(freq_vals, bins=20, orientation='horizontal')
    ax[1,1].set_title(f"Recent histogram for device {i}")
    ax[1,1].set_xlabel('Count')
    line_list.append([ln[0], ln_recent[0]])
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
        line_list[i][0].set_data(time_vals, freq_vals)
        
        # Apply median filter to full series
        _, filter_length = validate_numeric_input(
            median_filter_entry.get(), min_val=1, max_val=1000, param_name="Filter length"
        )
        if filter_length is None:
            filter_length = 5
        filtered_vals = apply_filter(freq_vals, filter_length, use_mean_var.get())
        
        # Get frequency bounds
        _, min_freq = validate_numeric_input(min_freq_entry.get(), min_val=0, param_name="Min frequency")
        _, max_freq = validate_numeric_input(max_freq_entry.get(), min_val=0, param_name="Max frequency")
        
        # Check if filtered values exceed bounds and play/stop warning accordingly
        if min_freq is not None and max_freq is not None:
            if (filtered_vals[-1] < min_freq) | (filtered_vals[-1] > max_freq):
                play_warning()
            else:
                stop_warning()
        
        # Plot both raw and filtered data
        line_list[i][0][0].set_data(time_vals, freq_vals)
        
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
        line_list[i][1].set_data(recent_times, recent_freqs)
        
        # Apply median filter to recent series
        filtered_recent = apply_filter(recent_freqs, filter_length, use_mean_var.get())
        
        # Plot both raw and filtered recent data
        line_list[i][1][0].set_data(recent_times, recent_freqs)
        
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
        
    # Apply parameters if this is the first time
    if len(bound_lines) == 0:
        apply_parameters()

    plt.pause(0.1)  # Add small delay and handle GUI events
    root.update()  # Update the tkinter window
