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
from datetime import datetime
import pytz
import json
from upload_to_s3 import main as upload_to_s3

# Create the main window
root = tk.Tk()
root.title("Frequency Visualization")

def read_parameters():
    """Read visualization parameters from file"""
    try:
        params_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'artifacts', 'visualization_params.json')
        with open(params_file, 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"Failed to read parameters: {e}")
        return {
            "y_min": "0",
            "y_max": "100", 
            "time_window": "5",
            "filter_length": "5",
            "min_freq": "35",
            "max_freq": "75",
            "use_mean": True
        }


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

def update_bound_fills(ln, min_freq, max_freq):
    """Update bound fills for a given line plot"""
    # Clear existing fills
    for collection in ln.axes.collections[:]:
        collection.remove()
    
    # Get axes limits
    xmin, xmax = ln.axes.get_xlim()
    ymin, ymax = ln.axes.get_ylim()
    
    # Get the latest filtered value
    filtered_data = ln.get_ydata()
    latest_value = filtered_data[-1] if len(filtered_data) > 0 else None
    
    # Add fill based on the latest value
    if latest_value is not None:
        if min_freq <= latest_value <= max_freq:
            # Within bounds (light blue)
            ln.axes.fill_between([xmin, xmax], min_freq, max_freq, 
                               color='lightblue', alpha=0.7)
        elif latest_value > max_freq:
            # Above bounds (light red)
            ln.axes.fill_between([xmin, xmax], max_freq, ymax, 
                               color='lightcoral', alpha=0.7)
        else:  # latest_value < min_freq
            # Below bounds (light red)
            ln.axes.fill_between([xmin, xmax], ymin, min_freq, 
                               color='lightcoral', alpha=0.7)

def apply_parameters():
    """Apply parameters from file"""
    global bound_lines
    
    # Remove existing bound lines
    for line in bound_lines:
        if line.axes is not None:  # Check if line still exists
            line.remove()
    bound_lines = []
    
    # Read parameters
    params = read_parameters()
    
    # Validate parameters
    valid_ymin, ymin = validate_numeric_input(params["y_min"], param_name="Y-min")
    valid_ymax, ymax = validate_numeric_input(params["y_max"], param_name="Y-max")
    valid_time, time_window = validate_numeric_input(
        params["time_window"], min_val=0.1, max_val=60, param_name="Time window"
    )
    valid_filter, filter_length = validate_numeric_input(
        params["filter_length"], min_val=1, max_val=1000, param_name="Filter length"
    )
    valid_min_freq, min_freq = validate_numeric_input(
        params["min_freq"], min_val=0, param_name="Min frequency"
    )
    valid_max_freq, max_freq = validate_numeric_input(
        params["max_freq"], min_val=0, param_name="Max frequency"
    )
        
    if valid_ymin and valid_ymax and valid_time and valid_filter and valid_min_freq and valid_max_freq:
        if ymin >= ymax:
            print("Y-min must be less than Y-max")
            return

        # Apply y-axis limits to line plots
        for ln in np.array(line_list).flatten():
            ln.axes.set_ylim(ymin, ymax)
            
            # Add new bound lines and fills
            if valid_min_freq and valid_max_freq:
                # Add bound lines
                min_line = ln.axes.axhline(y=min_freq, color='r', linestyle='--', label='Bound Min')
                max_line = ln.axes.axhline(y=max_freq, color='r', linestyle='--', label='Bound Max')
                bound_lines.extend([min_line, max_line])
                
                # Get axes limits
                xmin, xmax = ln.axes.get_xlim()
                ymin, ymax = ln.axes.get_ylim()
                
                # Get the latest filtered value
                filtered_data = ln.get_ydata()
                latest_value = filtered_data[-1] if len(filtered_data) > 0 else None
                
                # Update bound fills
                update_bound_fills(ln, min_freq, max_freq)
        
        # Apply y-axis limits to histograms
        for hist_pair in hist_list:
            for hist_ax in hist_pair:
                hist_ax.set_ylim(ymin, ymax)
        
        print("Parameters applied successfully")
    else:
        print("Parameter validation failed")


plt.ion()

script_path = os.path.realpath(__file__)
src_dir = os.path.dirname(script_path)
base_dir = os.path.dirname(src_dir)
artifact_dir = os.path.join(base_dir, 'artifacts')
recent_data_dir = os.path.join(artifact_dir, 'recent_data')
os.makedirs(recent_data_dir, exist_ok=True)

# Get all files in artifact_dir
freq_files = glob(os.path.join(artifact_dir, 'freq_data_device*.csv'))
print(f'Artifact directory: {artifact_dir}')
print('Frequency files:')
pp(freq_files)
print()

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
filtered_line_list = []
for i, freq in enumerate(freq_data):
    freq_vals = freq['freq'].values
    # Convert from Hz to RPM
    freq_vals = freq_vals * 60
    fig, ax = plt.subplots(2, 2, figsize=(8, 4))
    # Format time values to be more readable
    time_vals = freq['time'].astype('datetime64[s]').values
    # Full time series plot
    ln = ax[0,0].plot(time_vals, freq_vals, '-o', label='Raw', color='b')
    ax[0,0].set_title(f"Full freq data for device {i}")
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('Frequency (RPM)')
    ax[0,0].tick_params(axis='x', rotation=45)
    ax[0,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[0,0].xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

    # Recent time series plot
    ln_recent = ax[0,1].plot(time_vals, freq_vals, '-o', label='Raw', color='b') 
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

    filtered_line_list.append([None, None])

# Load end of file and update plots continuously
while True:
    for i, f in enumerate(freq_files):
        # Read the entire file each time to get full history
        freq = pd.read_csv(f, header=None)
        freq.columns = cols
        
        time_vals = freq['time'].astype('datetime64[s]').values
        freq_vals = freq['freq'].values * 60  # Convert to RPM
        
        # Get recent data window from params file
        params = read_parameters()
        _, time_val = validate_numeric_input(
            params["time_window"], min_val=0.1, max_val=180, param_name="Time window"
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
        # time_vals = np.float64(time_vals)
        line_list[i][0].set_data(time_vals, freq_vals)
        
        # Apply median filter to full series
        params = read_parameters()
        _, filter_length = validate_numeric_input(
            params["filter_length"], min_val=1, max_val=1000, param_name="Filter length"
        )
        if filter_length is None:
            filter_length = 5
        filtered_vals = apply_filter(freq_vals, filter_length, params["use_mean"])
        
        # Get frequency bounds
        _, min_freq = validate_numeric_input(params["min_freq"], min_val=0, param_name="Min frequency")
        _, max_freq = validate_numeric_input(params["max_freq"], min_val=0, param_name="Max frequency")
        
        # Check if filtered values exceed bounds and play/stop warning accordingly
        if min_freq is not None and max_freq is not None:
            if (filtered_vals[-1] < min_freq) | (filtered_vals[-1] > max_freq):
                play_warning()
            else:
                stop_warning()
        
        # Plot both raw and filtered data
        line_list[i][0].set_data(time_vals, freq_vals)
        
        # Get the axes object from the first line
        ax = line_list[i][0].axes
        
        # Check if we already have a filtered line plotted
        if len(ax.lines) > 1:
            ax.lines[1].set_data(time_vals, filtered_vals)
        else:
            filtered_line = ax.plot(time_vals, filtered_vals, 'r-', label='Filtered')[0]
            filtered_line_list[i][0] = filtered_line
            ax.legend()
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.tick_params(axis='x', rotation=45)
        
        # Update recent time series
        # recent_times = np.float64(recent_times)
        line_list[i][1].set_data(recent_times, recent_freqs)
        
        # Apply median filter to recent series
        params = read_parameters()
        filtered_recent = apply_filter(recent_freqs, filter_length, params["use_mean"])
        
        # Plot both raw and filtered recent data
        line_list[i][1].set_data(recent_times, recent_freqs)
        
        # Get the axes object from the recent line
        ax_recent = line_list[i][1].axes
        
        # Check if we already have a filtered line plotted
        if len(ax_recent.lines) > 1:
            ax_recent.lines[1].set_data(recent_times, filtered_recent)
        else:
            filtered_line = ax_recent.plot(recent_times, filtered_recent, 'r-', label='Filtered')[0]
            filtered_line_list[i][1] = filtered_line
            ax_recent.legend()
        
        ax_recent.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax_recent.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax_recent.tick_params(axis='x', rotation=45)
        
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

        # Write recent data to file
        recent_data = pd.DataFrame({
            'time': recent_times,
            'freq': recent_freqs,
            'freq_filtered': filtered_recent
        })
        recent_data.to_csv(
            os.path.join(recent_data_dir, f'recent_data_device_{i}.csv'),
            index=False
        )
        
        # Upload new data to S3
        try:
            upload_to_s3()
        except Exception as e:
            print(f"Failed to upload to S3: {e}")
        
        # Write frequency bounds and y-limits to file
        if min_freq is not None and max_freq is not None:
            local_tz = datetime.now().astimezone().tzinfo
            bounds_data = pd.DataFrame({
                'min_freq': [min_freq],
                'max_freq': [max_freq],
                'y_min': [float(params["y_min"])],
                'y_max': [float(params["y_max"])],
                'timezone': [str(local_tz)]
            })
            bounds_data.to_csv(
                os.path.join(recent_data_dir, f'freq_bounds_device_{i}.csv'),
                index=False
            )
        
        # Update axis limits
        for ln in line_list[i]:
            ln.axes.relim()
            ln.axes.autoscale_view()

        # Update bound fills for both full and recent plots
        params = read_parameters()
        _, min_freq = validate_numeric_input(params["min_freq"], min_val=0, param_name="Min frequency")
        _, max_freq = validate_numeric_input(params["max_freq"], min_val=0, param_name="Max frequency")
        if min_freq is not None and max_freq is not None:
            for ln in filtered_line_list[i]: 
                update_bound_fills(ln, min_freq, max_freq)
        
    # Apply parameters if this is the first time
    if len(bound_lines) == 0:
        apply_parameters()
        

    plt.pause(0.1)  # Add small delay and handle GUI events
    root.update()  # Update the tkinter window
