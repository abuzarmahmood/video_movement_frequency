import numpy as np
import cv2
from time import sleep, time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from scipy.signal import welch

base_dir = '/home/abuzarmahmood/projects/video_movement_frequency'
plot_dir = os.path.join(base_dir, 'plots')
artifact_dir = os.path.join(base_dir, 'artifacts')

def calc_freq(data, fs):
    """
    Calculate frequency of data using Welch's method

    Parameters
    ----------
    data : np.array, shape (n_samples, n_features)
    fs : int, sampling rate

    Returns
    -------
    freq : float, dominant frequency
    """
    pca_data = PCA(n_components=1, whiten=True).fit_transform(data)
    f, Pxx = welch(pca_data.flatten(), fs=fs)
    peak_freq = f[np.argmax(Pxx)]
    return peak_freq

# # Create class as thread that will run in the background
# # and will run computation if new payload is received
# class FreqCalc:
#     def __init__(self):
#         self.payload = None
#         self.freq = 0
#         self.last_time = datetime.now().strftime("%H:%M:%S")
#         self.counter = 0
#         self.last_counter = 0
# 
#     # Function to calculate frequency
#     # payload shape: [(n_samples, n_features), sampling_rate]
#     def calc_freq(self):
#         if self.payload is not None:
#             fs = self.payload[1]
#             data = self.payload[0]
#             # Perform PCA
#             pca_data = PCA(n_components=1, whiten=True).fit_transform(self.data)
#             # Perform Welch's method
#             f, Pxx = welch(pca_data.flatten(), fs=fs) 
#             # Find peak frequency
#             peak_freq = f[np.argmax(Pxx)]
#             self.freq = peak_freq
#             self.payload = None
#             self.last_time = datetime.now().strftime("%H:%M:%S") 
#             self.last_counter = self.counter
#         self.counter += 1
# 
# 
# # Run as thread
# from threading import Thread
# 
# this_freq_calc = FreqCalc()
# thread = Thread(target=this_freq_calc.calc_freq)
# thread.start()
# # thread.join()

cap = cv2.VideoCapture(0)

# Get camera resolution 
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Get fps
fps = cap.get(cv2.CAP_PROP_FPS)

# Set lower resolution
# militiplier = 0.1
# cap.set(3, width * militiplier)
# cap.set(4, height * militiplier)
cap.set(3, 320)
cap.set(4, 180)

freq_file = os.path.join(artifact_dir, "freq_data.csv")
if os.path.exists(freq_file):
    os.remove(freq_file)
n_max_var_pixels = 25
n_history = 100
counter = 0
time_stamps = []
frame_list = []
max_var_timeseries = []
fs_list = []
while(True):
    # Recreate these lists each time, so that they are not stored
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    time_stamps.append(time())
    # Have to make a copy of the frame, otherwise it will be overwritten
    # with annotations below
    frame_list.append(gray.copy())
    sleep(0.01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    counter += 1

    # Once at 100 frames, calculate:
    # 1) Average frame rate and display on image
    # 2) Pixels with most variability, and display on image
    # if len(time_stamps) > n_history:
    if counter % n_history == 0 and counter > 0:
        frame_rate = 1 / np.mean(np.diff(time_stamps[-n_history:]))
        variance = np.var(frame_list[-n_history:], axis=0)
        var_color = cv2.applyColorMap(
            np.uint8(variance / np.max(variance) * 255), cv2.COLORMAP_JET)
        max_var_pixels = np.unravel_index(
                np.argsort(variance.ravel())[-n_max_var_pixels:], 
                variance.shape)
        max_var_pixel_vals = np.stack(frame_list[-n_history:], axis=0)[
            :, max_var_pixels[0], max_var_pixels[1]] 
        max_var_timeseries.append(max_var_pixel_vals)
        freq_val = calc_freq(max_var_pixel_vals, frame_rate)
        fs_list.append(frame_rate)
        time_stamps = []
        frame_list = []
        # Save timeseries of max variance pixels
        # print(f"Frame rate: {frame_rate}")
        out_data = [frame_rate, freq_val, datetime.now().strftime('%H:%M:%S'), counter]
        with open(freq_file, 'a') as f:
            f.write(','.join(map(str, out_data)) + '\n')
    if counter > n_history:
        print_str_list = [
                f"Frame rate: {frame_rate:.2f} fps",
                f"Peak frequency: {freq_val:.2f} Hz",
                f"Time: {datetime.now().strftime('%H:%M:%S')}",
                f"Counter: {counter}"
                ]
        for i, print_str in enumerate(print_str_list):
            cv2.putText(gray, 
                        print_str,
                        (10, 10 + 20 * i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2)
        for i in range(n_max_var_pixels):
            cv2.circle(gray, 
                       (max_var_pixels[1][i], max_var_pixels[0][i]), 
                       3, (255, 0, 0), -1)
        cv2.imshow('var', var_color)
    cv2.imshow('frame', gray) 

# Save max_var_timeseries as artifact
np.save(os.path.join(artifact_dir, "max_var_timeseries.npy"), max_var_timeseries)

cat_time_series = np.stack(max_var_timeseries, axis=0)
long_time_series = cat_time_series.reshape(-1, n_max_var_pixels)

fig, ax = plt.subplots(2,1, figsize=(10, 5))
ax[0].plot(long_time_series)
ax[0].set_title("Time series of most variable pixels")
ax[1].imshow(long_time_series.T, aspect='auto', interpolation='none')
ax[1].set_title("Heatmap of most variable pixels")
save_path = os.path.join(plot_dir, "most_var_pixels.png")
plt.savefig(save_path)
plt.close()

freq_data = [calc_freq(x, this_fs) for x, this_fs in zip(max_var_timeseries, fs_list)]

# Plot timeseries after calculating PCA
# diff_timeseries = [np.diff(x, axis=-1) for x in max_var_timeseries]
pca_data = [PCA(n_components = 1, whiten=True).\
        fit_transform(x) for x in max_var_timeseries]
        # fit_transform(x) for x in diff_timeseries]
cat_pca_data = np.concatenate(pca_data, axis=0) 

# Calculate frequency
# freq_data= []
# for x in pca_data:
#     f, Pxx = welch(x.flatten(), fs = 15)
#     freq_data.append(f[np.argmax(Pxx)])
# freq_data = []
# for x in max_var_timeseries:
#     this_freq_calc.payload = [x, 15]
#     sleep(1)
#     freq_data.append(this_freq_calc.freq)

freq_data_timeseries = [np.repeat(x, n_history) for x in freq_data]

fig, ax = plt.subplots(2,1, figsize=(15, 10), sharex=True)
ax[0].plot(cat_pca_data, '-x')
for i in range(0, len(cat_pca_data), n_history):
    ax[0].axvline(i, color='r', linestyle='--')
ax[0].set_title("PCA of most variable pixels")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("PCA")
ax[1].plot(np.concatenate(freq_data_timeseries, axis=0), '-x')
for i in range(0, len(cat_pca_data), n_history):
    ax[1].axvline(i, color='r', linestyle='--')
ax[1].set_title("Dominant frequency")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Frequency (Hz)")
plt.savefig(os.path.join(plot_dir, "pca_most_var_pixels.png"))
plt.close()


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
