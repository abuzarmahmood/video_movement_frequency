"""
1) Load frames from camera
2) Reduce frame to timeseries
3) Predict dominant frequency using welch method
4) Display dominant frequency timeseries
"""

import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import time
from tqdm import tqdm
import threading
from sklearn.decomposition import PCA

# # Load video
# camera = iio.get_reader("<video0>")
# meta = camera.get_meta_data()
# delay = 1/meta["fps"]
# for frame_counter in range(15):
#     frame = camera.get_next_data()
#     time.sleep(delay)
# camera.close()
# 
# plt.imshow(frame)
# plt.show()

# Create thread to record frames and append to list

class FrameRecorder(threading.Thread):
    """
    Only keep the most recent frame
    """
    def __init__(self, camera): 
        threading.Thread.__init__(self)
        self.camera = camera
        self.frame = None
        self.frame_time = None
        self.stop = False
        meta = camera.get_meta_data()
        self.delay = 1/meta["fps"]

    def run(self):
        while not self.stop:
            self.frame = self.camera.get_next_data()
            self.frame_time = time.time()
            time.sleep(self.delay)

    def get_frame(self):
        return self.frame

    def get_frame_time(self):
        return self.frame_time

    def stop(self):
        self.stop = True

camera = iio.get_reader("<video0>")
this_recorder = FrameRecorder(camera)
this_recorder.start()

# Perform frequency estimation
# 1) Take first n frames to infer PCA
# 2) Reduce frames to single values
# 3) Perform welch method
# 4) Display dominant frequency

n_optim_seconds = 5
meta = camera.get_meta_data()
n_optim_frames = int(n_optim_seconds * meta["fps"])

frames = []
frame_times = []
current_time = this_recorder.get_frame_time()
pbar = tqdm(total=n_optim_frames)
while len(frames) < n_optim_frames:
    if this_recorder.get_frame_time() > current_time:
        frames.append(this_recorder.get_frame())
        frame_times.append(this_recorder.get_frame_time())
        current_time = this_recorder.get_frame_time()
        pbar.update(1)
pbar.close()

# Reduce frames to single values
frame_array = np.stack(frames)
flat_frame_array = np.reshape(frame_array, (frame_array.shape[0], -1))
pca = PCA(n_components=1)
pca.fit(flat_frame_array)
reduced_frames = pca.transform(flat_frame_array)

t_vec = np.array(frame_times)
t_vec = t_vec - t_vec[0]
plt.plot(t_vec, reduced_frames)
plt.show()

plt.imshow(frames[0])
plt.show()

############################################################
# Record video for n seconds

n = 10 # seconds
meta = camera.get_meta_data()
n_frames = int(n * meta["fps"])

############################################################
# Use subprocess to record a video for n seconds using ffmpeg
n = 10

import subprocess
import os
import time

# Create a temporary file
temp_file = "~/Desktop/temp.mp4"
temp_file = os.path.expanduser(temp_file)
if os.path.exists(temp_file):
    os.remove(temp_file)

# Record video
command = ["ffmpeg", "-f", "v4l2", "-i", "/dev/video0", "-t", str(n), temp_file]
subprocess.run(command)

# Load video
camera = iio.get_reader(temp_file)
meta = camera.get_meta_data()
duration = meta["duration"]
fps = meta["fps"]
n_frames = int(duration * fps)

# Load frames
frames = []
for frame_counter in range(n_frames):
    frame = camera.get_next_data()
    frames.append(frame)

# Flatten frames and perform pca
frame_stack = np.stack(frames)
flat_frame_stack = np.reshape(frame_stack, (frame_stack.shape[0], -1))

pca = PCA(n_components=1)
pca.fit(flat_frame_stack)
reduced_frames = pca.transform(flat_frame_stack)

reduced_frames = reduced_frames.flatten()[100:]
plt.plot(reduced_frames)
plt.show()

# Remove temporary file
os.remove(temp_file)

# Calculate dominant frequency
f, Pxx = welch(reduced_frames, fs=fps, nperseg=1024)
plt.plot(f, Pxx)
plt.show()

# Find dominant frequency
dominant_frequency = f[np.argmax(Pxx)]

