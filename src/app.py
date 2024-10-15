"""
1) Load frames from camera
2) Reduce frame to timeseries
3) Predict dominant frequency using welch method
4) Display dominant frequency timeseries
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import time
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
import imageio as iio
import subprocess

############################################################
# Use subprocess to record a video for n seconds using ffmpeg
n = 10 # seconds

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
# plt.plot(reduced_frames)
# plt.show()

# Remove temporary file
os.remove(temp_file)

# Calculate dominant frequency
f, Pxx = welch(reduced_frames, fs=fps, nperseg=1024)
plt.plot(f, Pxx)
plt.show()

# Find dominant frequency
dominant_frequency = f[np.argmax(Pxx)]

############################################################

# Convert above process into a class

class CameraFrequency:
    """
    Class to calculate dominant frequency from camera
    """

    def __init__(self, 
                 n_seconds=10,
                 temp_file="~/Desktop/temp.mp4",
                 random_dims = 1000,
                 pca_train_n_frames = 1000,
                 ):
        self.n_seconds = n_seconds
        self.temp_file = os.path.expanduser(temp_file)
        self.random_projection = None
        self.random_dims = random_dims
        self.pca_train_n_frames = pca_train_n_frames
        self.pca = None

    def capture_video(self):
        # Create a temporary file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

        # Record video
        command = ["ffmpeg", "-f", "v4l2", "-i", "/dev/video0", "-t", 
                   str(self.n_seconds), self.temp_file]
        subprocess.run(command)

    def load_frames(self):
        camera = iio.get_reader(self.temp_file)
        meta = camera.get_meta_data()
        self.duration = meta["duration"]
        self.fps = meta["fps"]
        n_frames = int(self.duration * self.fps)

        # Load frames
        frames = []
        for frame_counter in trange(n_frames):
            frame = camera.get_next_data()
            flat_frame = np.reshape(frame, -1)
            if self.random_projection is None:
                self.random_projection = np.random.randn(flat_frame.shape[0], self.random_dims)
            reduced_frame = np.dot(flat_frame, self.random_projection)
            frames.append(reduced_frame)

        # Flatten frames and perform pca
        frame_stack = np.stack(frames)
        flat_frame_stack = np.reshape(frame_stack, (frame_stack.shape[0], -1))

        if self.pca is None:
            self.pca = PCA(n_components=1)
            train_inds = np.random.choice(flat_frame_stack.shape[0], self.pca_train_n_frames)
            self.pca.fit(flat_frame_stack[train_inds, :])
        reduced_frames = self.pca.transform(flat_frame_stack)

        self.reduced_frames = reduced_frames.flatten()# [100:]

    def remove_temp_file(self):
        os.remove(self.temp_file)

    def calculate_dominant_frequency(self):
        f, Pxx = welch(self.reduced_frames, fs=self.fps, nperseg=1024)
        self.f = f
        self.Pxx = Pxx

        self.dominant_frequency = f[np.argmax(Pxx)]

    def generate_plots(self):
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        t_vec = np.arange(0, len(self.reduced_frames)) / self.fps 
        ax[0].plot(self.reduced_frames)
        ax[1].plot(self.f, self.Pxx)
        ax[0].set_title("Reduced Frames")
        ax[1].set_title("Power Spectral Density")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Power")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Reduced Frames")
        ax[1].axvline(self.dominant_frequency, color="red", linestyle="--")
        plt.show()

    def run_process(self):
        """
        If pca already exists, use it
        """

# Create an instance of the class
camera_frequency = CameraFrequency(n_seconds = 15, random_dims = 100)
# camera_frequency.capture_video()
camera_frequency.load_frames()
camera_frequency.calculate_dominant_frequency()
camera_frequency.generate_plots()

camera_frequency.remove_temp_file()
