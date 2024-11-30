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
from glob import glob
# import threading

############################################################

# Convert above process into a class

class CameraFrequency:
# class CameraFrequency(threading.Thread):
    """
    Class to calculate dominant frequency from camera
    """

    def __init__(self, 
                 n_seconds=10,
                 temp_file="~/Desktop/temp.mp4",
                 output_path = "~/Desktop/freq_out.txt",
                 random_dims = 1000,
                 pca_train_n_frames = 1000,
                 video_device = None, 
                 ):
        # threading.Thread.__init__(self)
        self.n_seconds = n_seconds
        self.temp_file = os.path.expanduser(temp_file)
        self.output_path = os.path.expanduser(output_path)
        self.random_projection = None
        self.random_dims = random_dims
        self.pca_train_n_frames = pca_train_n_frames
        self.pca = None
        self.optimize_pca = False
        self.stop = False
        # self.time_freq_list = []
        self.temp_file_access_time = None
        self.loop_delay = 1 # second

        # Check video devices, if only 1 device, use it
        # Else, ask user to input device
        if video_device is None:
            video_devices = sorted(glob("/dev/video*"))
            if len(video_devices) == 1:
                self.video_device = video_devices[0]
            else:
                print("Multiple video devices found. Please specify the device.")
                print(list(enumerate(video_devices)))
                video_index = int(input("Enter the index of the video device: "))
                self.video_device = video_devices[video_index]
        else:
            self.video_device = video_device

    def capture_video(self):
        # Create a temporary file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

        print("Capturing video")

        # Record video
        command = ["ffmpeg", 
                   # "-f", "v4l2", 
                   "-r", "30",
                   "-i", self.video_device, 
                   "-t", str(self.n_seconds), 
                   self.temp_file]
        # subprocess.Popen(command, shell=True)
        subprocess.run(command)

    def load_frames(self):
        # if self.check_file_update():
        print("Loading frames")
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
        diff_frame_stack = np.diff(flat_frame_stack, axis=0)
        zscore_diff_frame_stack = (diff_frame_stack - np.mean(diff_frame_stack, axis=0)) / np.std(diff_frame_stack, axis=0)

        if self.pca is None or self.optimize_pca:
            self.pca = PCA(n_components=1)
            max_n = min(self.pca_train_n_frames, flat_frame_stack.shape[0]-1)
            train_inds = np.random.choice(flat_frame_stack.shape[0]-1, max_n, replace=False)
            self.pca.fit(zscore_diff_frame_stack[train_inds, :])
        reduced_frames = self.pca.transform(zscore_diff_frame_stack)

        self.reduced_frames = reduced_frames.flatten()# [100:]

    def remove_temp_file(self):
        if os.path.exists(self.temp_file):
            print("Removing temp file")
            os.remove(self.temp_file)
        else:
            print("Temp file not found...moving on")


    def calculate_dominant_frequency(self):
        f, Pxx = welch(self.reduced_frames, fs=self.fps, nperseg=1024)
        self.f = f
        self.Pxx = Pxx

        self.dominant_frequency = f[np.argmax(Pxx)]

    def optimize_pca(self):
        self.optimize_pca = True

    def write_tuple_to_file(self, data):
        with open(self.output_path, 'a') as f:
            f.write(str(data)+"\n")

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

    def run(self):
        while not self.stop:
            self.capture_video()
            self.load_frames()
            self.calculate_dominant_frequency()
            self.generate_plots()

            self.remove_temp_file()
            # self.time_freq_list.append((time.time(), self.dominant_frequency))
            time_freq_tuple = ((time.time(), self.dominant_frequency))
            self.write_tuple_to_file(time_freq_tuple)
            time.sleep(int(self.loop_delay))


# Create an instance of the class
camera_frequency = CameraFrequency(
        n_seconds = 5, 
        random_dims = 100,
        video_device = "/dev/video0",)
camera_frequency.run()
