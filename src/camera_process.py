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
import threading
import sys

# base_dir = '/home/abuzarmahmood/projects/video_movement_frequency'
script_path = os.path.realpath(__file__)
base_dir = os.path.dirname(os.path.dirname(script_path))
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


def get_capture(device_id=0, width=320, height=180):
    """
    Get capture object for camera

    Parameters
    ----------
    device_id : int, default 0
    width : int, default 320
    height : int, default 180

    Returns
    -------
    cap : cv2.VideoCapture object
    """
    cap = cv2.VideoCapture(device_id)
    cap.set(3, width)
    cap.set(4, height)
    return cap

def run_cap_freq_estim(device_id, artifact_dir, plot_dir):
    cap = get_capture(
            device_id=device_id, 
            width=320, 
            height=180)
    cv2.namedWindow(f'frame_{device_id}')
    cv2.namedWindow(f'var_{device_id}')
    freq_file = os.path.join(artifact_dir, f"freq_data_device{device_id}.csv")
    if os.path.exists(freq_file):
        # Ask user if they want to overwrite
        print(f"File {freq_file} already exists. Overwrite? (y/n)")
        user_input = input()
        while user_input not in ['y', 'n']:
            print("Invalid input. Please enter 'y' or 'n'")
            user_input = input()
        if user_input == 'y':
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
        counter += 1
        sleep(0.01)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Once at 100 frames, calculate:
        # 1) Average frame rate and display on image
        # 2) Pixels with most variability, and display on image
        # if len(time_stamps) > n_history:
        if counter % n_history == 0 and counter > 0:
            frame_rate = 1 / np.mean(np.diff(time_stamps[-n_history:]))
            variance = np.var(frame_list[-n_history:], axis=0)
            var_color = cv2.applyColorMap(
                np.uint8(variance / np.max(variance) * 255), cv2.COLORMAP_JET)
            # Instead of using the most variable pixels, sample pixels
            # weights by variance
            random_weights = np.random.rand(*variance.shape)
            var_weights = variance * random_weights
            max_var_pixels = np.unravel_index(
                    np.argsort(var_weights.ravel())[-n_max_var_pixels:], 
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
            # Time in HH:MM:SS
            str_time = datetime.now().strftime('%H:%M:%S')
        if counter > n_history:
            print_str_list = [
                    f"Frame rate: {frame_rate:.2f} fps",
                    f"Peak frequency: {freq_val:.2f} Hz",
                    f"Time: {str_time}",
                    f"Counter: {counter}"
                    ]
            for i, print_str in enumerate(print_str_list):
                cv2.putText(gray, 
                            print_str,
                            (10, 10 + 20 * i), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                           (255, 0, 0), 
                            2)
            for i in range(n_max_var_pixels):
                cv2.circle(gray, 
                           (max_var_pixels[1][i], max_var_pixels[0][i]), 
                           3, 
                           (255, 0, 0), 
                           -1)
            cv2.imshow(f'var_{device_id}', var_color)
        cv2.imshow(f'frame_{device_id}', gray) 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


############################################################
############################################################

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        # camPreview(self.previewName, self.camID)
        run_cap_freq_estim(self.camID, artifact_dir, plot_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide camera index")
        sys.exit()

    camera_ind = sys.argv[1]
    print(f"Running camera {camera_ind}")

    thread1 = camThread("Camera 1", int(camera_ind))
    thread1.start()
    thread1.join()

    # Force camera release
    cv2.destroyAllWindows()
