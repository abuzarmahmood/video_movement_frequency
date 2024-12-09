import numpy as np
import cv2
import json
from time import sleep, time
from datetime import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from scipy.signal import welch
import threading
import sys
from functools import partial

script_path = os.path.realpath(__file__)
base_dir = os.path.dirname(os.path.dirname(script_path))
artifact_dir = os.path.join(base_dir, 'artifacts')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)



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


def load_roi(device_id):
    """Load ROI coordinates from JSON file if it exists"""
    roi_file = os.path.join(artifact_dir, f"roi_device_{device_id}.json")
    if os.path.exists(roi_file):
        with open(roi_file, 'r') as f:
            roi_data = json.load(f)
            return roi_data['roi']
    return None

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
    # Check if device_id is an integer
    if isinstance(device_id, int):
        cap = cv2.VideoCapture(device_id)
        cap.set(3, width)
        cap.set(4, height)
    elif isinstance(device_id, str):
        cap = cv2.VideoCapture(device_id)
    return cap

def run_cap_freq_estim(
        device_id, 
        artifact_dir, 
        n_history=100, 
        no_overwrite=True, 
        animal_number=None,
        roi=None,
        use_roi=False
        ):
    """
    Run camera frequency estimation
    
    Parameters
    ----------
    device_id : int or str
        Camera device index or video file path
    artifact_dir : str
        Directory to save frequency data
    n_history : float
        Number of data points to use as history 
    no_overwrite : bool
        If True, don't overwrite existing files
    roi : tuple, optional
        Region of interest (x, y, width, height) to analyze. If None, use full frame.
    """
    # Only load ROI from file if --use-roi flag is given and no explicit ROI provided
    if roi is not None:
        print(f"Using given ROI: {roi}")
    if roi is None and use_roi: 
        print(f'Using ROI from file for device {device_id}')
        roi = load_roi(device_id)
        if roi is not None:
            print(f"Loaded ROI from file: {roi}")
        else:
            print("No ROI found in file... Using full frame.")
    cap = get_capture(
            device_id=device_id, 
            width=320, 
            height=180)
    if not isinstance(device_id, int):
        device_id = 'test'
    cv2.namedWindow(f'frame_{device_id}')
    cv2.namedWindow(f'var_{device_id}')
    file_id = f"animal{animal_number}" if animal_number is not None else f"device{device_id}"
    freq_file = os.path.join(artifact_dir, f"freq_data_{file_id}.csv")
    # Check if file exists
    # If no_overwrite is False, then overwrite
    if os.path.exists(freq_file):
        if no_overwrite:  # append mode
            pass  # keep existing file
        elif args.force_overwrite:
            os.remove(freq_file)  # force overwrite without asking
        else:
            # Ask user if they want to overwrite
            print(f"File {freq_file} already exists. Overwrite? (y/n)")
            user_input = input()
            while user_input not in ['y', 'n']:
                print("Invalid input. Please enter 'y' or 'n'")
                user_input = input()
            if user_input == 'y':
                os.remove(freq_file)
    n_max_var_pixels = 25
    counter = 0
    time_stamps = []
    frame_list = []
    roi_list = []
    # max_var_timeseries = []
    # fs_list = []
    # If a video input is given, run loop until video ends
    if device_id == 'test':
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        def condition_tester(counter):
            return counter < n_frames
    else:
        condition_tester = lambda x: True
    # while(True):
    print(f'Using history of {n_history} frames')
    while condition_tester(counter):
        # Recreate these lists each time, so that they are not stored
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If ROI is specified, only use that region
        if roi is not None:
            x, y, w, h = roi
            gray_roi = gray[y:y+h, x:x+w]
        else:
            gray_roi = gray
        roi_list.append(gray_roi)

        # Draw ROI rectangle if specified
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)

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
            # Calculate variance on ROI if specified
            if roi is not None:
                x, y, w, h = roi
                # frame_array = np.array([f[y:y+h, x:x+w] for f in frame_list[-n_history:]])
                frame_array = np.array(roi_list[-n_history:])
                roi_list = []
                variance = np.var(frame_array, axis=0)
            else:
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
            # If ROI is specified, convert max_var_pixels to full frame coordinates
            if roi is not None:
                x, y, w, h = roi
                max_var_pixels = (max_var_pixels[0] + y, max_var_pixels[1] + x)
            max_var_pixel_vals = np.stack(frame_list[-n_history:], axis=0)[
                :, max_var_pixels[0], max_var_pixels[1]] 
            # max_var_timeseries.append(max_var_pixel_vals)
            freq_val = calc_freq(max_var_pixel_vals, frame_rate)
            # fs_list.append(frame_rate)
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
                    f"Peak frequency: {freq_val:.2f} Hz, {freq_val * 60:.2f} bpm",
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
    def __init__(
            self, 
            previewName, 
            camID, 
            n_history=100, 
            no_overwrite=True, 
            animal_number=None, 
            roi=None,
            use_roi=False
            ):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.n_history = n_history
        self.no_overwrite = no_overwrite
        self.animal_number = animal_number
        self.roi = roi
        self.use_roi = use_roi

    def run(self):
        print("Starting " + self.previewName)
        # camPreview(self.previewName, self.camID)
        run_cap_freq_estim(self.camID, artifact_dir,
                          n_history=self.n_history,
                          no_overwrite=self.no_overwrite,
                          animal_number=self.animal_number,
                          roi=self.roi,
                          use_roi=self.use_roi
                           )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera frequency estimation')
    parser.add_argument('camera_index', type=int, help='Camera device index')
    parser.add_argument('--n_history', type=int, default=100,
                      help='Number of frames to use for frequency estimation (default: 100)')
    parser.add_argument('--append', action='store_true', help='Append to existing files instead of overwriting')
    parser.add_argument('--force-overwrite', action='store_true', help='Force overwrite existing files without prompting')
    parser.add_argument('--animal-number', type=int, help='Animal number to use in output filename')
    parser.add_argument('--use-roi', action='store_true',
                      help='Use ROI from saved file')
    parser.add_argument('--roi', type=int, nargs=4, 
                      metavar=('x', 'y', 'width', 'height'),
                      help='Region of interest (x y width height)')
    args = parser.parse_args()

    print(f"Running camera {args.camera_index} with n_history={args.n_history}")

    thread1 = camThread("Camera 1", args.camera_index,
                       n_history=args.n_history,
                       no_overwrite=args.append,  # map append to the internal no_overwrite flag
                       animal_number=args.animal_number,
                       roi=args.roi if args.roi else None,
                       use_roi=args.use_roi,
                        )
    thread1.start()
    thread1.join()

    # Force camera release 
    cv2.destroyAllWindows()
