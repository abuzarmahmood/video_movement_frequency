import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera_process import calc_freq, run_cap_freq_estim
import os
import tempfile
import pandas as pd

base_dir = '/home/abuzarmahmood/projects/video_movement_frequency'

def generate_chirp_video(
    duration=20,  # seconds
    fps=30,
    start_freq=0.5,  # Hz
    end_freq=5.0,  # Hz
    resolution=(320, 240),
    output_file=None
):
    """
    Generate a video with a spatial wave chirp pattern.
    The frequency increases linearly from start_freq to end_freq over the duration.
    """
    if output_file is None:
        output_file = os.path.join(tempfile.gettempdir(), 'chirp_test.avi')
    
    # Calculate number of frames
    n_frames = int(duration * fps)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, resolution, False)
    
    # Time vector
    t = np.linspace(0, duration, n_frames)
    
    # Frequency vector (linear chirp)
    freq = np.linspace(start_freq, end_freq, n_frames)
    
    # Generate frames
    x = np.linspace(0, 2*np.pi, resolution[0])
    spatial_freq = 5  # constant spatial frequency
    
    # Phase will increase based on instantaneous frequency
    phase = np.cumsum(freq) / fps * 2 * np.pi
    
    for i in range(n_frames):
        # Create spatial wave pattern with temporal modulation
        wave = np.sin(spatial_freq * x + phase[i])
        # Expand to 2D
        frame = np.tile(wave, (resolution[1], 1))
        # Normalize to 0-255 range
        frame = ((frame + 1) * 127.5).astype(np.uint8)
        out.write(frame)
    
    out.release()
    return output_file, freq

def analyze_chirp_video(video_path, artifact_dir, plot_dir):
    """
    Analyze the chirp video and compare estimated frequencies with ground truth.
    """
    # Create directories if they don't exist
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Run frequency estimation
    estimated_freqs = run_cap_freq_estim(video_path, artifact_dir, plot_dir)
    
    return estimated_freqs

def plot_results(true_freq, 
                 estimated_x,
                 estimated_freq, 
                 plot_dir):
    """
    Plot the comparison between true and estimated frequencies.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(true_freq, label='True Frequency')
    plt.plot(estimated_x, estimated_freq, label='Estimated Frequency')
    plt.xlabel('Frame Number')
    plt.ylabel('Frequency (Hz)')
    plt.title('True vs Estimated Frequency')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(plot_dir, 'freq_comparison.png')
    plt.savefig(plot_path)
    plt.close()

def main():
    # Set up directories
    artifact_dir = os.path.join(base_dir, 'artifacts') 
    plot_dir = os.path.join(base_dir, 'plots') 
    
    # Generate test video
    video_path, true_freq = generate_chirp_video(
        duration=20,
        fps=30,
        start_freq=0.5,
        end_freq=5.0,
        output_file=os.path.join(artifact_dir, 'chirp_test.avi')
    )
    
    # Analyze video
    # This will eventually break because it expects a camera input
    _ = analyze_chirp_video(video_path, artifact_dir, plot_dir)
    estim_data = pd.read_csv(os.path.join(artifact_dir, 'freq_data_devicetest.csv'),
                             header=None)
    estim_data.columns = ['frame_rate', 'freq_val', 'time', 'counter']
    estimated_x = estim_data['counter']
    estimated_freq = estim_data['freq_val']
    
    # Plot results
    plot_results(true_freq, estimated_x, estimated_freq, plot_dir)
    
    # Clean up
    # os.remove(video_path)

if __name__ == "__main__":
    main()
