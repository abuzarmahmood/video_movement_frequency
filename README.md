# Video Movement Frequency Analysis

A Python application for real-time analysis of movement frequencies from video sources. This tool is designed for monitoring and analyzing periodic movements captured through video cameras, with features for automated frequency detection, visualization, and data management.

## Features

- **Multi-Camera Support**: Process multiple video feeds simultaneously
- **ROI Selection**: Define specific regions of interest for focused analysis
- **Real-time Analysis**:
  - Frequency detection using PCA and spectral analysis
  - Configurable analysis parameters
  - Live visualization of results
- **Data Management**:
  - Automatic data export to CSV
  - AWS S3 integration for cloud storage
  - Historical data tracking
- **Visualization**:
  - Real-time plots of frequency data
  - Histogram views
  - Configurable display bounds
- **Alert System**: Audio warnings for out-of-bounds frequencies

## Requirements

### Software Dependencies
- Python 3.6+
- OpenCV (cv2)
- Scientific Computing:
  - NumPy
  - SciPy
  - Pandas
  - scikit-learn
- Visualization:
  - Matplotlib
  - tkinter
- Audio:
  - pydub
- AWS SDK:
  - boto3

### Hardware Requirements
- One or more USB cameras
- Audio output (for warnings)
- Sufficient CPU for real-time processing
- Network connection (for S3 uploads)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video-frequency-analysis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials (if using S3 upload):
   - Copy the template: `cp src/config.template.json config.json`
   - Edit `config.json` with your AWS credentials:
     ```json
     {
         "aws": {
             "AWS_S3_KEY": "your-access-key",
             "AWS_S3_SECRET": "your-secret-key",
             "AWS_DEFAULT_REGION": "your-region",
             "AWS_S3_BUCKET": "your-bucket-name"
         }
     }
     ```
   - Or set environment variables:
     ```bash
     export AWS_S3_KEY=your-access-key
     export AWS_S3_SECRET=your-secret-key
     export AWS_DEFAULT_REGION=your-region
     export AWS_S3_BUCKET=your-bucket-name
     ```

## Usage

### 1. ROI Selection
Define regions of interest for each camera:
```bash
python src/roi_selector.py <camera-index>
```
- Click and drag to select ROI
- Press SPACE to confirm selection
- Press ESC to cancel

### 2. Camera Processing
Start video capture and analysis:
```bash
python src/camera_process.py <camera-index> [options]
```

Options:
- `--n_history N`: Number of frames for analysis window (default: 100)
- `--append`: Append to existing data files
- `--force-overwrite`: Overwrite existing files without prompting
- `--animal-number N`: Set animal identifier for output files
- `--use-roi`: Use previously saved ROI
- `--roi x y w h`: Set ROI manually (x y width height)

### 3. Frequency Visualization
Launch the visualization GUI:
```bash
python src/process_freq.py
```

GUI Features:
- Real-time frequency plots
- Adjustable Y-axis limits
- Time window selection
- Filtering options
- Frequency bound settings
- Warning system controls

### 4. Data Management
- Data is automatically saved to CSV files in `artifacts/` directory
- Recent data is continuously uploaded to S3 if configured
- Use the GUI's "Purge S3 Bucket" button to clear cloud storage

## Troubleshooting

### Common Issues
1. Camera Access:
   - Ensure camera permissions are set correctly
   - Check USB connections
   - Verify camera index matches hardware

2. Performance:
   - Reduce frame resolution if CPU usage is high
   - Adjust analysis window size
   - Close other applications using the camera

3. S3 Upload:
   - Verify network connection
   - Check AWS credentials
   - Ensure S3 bucket exists and is accessible

### Platform-Specific Notes

#### Raspberry Pi
- OpenCV installation may take ~2 hours
- Installation guide: https://singleboardblog.com/install-python-opencv-on-raspberry-pi/
- Consider using pre-built OpenCV packages:
  ```bash
  sudo apt-get install python3-opencv
  ```

#### Windows
- Ensure Microsoft Visual C++ is installed for OpenCV
- Use DirectShow camera backend if default fails

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
