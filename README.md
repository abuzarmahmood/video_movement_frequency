# Video Movement Frequency Analysis

A Python application for real-time analysis of movement frequencies from video sources, with features for:
- Camera feed processing and frequency analysis
- ROI (Region of Interest) selection
- Real-time visualization with configurable bounds
- Data export and S3 upload capabilities
- Warning system for out-of-bounds frequencies

## Requirements

- Python 3.6+
- OpenCV
- NumPy, SciPy, Pandas
- Matplotlib
- AWS credentials (for S3 upload features)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
   - Copy `src/config.template.json` to `config.json`
   - Add your AWS credentials to `config.json`
   - Or set environment variables: AWS_S3_KEY, AWS_S3_SECRET, AWS_DEFAULT_REGION

## Usage

### ROI Selection
```bash
python src/roi_selector.py <camera-index>
```

### Frequency Analysis
```bash
python src/camera_process.py <camera-index> [options]
```

Options:
- `--n_history`: Number of frames for analysis (default: 100)
- `--append`: Append to existing files
- `--force-overwrite`: Overwrite existing files
- `--animal-number`: Set animal number for output
- `--use-roi`: Use saved ROI
- `--roi`: Set ROI manually (x y width height)

### GUI Visualization
```bash
python src/process_freq.py
```

## Notes

- OpenCV installation on Raspberry Pi may take ~2 hours
- For RPi installation guide: https://singleboardblog.com/install-python-opencv-on-raspberry-pi/
