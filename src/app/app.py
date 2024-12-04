import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from glob import glob
import os
import time
import threading
import atexit
import pygame.mixer
import pygame
import boto3
import io
from botocore.exceptions import ClientError

# Initialize pygame mixer and global variables
pygame.mixer.init()


# Create a simple warning beep
def create_warning_beep():
    sample_rate = 44100
    duration = 0.5  # seconds
    frequency = 440  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = np.sin(2 * np.pi * frequency * t)
    # Create stereo by duplicating mono signal
    stereo = np.vstack((samples, samples)).T
    # Ensure array is contiguous and in the correct format
    scaled = np.ascontiguousarray(stereo * 32767, dtype=np.int16)
    return pygame.sndarray.make_sound(scaled)

# Initialize warning sound
warning_beep = create_warning_beep()


st.set_page_config(page_title="Frequency Monitor", layout="wide")
st.title("Real-time Frequency Monitoring")

# AWS Configuration
st.sidebar.header("Data Source Configuration")
use_s3 = st.sidebar.checkbox("Use AWS S3")

if use_s3:
    aws_access_key = st.sidebar.text_input("AWS Access Key ID", type="password")
    aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", type="password")
    s3_bucket = st.sidebar.text_input("S3 Bucket Name")
    s3_prefix = st.sidebar.text_input("S3 Prefix (folder path)", value="recent_data/")

# Get data directory
base_dir = '/home/abuzarmahmood/projects/video_movement_frequency'
recent_data_dir = os.path.join(base_dir, 'artifacts', 'recent_data')

def get_s3_client():
    """Create and return an S3 client using provided credentials"""
    if not (aws_access_key and aws_secret_key):
        st.error("AWS credentials are required when using S3")
        return None
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        return s3_client
    except Exception as e:
        st.error(f"Failed to create S3 client: {str(e)}")
        return None

def load_s3_data(device_num):
    """Load data from S3 bucket"""
    s3_client = get_s3_client()
    if not s3_client:
        return None, None
    
    try:
        # Construct S3 paths
        data_key = f"{s3_prefix.rstrip('/')}recent_data_device_{device_num}.csv"
        bounds_key = f"{s3_prefix.rstrip('/')}freq_bounds_device_{device_num}.csv"
        
        # Get data file
        response = s3_client.get_object(Bucket=s3_bucket, Key=data_key)
        data = pd.read_csv(io.BytesIO(response['Body'].read()))
        data['time'] = pd.to_datetime(data['time'])
        
        # Try to get bounds file
        try:
            bounds_response = s3_client.get_object(Bucket=s3_bucket, Key=bounds_key)
            bounds = pd.read_csv(io.BytesIO(bounds_response['Body'].read()))
        except ClientError:
            bounds = None
            
        return data, bounds
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None, None
        else:
            st.error(f"Error accessing S3: {str(e)}")
            return None, None
    except Exception as e:
        st.error(f"Error loading S3 data: {str(e)}")
        return None, None

# Function to load and process data
def load_data(device_num):
    if use_s3:
        return load_s3_data(device_num)
    
    try:
        recent_data_file = os.path.join(recent_data_dir, f'recent_data_device_{device_num}.csv')
        bounds_file = os.path.join(recent_data_dir, f'freq_bounds_device_{device_num}.csv')
        
        if not os.path.exists(recent_data_file):
            return None, None
        
        data = pd.read_csv(recent_data_file)
        data['time'] = pd.to_datetime(data['time'])
        
        bounds = None
        if os.path.exists(bounds_file):
            bounds = pd.read_csv(bounds_file)
        
        return data, bounds
    except Exception as e:
        st.error(f"Error loading data for device {device_num}: {str(e)}")
        return None, None

# Function to create plot
def create_plot(data, bounds, device_num):
    fig = go.Figure()
    
    # Add raw frequency data
    fig.add_trace(go.Scatter(
        x=data['time'],
        y=data['freq'],
        mode='lines+markers',
        name='Raw Frequency',
        line=dict(color='blue', width=1),
        marker=dict(size=4)
    ))
    
    # Add filtered frequency data
    fig.add_trace(go.Scatter(
        x=data['time'],
        y=data['freq_filtered'],
        mode='lines',
        name='Filtered Frequency',
        line=dict(color='red', width=2)
    ))
    
    # Add bound fills if bounds exist
    if bounds is not None:
        min_freq = bounds['min_freq'].iloc[0]
        max_freq = bounds['max_freq'].iloc[0]
        
        # Add bound lines
        fig.add_hline(y=min_freq, line_dash="dash", line_color="red", name="Lower Bound")
        fig.add_hline(y=max_freq, line_dash="dash", line_color="red", name="Upper Bound")
        
        # Add fills
        y_range = [data['freq_filtered'].min(), data['freq_filtered'].max()]
        
        # Fill between bounds (green)
        fig.add_hrect(y0=min_freq, y1=max_freq,
                     fillcolor="lightgreen", opacity=0.2,
                     layer="below", name="Normal Range")
        
        # Fill below min (red)
        fig.add_hrect(y0=y_range[0], y1=min_freq,
                     fillcolor="red", opacity=0.2,
                     layer="below", name="Below Range")
        
        # Fill above max (red)
        fig.add_hrect(y0=max_freq, y1=y_range[1],
                     fillcolor="red", opacity=0.2,
                     layer="below", name="Above Range")
    
    # Set y-axis limits from bounds if available
    y_min = None
    y_max = None
    if bounds is not None and not bounds.empty:
        if 'y_min' in bounds.columns and 'y_max' in bounds.columns:
            y_min = bounds['y_min'].iloc[0]
            y_max = bounds['y_max'].iloc[0]

    fig.update_layout(
        title=f"Device {device_num} Frequency Data",
        xaxis_title="Time",
        yaxis_title="Frequency (RPM)",
        height=400,
        showlegend=True,
        yaxis=dict(
            range=[y_min, y_max] if y_min is not None and y_max is not None else None
        )
    )
    
    return fig

# Main app layout
st.sidebar.header("Controls")

# Auto-refresh interval
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=1,
    max_value=60,
    value=1
)

# Get available devices
if use_s3:
    s3_client = get_s3_client()
    if s3_client:
        try:
            response = s3_client.list_objects_v2(
                Bucket=s3_bucket,
                Prefix=f"{s3_prefix.rstrip('/')}/recent_data_device_"
            )
            device_files = [obj['Key'] for obj in response.get('Contents', [])]
            device_numbers = [int(f.split('_')[-1].split('.')[0]) for f in device_files]
        except Exception as e:
            st.error(f"Error listing S3 objects: {str(e)}")
            device_numbers = []
    else:
        device_numbers = []
else:
    device_files = glob(os.path.join(recent_data_dir, 'recent_data_device_*.csv'))
    device_numbers = [int(f.split('_')[-1].split('.')[0]) for f in device_files]

if not device_numbers:
    st.warning("No device data found in the recent data directory.")
else:
    # Create columns for metrics
    cols = st.columns(len(device_numbers))
    
    # Create plots for each device
    for i, device_num in enumerate(device_numbers):
        try:
            data, bounds = load_data(device_num)
            
            if data is not None and not data.empty:
                with cols[i]:
                    try:
                        # Display current frequency and delay
                        current_freq = data['freq_filtered'].iloc[-1]
                        last_time = data['time'].iloc[-1]
                        delay = (pd.Timestamp.now() - pd.to_datetime(last_time)).total_seconds()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                f"Device {device_num} Current Frequency",
                                f"{current_freq:.1f} RPM"
                            )
                        with col2:
                            st.metric(
                                "Delay",
                                f"{delay:.1f} seconds"
                            )
                        
                        # Check if frequency is within bounds
                        if bounds is not None and not bounds.empty:
                            min_freq = bounds['min_freq'].iloc[0]
                            max_freq = bounds['max_freq'].iloc[0]
                            freq_out_of_bounds = current_freq < min_freq or current_freq > max_freq
                            if freq_out_of_bounds:
                                st.error("⚠️ Frequency out of bounds!")
                                warning_beep.play()
                            else:
                                st.success("✅ Frequency within bounds")
                        
                        # Create and display plot
                        fig = create_plot(data, bounds, device_num)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error processing data for Device {device_num}: {str(e)}")
            else:
                with cols[i]:
                    st.warning(f"No valid data available for Device {device_num}")
        except Exception as e:
            with cols[i]:
                st.error(f"Error processing Device {device_num}: {str(e)}")
    
    # Auto-refresh
    time.sleep(refresh_interval)
    # st.experimental_rerun()
    st.rerun()
