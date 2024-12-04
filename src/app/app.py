import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from glob import glob
import os
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Frequency Monitor", layout="wide")
st.title("Real-time Frequency Monitoring")

# Get data directory
base_dir = '/home/abuzarmahmood/projects/video_movement_frequency'
recent_data_dir = os.path.join(base_dir, 'artifacts', 'recent_data')

# Function to load and process data
def load_data(device_num):
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
    
    fig.update_layout(
        title=f"Device {device_num} Frequency Data",
        xaxis_title="Time",
        yaxis_title="Frequency (RPM)",
        height=400,
        showlegend=True
    )
    
    return fig

# Main app layout
st.sidebar.header("Controls")

# Auto-refresh interval
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=1,
    max_value=60,
    value=5
)

# Get available devices
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
                        # Display current frequency
                        current_freq = data['freq_filtered'].iloc[-1]
                        st.metric(
                            f"Device {device_num} Current Frequency",
                            f"{current_freq:.1f} RPM"
                        )
                        
                        # Check if frequency is within bounds
                        if bounds is not None and not bounds.empty:
                            min_freq = bounds['min_freq'].iloc[0]
                            max_freq = bounds['max_freq'].iloc[0]
                            if current_freq < min_freq or current_freq > max_freq:
                                st.error("⚠️ Frequency out of bounds!")
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
