import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Tuple
from mouse import MouseDay
import sys

# for now, only working with the first 4464 frames for Mouse25 20240425
# need to filter the cascade spike data before plotting
# TEMPORARY
tseries_max = 4464
N_PARTS = 14

def plot_mouseday_data(mouse_day, event_key: str, figsize: Tuple[int, int] = (16, 10)):
    """
    Plot interpolated kinematic positions and calcium spikes with event labels
    
    Parameters:
    -----------
    mouse_day : MouseDay
        Instance of the MouseDay class
    event_key : str
        Specifies which 2.5min chunk to analyze
    figsize : tuple
        Figure size (width, height)
    """
    
    # Get interpolated kinematic averages (2 cameras)
    cam1_avg, cam2_avg = mouse_day.interpolate_avgkin2cal(event_key)
    
    # Get calcium data
    cal_tseries = mouse_day.cal_tstamp_dict[event_key]
    # TEMPORARY LIMIT TO FIRST 2.5 MINUTES (4464 FRAMES)
    max_frames = len(cal_tseries)
    cal_spikes = mouse_day.cal_spks[:, :max_frames]

    start_time = cal_tseries[0]
    end_time = cal_tseries[-1]
    
    # Get event data
    cal_event_times = mouse_day.cal_event_frames
    event_labels = mouse_day.event_labels
    
    # Another temporary limit
    temp_mask = cal_event_times <= max_frames
    cal_event_times = cal_event_times[temp_mask]
    max_events = len(cal_event_times)
    event_labels = event_labels[:max_events]
    
    # Create color mapping for events
    unique_events = np.unique(event_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
    event_colors = dict(zip(unique_events, colors))
    
    # Create figure with subplots
    fig, axes = plt.subplot_mosaic([['left_top', 'right_top'], 
                                    ['bottom', 'bottom']], 
                                figsize=figsize,
                                gridspec_kw={'height_ratios': [1, 2]})
    
    # Subplot 1: Camera 1 - X and Y positions over time
    ax1 = axes['left_top']
    ax1.plot(cal_tseries, cam1_avg[0, :], 'b-', linewidth=1.5, label='X position', alpha=0.8)
    ax1.plot(cal_tseries, cam1_avg[1, :], 'r-', linewidth=1.5, label='Y position', alpha=0.8)

    # Add event markers
    for i, (frame, label) in enumerate(zip(cal_event_times, event_labels)):
        color = event_colors[label]
        frame_idx = int(frame)
        
        # Plot vertical lines for events
        # ax1.axvline(cal_tseries[frame_idx], color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        # Add markers at kinematic positions
        ax1.plot(cal_tseries[frame_idx], cam1_avg[0, frame_idx], 'o', color=color, markersize=5, 
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
        ax1.plot(cal_tseries[frame_idx], cam1_avg[1, frame_idx], 's', color=color, markersize=5, 
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)

    ax1.set_xlim(start_time, end_time)
    ax1.set_xlabel('CALCIUM Time (s)')
    ax1.set_ylabel('Position (pixels)')
    ax1.set_title(f'Camera 1 - Hand Position\n{mouse_day.mouseID} {mouse_day.day}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Camera 2 - X and Y positions over time
    ax2 = axes['right_top']
    ax2.plot(cal_tseries, cam2_avg[0, :], 'b-', linewidth=1.5, label='X position', alpha=0.8)
    ax2.plot(cal_tseries, cam2_avg[1, :], 'r-', linewidth=1.5, label='Y position', alpha=0.8)
    
    # Add event markers
    for i, (frame, label) in enumerate(zip(cal_event_times, event_labels)):
        color = event_colors[label]
        frame_idx = int(frame)
        
        # Add markers at kinematic positions
        ax2.plot(cal_tseries[frame_idx], cam2_avg[0, frame_idx], 'o', color=color, markersize=5, 
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
        ax2.plot(cal_tseries[frame_idx], cam2_avg[1, frame_idx], 's', color=color, markersize=5, 
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
    
    ax2.set_xlim(start_time, end_time)
    ax2.set_xlabel('CALCIUM Time (s)')
    ax2.set_ylabel('Position (pixels)')
    ax2.set_title(f'Camera 2 - Hand Position\n{mouse_day.mouseID} {mouse_day.day}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
     
    # Subplot 3: Calcium Spikes Heat Map
    ax3 = axes['bottom']
    
    # Normalize the data ??
    clipped_cal_spikes = cal_spikes[32:-32, 32:-32]
    n_neurons, n_frames = clipped_cal_spikes.shape

    # mean = np.mean(clipped_cal_spikes)
    # std_dev = np.std(clipped_cal_spikes)
    # z_scores = (clipped_cal_spikes - mean) / std_dev

    im = ax3.imshow(cal_spikes, cmap='jet',origin='lower', aspect='auto')

    ax3.margins(1)
    ax3.set_xlim(0, n_frames-1)
    ax3.set_ylim(0, n_neurons-1)
    ax3.set_xlabel('Frames')
    ax3.set_ylabel('Neurons')
    ax3.set_title(f'Calcium Spikes\n{mouse_day.mouseID} {mouse_day.day}')

    # Overall title
    fig.suptitle(f'Mouse Day Analysis: {mouse_day.mouseID} - {mouse_day.day} - {event_key}', 
                fontsize=16, y=0.98)
    fig.subplots_adjust(hspace=0.001)
    plt.tight_layout()
    return fig

def plot_interp_test(mouse_day, event_key: str, figsize: Tuple[int, int] = (16, 10)):
    """
    Creates 8 plots: regular kinematic data by kinematic time frames, and interpolated data by calcium time frames. 
    Test by visual inspection to make sure they're the same shape. 

            Regular         Interpolated

    Cam1    x1_plot         x1_interp_plot
            y1_plot         y1_interp_plot
    Cam2    x2_plot         x2_interp_plot
            y2_plot         y2_interp_plot
        
    """
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=figsize)
    
    # Regular Data
    kin_times = mouse_day.kin_tstamp_dict[event_key]
    max_frames = len(kin_times)
    curr_kin_mats = mouse_day.kin_mats[event_key]
    cam1_kin_mat, cam2_kin_mat = curr_kin_mats
    
    # Camera 1
    cam1_avgs = mouse_day.get_avg_coordinates(cam1_kin_mat)[:max_frames]
    x1 = cam1_avgs[:, 0]
    axes[0, 0].plot(kin_times, x1)
    axes[0, 0].set_xlabel('Kinematic Time')
    axes[0, 0].set_ylabel('X1 Coordinate')
    axes[0, 0].set_title('Camera 1 - X Coordinate (Regular)')
    
    y1 = cam1_avgs[:, 1]
    axes[1, 0].plot(kin_times, y1)
    axes[1, 0].set_xlabel('Kinematic Time')
    axes[1, 0].set_ylabel('Y1 Coordinate')
    axes[1, 0].set_title('Camera 1 - Y Coordinate (Regular)')
    
    # Camera 2
    cam2_avgs = mouse_day.get_avg_coordinates(cam2_kin_mat)[:max_frames]
    x2 = cam2_avgs[:, 0]
    axes[2, 0].plot(kin_times, x2)
    axes[2, 0].set_xlabel('Kinematic Time')
    axes[2, 0].set_ylabel('X2 Coordinate')
    axes[2, 0].set_title('Camera 2 - X Coordinate (Regular)')
    
    y2 = cam2_avgs[:, 1]
    axes[3, 0].plot(kin_times, y2)
    axes[3, 0].set_xlabel('Kinematic Time')
    axes[3, 0].set_ylabel('Y2 Coordinate')
    axes[3, 0].set_title('Camera 2 - Y Coordinate (Regular)')
    
    # Interpolated Data
    cal_times = mouse_day.cal_tstamp_dict[event_key]
    max_frames = len(cal_times)
    cam1_interp, cam2_interp = mouse_day.interpolate_avgkin2cal(event_key)
    print(cam1_interp)
    print(cam2_interp)
    cam1_interp = cam1_interp[:, :max_frames]
    cam2_interp = cam2_interp[:, :max_frames]
    
    # Camera 1
    x1_interp = cam1_interp[0, :]
    axes[0, 1].plot(cal_times, x1_interp)
    axes[0, 1].set_xlabel('Calcium Time')
    axes[0, 1].set_ylabel('X1 Coordinate')
    axes[0, 1].set_title('Camera 1 - X Coordinate (Interpolated)')
    
    y1_interp = cam1_interp[1, :]
    axes[1, 1].plot(cal_times, y1_interp)
    axes[1, 1].set_xlabel('Calcium Time')
    axes[1, 1].set_ylabel('Y1 Coordinate')
    axes[1, 1].set_title('Camera 1 - Y Coordinate (Interpolated)')
    
    # Camera 2
    x2_interp = cam2_interp[0, :]
    axes[2, 1].plot(cal_times, x2_interp)
    axes[2, 1].set_xlabel('Calcium Time')
    axes[2, 1].set_ylabel('X2 Coordinate')
    axes[2, 1].set_title('Camera 2 - X Coordinate (Interpolated)')
    
    y2_interp = cam2_interp[1, :]
    axes[3, 1].plot(cal_times, y2_interp)
    axes[3, 1].set_xlabel('Calcium Time')
    axes[3, 1].set_ylabel('Y2 Coordinate')
    axes[3, 1].set_title('Camera 2 - Y Coordinate (Interpolated)')


    # Align y-axis limits across regular and interpolated plots for each coordinate
    # Camera 1 X coordinate (row 0)
    y_min_x1 = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    y_max_x1 = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(y_min_x1, y_max_x1)
    axes[0, 1].set_ylim(y_min_x1, y_max_x1)
    
    # Camera 1 Y coordinate (row 1)
    y_min_y1 = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    y_max_y1 = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(y_min_y1, y_max_y1)
    axes[1, 1].set_ylim(y_min_y1, y_max_y1)
    
    # Camera 2 X coordinate (row 2)
    y_min_x2 = min(axes[2, 0].get_ylim()[0], axes[2, 1].get_ylim()[0])
    y_max_x2 = max(axes[2, 0].get_ylim()[1], axes[2, 1].get_ylim()[1])
    axes[2, 0].set_ylim(y_min_x2, y_max_x2)
    axes[2, 1].set_ylim(y_min_x2, y_max_x2)
    
    # Camera 2 Y coordinate (row 3)
    y_min_y2 = min(axes[3, 0].get_ylim()[0], axes[3, 1].get_ylim()[0])
    y_max_y2 = max(axes[3, 0].get_ylim()[1], axes[3, 1].get_ylim()[1])
    axes[3, 0].set_ylim(y_min_y2, y_max_y2)
    axes[3, 1].set_ylim(y_min_y2, y_max_y2)
    
    # Add overall title
    fig.suptitle(f'Kinematic Data Comparison - {event_key}', fontsize=16, y=0.98)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig

def plot_tseries_tstamps(mouse_day, event_key: str, figsize: Tuple[int, int] = (16, 10)):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    print(mouse_day.cal_tseries)
    axes[0, 0].plot(mouse_day.cal_tseries)
    axes[0, 1].plot(mouse_day.kin_tseries)
    axes[1, 0].plot(mouse_day.cal_tstamp_dict[event_key])
    axes[1, 1].plot(mouse_day.kin_tstamp_dict[event_key])
    return fig

if __name__ == "__main__":
    mouse_day = MouseDay('mouse25', '20240425')
    event_keys = mouse_day.seg_keys
    event_key = event_keys[0]  # Use first available key
    
    # Create comprehensive plot
    # fig1 = plot_mouseday_data(mouse_day, event_key)
    # plt.show()

    # Test Interpolation function
    fig4 = plot_interp_test(mouse_day, event_key)
    plt.show()

    # # Test Tstamps
    # fig3 = plot_tseries_tstamps(mouse_day, event_key)
    # plt.show()
    
    print("Example usage:")
    print("mouse_day = MouseDay('mouse25', '20240425')")
    print("event_key = list(mouse_day.kin_event_times.keys())[0]")
    print("fig = plot_mouseday_data(mouse_day, event_key)")
    print("plt.show()")