import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Tuple
from mouse import MouseDay
import interp
import src.IO as io

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
    axes[0, 0].plot(mouse_day.cal_tseries)
    axes[0, 1].plot(mouse_day.kin_tseries)
    axes[1, 0].plot(mouse_day.cal_tstamp_dict[event_key])
    axes[1, 1].plot(mouse_day.kin_tstamp_dict[event_key])
    return fig

def plot_r2_scores(behaviors: dict[int, str], beh_models_scores: list[list[float]], general_model_score: list[float], figsize: Tuple[int, int]=(16, 10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    # Calculate means and standard deviations for behavior models
    beh_means = [np.mean(scores) for scores in beh_models_scores]
    beh_stds = [np.std(scores) for scores in beh_models_scores]
    
    # Calculate mean and standard deviation for general model
    general_mean = np.mean(general_model_score)
    general_std = np.std(general_model_score)
    
    # Prepare data for plotting
    labels = list(behaviors.values()) + ["General Model"]
    means = beh_means + [general_mean]
    stds = beh_stds + [general_std]
    
    # Create bar plot with error bars
    bars = ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['skyblue'] * len(behaviors) + ['orange'])
    
    # Customize the plot
    ax.set_ylim(-1, 1)
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Model")
    ax.set_title("Model Performance: Mean R² Scores with Standard Deviations")
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if there are many behaviors
    if len(labels) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_kin_predictions(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    """
    True and pred positions are shape (nsamples, 4)
    Each column holds the following data:
    cam1_x cam1_y cam2_x cam2_y
    """
    w = io.load_decoded_weights(mouse_day.mouseID, mouse_day.day)
    X = mouse_day.get_trimmed_spks()
    true_positions = mouse_day.get_trimmed_avg_locs()
    pred_positions = X @ w

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    trimmed_tstamps = mouse_day.get_trimmed_cal_tstamps()
    # Camera 1 X positions
    axes[0, 0].plot(trimmed_tstamps, true_positions[:, 0], 'b-', label="True", linewidth=1.5)
    axes[0, 0].plot(trimmed_tstamps, pred_positions[:, 0], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[0, 0].set_title("Camera 1 - X Position")
    axes[0, 0].set_xlabel("Time (since Unix Epoch - ns)")
    axes[0, 0].set_ylabel("X Position (pixels)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Camera 1 Y positions
    axes[0, 1].plot(trimmed_tstamps, true_positions[:, 1], 'b-', label="True", linewidth=1.5)
    axes[0, 1].plot(trimmed_tstamps, pred_positions[:, 1], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[0, 1].set_title("Camera 1 - Y Position")
    axes[0, 1].set_xlabel("Time (since Unix Epoch - ns)")
    axes[0, 1].set_ylabel("Y Position (pixels)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Camera 2 X positions
    axes[1, 0].plot(trimmed_tstamps, true_positions[:, 2], 'b-', label="True", linewidth=1.5)
    axes[1, 0].plot(trimmed_tstamps, pred_positions[:, 2], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[1, 0].set_title("Camera 2 - X Position")
    axes[1, 0].set_xlabel("Time (since Unix Epoch - ns)")
    axes[1, 0].set_ylabel("X Position (pixels)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Camera 2 Y positions
    axes[1, 1].plot(trimmed_tstamps, true_positions[:, 3], 'b-', label="True", linewidth=1.5)
    axes[1, 1].plot(trimmed_tstamps, pred_positions[:, 3], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[1, 1].set_title("Camera 2 - Y Position")
    axes[1, 1].set_xlabel("Time (since Unix Epoch - ns)")
    axes[1, 1].set_ylabel("Y Position (pixels)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle("Kinematic Predictions: True vs Predicted Positions", fontsize=16, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig

def plot_kin_predictions_by_model(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    X = mouse_day.get_trimmed_spks()
    true_positions = mouse_day.get_trimmed_avg_locs()
    trimmed_tstamps = mouse_day.get_trimmed_cal_tstamps()
    model_types = list(mouse_day.BEHAVIOR_LABELS.values()) + ['general']

    fig, axes = plt.subplot_mosaic([['reach', 'grasp', 'carry'],
                                    ['non_movement_or_kept_jumping', 'fidget', 'eating'],
                                    ['grooming', 'non_behavior_event', 'general']], figsize=figsize)

    for i, model in enumerate(model_types):
        weights = io.load_decoded_weights(mouse_day.mouseID, mouse_day.day, model)
        pred_positions = X @ weights
        axes[model].plot(trimmed_tstamps, true_positions[:, 0], 'b-', label="True", linewidth=1.5)
        axes[model].plot(trimmed_tstamps, pred_positions[:, 0], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
        axes[model].set_title(f"{model} predictions")
        axes[model].set_xlabel("Time (since Unix Epoch - ns)")
        axes[model].set_ylabel("X Position (pixels)")
        axes[model].legend()
        axes[model].grid(True, alpha=0.3)
        
    # Add overall title
    fig.suptitle("Cam1 X Kinematics: True vs Predicted Positions by Behavior Model", fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    mouse_day = MouseDay('mouse25', '20240425')
    event_keys = mouse_day.seg_keys
    event_key = event_keys[0]  # Use first available key
    
    # Plotting Ridge Regression results
    fig2 = plot_kin_predictions_by_model(mouse_day)
    plt.show()

    # Create comprehensive plot
    # fig1 = plot_mouseday_data(mouse_day, event_key)
    # plt.show()

    # Test Interpolation function
    # for key in event_keys:
    #     fig4 = plot_interp_test(mouse_day, key)
    #     plt.show()