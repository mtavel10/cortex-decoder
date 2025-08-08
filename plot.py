import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
from typing import List, Dict, Tuple
from mouse import MouseDay
import decode as decode
import src.IO as io

# for now, only working with the first 4464 frames for Mouse25 20240425
# need to filter the cascade spike data before plotting
# TEMPORARY
tseries_max = 4464
N_PARTS = 14
CROSS_CLASS_MODE = [("natural", "reach"), ("natural", "grasp"), ("natural", "carry"), ("learned", "non_movement"), ("learned", "fidget"), ("learned", "eating")]
IN_CLASS_MODE = [("learned", "reach"), ("learned", "grasp"), ("learned", "carry"), ("natural", "non_movement"), ("natural", "fidget"), ("natural", "eating")]

# plt.rcParams.update({'font.size': 22}) # Sets the default font size to 12
behavior_model_types = ["reach", "grasp", "carry", "jumping", "fidget", "eating", "no behavior", "general"]

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
    ax1.plot(cal_tseries, cam1_avg[1, :], 'g-', linewidth=1.5, label='Y position', alpha=0.8)

    # Add event markers
    # for i, (frame, label) in enumerate(zip(cal_event_times, event_labels)):
    #     color = event_colors[label]
    #     frame_idx = int(frame)
        
    #     # Plot vertical lines for events
    #     # ax1.axvline(cal_tseries[frame_idx], color=color, linestyle='--', linewidth=2, alpha=0.7)
        
    #     # Add markers at kinematic positions
    #     ax1.plot(cal_tseries[frame_idx], cam1_avg[0, frame_idx], 'o', color=color, markersize=5, 
    #             markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
    #     ax1.plot(cal_tseries[frame_idx], cam1_avg[1, frame_idx], 's', color=color, markersize=5, 
    #             markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)

    ax1.set_xlim(start_time, end_time)
    ax1.set_xlabel('Unix Time (ns)')
    ax1.set_ylabel('Position (pixels)')
    ax1.set_title(f'Camera 1 - Average Hand Positions')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Camera 2 - X and Y positions over time
    ax2 = axes['right_top']
    ax2.plot(cal_tseries, cam2_avg[0, :], 'b-', linewidth=1.5, label='X position', alpha=0.8)
    ax2.plot(cal_tseries, cam2_avg[1, :], 'g-', linewidth=1.5, label='Y position', alpha=0.8)
    
    # # Add event markers
    # for i, (frame, label) in enumerate(zip(cal_event_times, event_labels)):
    #     color = event_colors[label]
    #     frame_idx = int(frame)
        
    #     # Add markers at kinematic positions
    #     ax2.plot(cal_tseries[frame_idx], cam2_avg[0, frame_idx], 'o', color=color, markersize=5, 
    #             markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
    #     ax2.plot(cal_tseries[frame_idx], cam2_avg[1, frame_idx], 's', color=color, markersize=5, 
    #             markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
    
    ax2.set_xlim(start_time, end_time)
    ax2.set_xlabel('Unix Time (ns)')
    ax2.set_ylabel('Position (pixels)')
    ax2.set_title(f'Camera 2 - Average Hand Positions')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
     
    # Subplot 3: Calcium Spikes Heat Map
    ax3 = axes['bottom']
    
    # Normalize the data ??
    clipped_cal_spikes = cal_spikes[32:-32, 32:-32]
    n_neurons, n_frames = clipped_cal_spikes.shape

    min_spk = np.min(clipped_cal_spikes)
    max_spk = np.max(clipped_cal_spikes)

    im = ax3.imshow(clipped_cal_spikes, norm=mcolors.Normalize(min_spk, max_spk))

    ax3.margins(1)
    ax3.set_xlim(0, n_frames-1)
    ax3.set_ylim(32, n_neurons-1)
    ax3.set_xlabel('Timebins')
    ax3.set_ylabel('Neurons')
    ax3.set_title(f'Calcium Spikes')

    # Overall title
    fig.suptitle(f'Data Collected for {mouse_day.mouseID} on {mouse_day.day}', 
                fontsize=16, y=0.98)
    fig.subplots_adjust(hspace=0.001)
    plt.tight_layout()
    return fig

def plot_spikes(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    fig, ax3 = plt.subplots(1, 1, figsize=figsize)
    cal_spikes = mouse_day.get_trimmed_spks().T
    # take the first 100 frames
    cal_spikes = cal_spikes[:, :500]
    print(cal_spikes.shape)
    
    n_neurons, n_frames = cal_spikes.shape

    heatmap = ax3.imshow(cal_spikes, cmap="inferno", norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(heatmap, ax=ax3)
    cbar.set_label("normalized inferred firing rate", rotation=270, labelpad=25)


    ax3.margins(1)
    ax3.set_xlim(32, n_frames-1)
    ax3.set_ylim(0, n_neurons-1)
    ax3.set_xlabel('Timebins')
    ax3.set_ylabel('Neurons')
    ax3.set_title(f'Inferred Calcium Spike Probabilities')

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
    plt.rcParams.update({'font.size': 12})
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
    # print(cam1_interp)
    # print(cam2_interp)
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


def plot_kin_predictions(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    """
    True and pred positions are shape (nsamples, 4)
    Each column holds the following data:
    cam1_x cam1_y cam2_x cam2_y
    """
    scores, pred_positions = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, "general")
    true_positions = mouse_day.get_trimmed_avg_locs()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    trimmed_tstamps = mouse_day.get_trimmed_cal_tstamps()
    # Camera 1 X positions
    axes[0, 0].plot(trimmed_tstamps, true_positions[:, 0], 'b-', label="True", linewidth=1.5)
    axes[0, 0].plot(trimmed_tstamps, pred_positions[:, 0], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[0, 0].set_title("Camera 1 - X Position")
    axes[0, 0].set_xlabel("Time (since Unix Epoch - ns)")
    axes[0, 0].set_ylabel("X Position (pixels)")
    axes[0, 0].legend(fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Camera 1 Y positions
    axes[0, 1].plot(trimmed_tstamps, true_positions[:, 1], 'b-', label="True", linewidth=1.5)
    axes[0, 1].plot(trimmed_tstamps, pred_positions[:, 1], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[0, 1].set_title("Camera 1 - Y Position")
    axes[0, 1].set_xlabel("Time (since Unix Epoch - ns)")
    axes[0, 1].set_ylabel("Y Position (pixels)")
    axes[0, 1].legend(fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Camera 2 X positions
    axes[1, 0].plot(trimmed_tstamps, true_positions[:, 2], 'b-', label="True", linewidth=1.5)
    axes[1, 0].plot(trimmed_tstamps, pred_positions[:, 2], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[1, 0].set_title("Camera 2 - X Position")
    axes[1, 0].set_xlabel("Unix Time (ns)")
    axes[1, 0].set_ylabel("X Position (pixels)")
    axes[1, 0].legend(fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Camera 2 Y positions
    axes[1, 1].plot(trimmed_tstamps, true_positions[:, 3], 'b-', label="True", linewidth=1.5)
    axes[1, 1].plot(trimmed_tstamps, pred_positions[:, 3], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[1, 1].set_title("Camera 2 - Y Position")
    axes[1, 1].set_xlabel("Unix Time (ns)")
    axes[1, 1].set_ylabel("Y Position (pixels)")
    axes[1, 1].legend(fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f"True vs Predicted Average Mouse-Paw Positions \n {mouse_day.mouseID}, {mouse_day.day}", fontsize=16, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig


def plot_kin_predictions_vertical(mouse_day: MouseDay, figsize: Tuple[int, int]=(8, 19)):
    """
    True and pred positions are shape (nsamples, 4)
    Each column holds the following data:
    cam1_x cam1_y cam2_x cam2_y
    """
    scores, pred_positions = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, "general")
    true_positions = mouse_day.get_trimmed_avg_locs()
    
    # Changed to 4 rows, 1 column for vertical stacking
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize)
    trimmed_tstamps = mouse_day.get_trimmed_cal_tstamps()
    
    # Create uniform x-axis range
    x_uniform = np.linspace(0, 1, len(trimmed_tstamps))
    
    # Camera 1 X positions
    axes[0].plot(x_uniform, true_positions[:, 0], 'b-', linewidth=1.5)
    axes[0].plot(x_uniform, pred_positions[:, 0], 'r-', linewidth=1.5, alpha=0.5)
    axes[0].set_title("Camera 1 - X Position")
    axes[0].grid(True, alpha=0.3)
    
    # Camera 1 Y positions
    axes[1].plot(x_uniform, true_positions[:, 1], 'b-', linewidth=1.5)
    axes[1].plot(x_uniform, pred_positions[:, 1], 'r-', linewidth=1.5, alpha=0.5)
    axes[1].set_title("Camera 1 - Y Position")
    axes[1].grid(True, alpha=0.3)
    
    # Camera 2 X positions
    axes[2].plot(x_uniform, true_positions[:, 2], 'b-', linewidth=1.5)
    axes[2].plot(x_uniform, pred_positions[:, 2], 'r-', linewidth=1.5, alpha=0.5)
    axes[2].set_title("Camera 2 - X Position")
    axes[2].grid(True, alpha=0.3)
    
    # Camera 2 Y positions
    axes[3].plot(x_uniform, true_positions[:, 3], 'b-', linewidth=1.5)
    axes[3].plot(x_uniform, pred_positions[:, 3], 'r-', linewidth=1.5, alpha=0.5)
    axes[3].set_title("Camera 2 - Y Position")
    axes[3].grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    return fig


def plot_predictions_notrue(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    """ For the poster """
    scores, pred_positions = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, "general")

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize)
    trimmed_tstamps = mouse_day.get_trimmed_cal_tstamps()
    # Camera 1 X positions
    axes[3].plot(trimmed_tstamps, pred_positions[:, 0], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[3].set_ylabel("Cam 1 \n X Position")
    axes[3].grid(True, alpha=0.3)
    axes[3].tick_params(axis='x', labelbottom=False)
    axes[3].tick_params(axis='y', labelleft=False)
    
    # Camera 1 Y positions
    axes[2].plot(trimmed_tstamps, pred_positions[:, 1], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[2].set_ylabel("Cam 1 \n Y Position")
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', labelbottom=False)
    axes[2].tick_params(axis='y', labelleft=False)
    
    # Camera 2 X positions
    axes[1].plot(trimmed_tstamps, pred_positions[:, 2], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[1].set_ylabel("Cam 2 \n X Position")
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', labelbottom=False)
    axes[1].tick_params(axis='y', labelleft=False)
    
    # Camera 2 Y positions
    axes[0].plot(trimmed_tstamps, pred_positions[:, 3], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes[0].set_ylabel("Cam 2 \n Y Position")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', labelbottom=False)
    axes[0].tick_params(axis='y', labelleft=False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()


def plot_kin_predictions_simplified(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    """
    True and pred positions are shape (nsamples, 4)
    Each column holds the following data:
    cam1_x cam1_y cam2_x cam2_y
    """
    scores, pred_positions = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, "general")
    true_positions = mouse_day.get_trimmed_avg_locs()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    trimmed_tstamps = mouse_day.get_trimmed_cal_tstamps()
    # Camera 1 X positions
    axes.plot(trimmed_tstamps, true_positions[:, 0], 'b-', label="True", linewidth=1.5)
    axes.plot(trimmed_tstamps, pred_positions[:, 0], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
    axes.set_title("Camera 1 - X Position")
    axes.set_xlabel("Time (since Unix Epoch - ns)")
    axes.set_ylabel("X Position (pixels)")
    axes.legend()
    axes.grid(True, alpha=0.3)
   
    # Add overall title
    fig.suptitle(f"True vs Predicted Average Mouse-Paw Positions \n {mouse_day.mouseID}, {mouse_day.day}", fontsize=16, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig

def plot_kin_predictions_by_model(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    KIN_DIMS = 4
    true_positions = mouse_day.get_trimmed_avg_locs()
    trimmed_tstamps = mouse_day.get_trimmed_cal_tstamps()
    model_types = list(mouse_day.BEHAVIOR_LABELS.values()) + ['general']
    figs = []

    for v in range(0, KIN_DIMS):
        print(v)
        fig, axes = plt.subplot_mosaic([['reach', 'grasp', 'carry'],
                                        ['non_movement_or_kept_jumping', 'fidget', 'eating'],
                                        ['grooming', 'non_behavior_event', 'general']], figsize=figsize)

        for i, model in enumerate(model_types):
            scores, pred_positions = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model)
            axes[model].plot(trimmed_tstamps, true_positions[:, v], 'b-', label="True", linewidth=1.5)
            axes[model].plot(trimmed_tstamps, pred_positions[:, v], 'r-', label="Predicted", linewidth=1.5, alpha=0.5)
            axes[model].set_title(f"{model} predictions")
            axes[model].set_xlabel("Time (since Unix Epoch - ns)")
            if (v % 2 == 0):
                axes[model].set_ylabel("X Position (pixels)")
            else:
                axes[model].set_ylabel("Y Position (pixels)")
            axes[model].legend()
            axes[model].grid(True, alpha=0.3)
            
        # Add overall title
        if (v == 0):
            title = "Cam1 X Kinematics"
        elif (v == 1):
            title = "Cam1 Y Kinematics"
        elif (v == 2):
            title = "Cam2 X Kinematics"
        elif (v == 3):
            title = "Cam2 Y Kinematics"

        fig.suptitle(f"{title}: True vs Predicted Positions by Behavior Model", fontsize=16, fontweight='bold')
        fig.tight_layout()
        figs.append(fig)

    return figs

def plot_predictions(mouse_day: MouseDay, train_type: str, test_type: str=None, figsize: Tuple[int, int]=(16, 10)):
    """
    Plots the 4 average kinematic predictions against the true locations (camera 1 x/y, camera 2 x/y). 
    Parameters
        mouse_day
        train_type: str
            data category that the model was trained on ("general", "inhibitory", "excitatory", "reach", "fidget", etc)
        test_type: str
            data category that the model was tested on (None if we aren't cross-testing)
            If None, the training data type is the same as the testing data type
    """
    KIN_DIMS = ["Cam1 X", "Cam1 Y", "Cam2 X", "Cam2 Y"]
    true_positions = mouse_day.get_trimmed_avg_locs()
    trimmed_tstamps = mouse_day.get_trimmed_cal_tstamps()
    figs = []

    for d, dim in enumerate(KIN_DIMS):
        # get it to work for one train-test type, then expand to plot multiple model/test combos at once
        fig, axes = plt.subplots(figsize=figsize)
        
        mode = ""
        line_type = ""
        test_data = ""
        if (test_type is None): # default: evaluating model performance on its own data
            mode = f"{train_type}"
            test_data = train_type
            line_type = "r-"
            print("YAY")
        else: # valuating model performance on another set of data
            mode = f"{train_type}_x_{test_type}"
            test_data = test_type
            line_type = "ro" # behavior predictions are erratic
            print("cross")

        print(mode)

        sc, pred_positions = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type=mode)
        axes.plot(trimmed_tstamps, true_positions[:, d], 'b-', label="True", linewidth=1.5)
        axes.plot(trimmed_tstamps, pred_positions[:, d], line_type, label="Predicted", linewidth=1.5, alpha=0.5)
        axes.set_title(f"{train_type} Model's {dim} Predictions on {test_data} data")
        axes.set_xlabel(f"{dim} Coordinates")
        axes.set_ylabel(f"Unix Time (ns)")
        figs.append(fig)
    
    plt.tight_layout()

    return 


def plot_model_performance(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    behaviors = mouse_day.BEHAVIOR_LABELS
    model_types = list(behaviors.values()) + ["general"]
    model_types.remove("grooming")

    model_scores: list[list[float]] = []
    for model in model_types:
        scores, preds = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model)
        model_scores.append(scores)
    
    # Calculate means and standard deviations for behavior models
    means = [np.mean(scores) for scores in model_scores]
    stds = [np.std(scores) for scores in model_scores]
    
    # Create bar plot with error bars
    bars = ax.bar(model_types, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['skyblue'] * (len(behaviors) - 1) + ['orange'])
    
    # Customize the plot
    ax.set_ylim(-1, 1)
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Model")
    ax.set_title("Model Performance: Mean R² Scores with Standard Deviations")
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if there are many behaviors
    if len(model_types) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_model_performance_swarm(mouse_day: MouseDay, model_name="ridge", figsize: Tuple[int, int]=(16, 10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Shitty workaround until i fix the behavior labels in the mouseday class
    behaviors = {key: value for key, value in mouse_day.BEHAVIOR_LABELS.items() if key != 6}
    model_types = list(behaviors.values())
    # model_types.remove('grooming')
    model_types.append('general')

    x_positions = []
    all_scores = []
    for i, model in enumerate(model_types):
        scores, preds = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type=f"{model}_{model_name}")
        
        # add jitter for better visibility
        jitter = np.random.normal(0, 0.005, len(scores))
        x_positions = [i] * len(scores) + jitter
        ax.scatter(x_positions, scores, alpha=0.5, s=1000)

        all_scores.append(scores)
    
    means = [np.mean(scores) for scores in all_scores]
    stds = [np.std(scores) for scores in all_scores]

      
    ax.errorbar(range(len(model_types)), means, xerr=0.2, fmt='.k', linewidth=3, alpha=0.8)
    ax.errorbar(range(len(model_types)), means, yerr=stds, fmt='.k', capsize=5, capthick=1.5, linewidth=1.5, alpha=0.8)
    

    ax.set_xticks(range(len(behavior_model_types)))
    ax.set_xticklabels(behavior_model_types, rotation=45, ha='right', fontsize=15)
    ax.set_xlabel("Model", fontsize=18)
    ax.set_ylabel("R² Score", fontsize=18)
    ax.set_title(f"Performance by Model - {model_name}", fontsize=30)
    ax.set_ylim(-1, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.5, len(model_types) - 0.5)

    plt.tight_layout()
    
    return fig

def plot_general_performance_by_beh(mouse_day: MouseDay, figsize: Tuple[int, int]=(16,10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    scores_by_beh = io.load_scores_by_beh(mouse_day.mouseID, mouse_day.day)
    behaviors = list(mouse_day.BEHAVIOR_LABELS.values())
    # behaviors.remove('grooming')
    
    # Calculate means and standard deviations for behavior scores
    means = [np.mean(scores) for scores in scores_by_beh.values()]
    stds = [np.std(scores) for scores in scores_by_beh.values()]

    # Create bar plot with error bars
    bars = ax.bar(behaviors, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['skyblue'])
    
    # Customize the plot
    ax.set_ylim(-2.5, 1)
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Model")
    ax.set_title("General Model Performance on Specific Behaviors")
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_cell_performance(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ex_scores, ex_preds = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="excitatory")
    in_scores, in_preds = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="inhibitory")

    means = [np.mean(ex_scores), np.mean(in_scores)]
    stds = [np.std(ex_scores), np.std(in_scores)]

    # Create bar plot with error bars
    bars = ax.bar(["excitatory", "inhibitory"], means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['skyblue'])
    
    # Customize the plot
    ax.set_ylim(0, 0.2)
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Model")
    ax.set_title("Excitatory vs Inhibitory Model Performance")
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_cell_performance_swarm(mouse_day: MouseDay, figsize: Tuple[int, int]=(10, 10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    # Load data
    ex_scores, ex_preds = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="excitatory")
    in_scores, in_preds = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="inhibitory")
    
    cell_types = ["excitatory", "inhibitory"]
    all_scores = [ex_scores, in_scores]
    
    # Create swarm plot with jitter
    for i, (cell_type, scores) in enumerate(zip(cell_types, all_scores)):
        # Add jitter for better visibility
        jitter = np.random.normal(0, 0.05, len(scores))
        x_positions = np.array([i] * len(scores)) + jitter
        ax.scatter(x_positions, scores, alpha=0.5, s=500)
    
    # Calculate means and stds
    means = [np.mean(scores) for scores in all_scores]
    stds = [np.std(scores) for scores in all_scores]
    
    # Horizontal error bars for visual separation
    ax.errorbar(range(len(cell_types)), means, xerr=0.2, fmt='.k', linewidth=3, alpha=0.8)
    # Vertical error bars for standard deviation
    ax.errorbar(range(len(cell_types)), means, yerr=stds, fmt='.k', capsize=5, capthick=1.5, linewidth=1.5, alpha=0.8)
    
    # Customize the plot
    ax.set_xticks([0, 1])
    ax.set_xticklabels(cell_types, ha='center', fontsize=18)
    ax.set_ylabel("R² Score", fontsize=18)
    ax.set_xlabel("Model", fontsize=18)
    ax.set_title("Model Performance by Cell Type", fontsize=30)
    ax.set_ylim(0, 0.15)
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_beh_class_performance(mouse_day: MouseDay, figsize: Tuple[int, int]=(16, 10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    learned_scores, p = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="learned_class")
    natural_scores, p = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="natural_class")
    learned_xtested_scores, p = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="learned_x_natural")
    natural_xtested_scores, p = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="natural_x_learned")

    means = [np.mean(learned_scores), np.mean(learned_xtested_scores), np.mean(natural_scores), np.mean(natural_xtested_scores)]
    stds = [np.std(learned_scores), np.std(learned_xtested_scores), np.std(natural_scores), np.std(natural_xtested_scores)]

    colors = ['skyblue', 'orange', 'lightgreen', 'orange']  # cross-tested bars are orange
    labels = ["Learned Behavior Model", "Learned Behavior Model (Cross-Tested)",
              "Natural Behavior Model", "Natural Behavior Model (Cross-Tested)"]

    # Create bar plot with error bars
    bars = ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=colors)

    # Create legend
    legend_elements = [
        Patch(facecolor='skyblue', alpha=0.7, label='"Learned" Behavior Class: Reach, Carry, Grasp'),
        Patch(facecolor='lightgreen', alpha=0.7, label='"Natural" Behavior Class: Non-movement, Fidget, Eat, Groom'),
        Patch(facecolor='orange', alpha=0.7, label='Cross-tested (train class ≠ test class)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Customize the plot
    ax.set_ylim(-1, 1)
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Model")
    ax.set_title("Behavior Class Model Performance: Learned vs. Natural, Cross-Tested")
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_cross_beh_performance(mouse_day: MouseDay, behaviors: list[str], figsize: Tuple[int, int]=(16, 10)):
    """
    Plots each individual behavior, cross-tested on a model trained on the opposite behavior "class" 
    (i.e. a model trained on "natural" beheavior data was cross tested on each individual "learned" behavior)
    """

    class_colors = {"learned": "lightsteelblue", "natural": "rosybrown"}

    scores_list = []
    num_natural = 0
    num_learned = 0
    for beh in behaviors:
        # determine the "cross class": for this behavior
        if decode.BEH_CLASSES[beh][0] in decode.BEH_CLASSES["natural"]:
            cross_class = "learned"
            num_natural += 1
        else:
            cross_class = "natural"
            num_learned += 1
        scores, preds = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type=f"{cross_class}_x_{beh}")
        scores_list.append(scores)
    colors = [class_colors["learned"]] * num_learned + [class_colors["natural"]] * num_natural
    
    # Calculate means and standard deviations
    means = [np.mean(scores) for scores in scores_list]
    stds = [np.std(scores) for scores in scores_list]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up x positions
    x_pos = np.arange(len(behaviors))
    
    # Create bars
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
     # Create legend
    legend_elements = [
        Patch(facecolor='lightsteelblue', alpha=0.7, label='"Natural" Model Performance on Learned Behaviors'),
        Patch(facecolor='rosybrown', alpha=0.7, label='"Learned" Model Performance on Natural Behaviors'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Customize the plot
    ax.set_xlabel('Behavior Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title("Cross-Tested Behavior Model Performance", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(behaviors, rotation=45, ha='right')
    # ax.set_ylim(-15, 0.25)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig

def plot_performance(mouse_day: MouseDay, modes: list[Tuple[str, str|None]], mode_type: str, figsize: Tuple[int, int]=(16, 10)):
    """
    Plots performance of different models dynamically based on the train type and test type (if test type is None, then the training data was the same type as the testing data)
    Parameters
        modes : List[Tuple[str | None]]
    
    """

    scores_list = []
    model_labels = []
    for mode in modes:
        if mode[1] == None: # default: training data class is the same as testing data class
            scores_list.append(io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type=mode[0])[0])
            model_labels.append(f"{mode[0]} \n x \n {mode[0]}")
        else:
            scores_list.append(io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type=f"{mode[0]}_x_{mode[1]}")[0])
            model_labels.append(f"{mode[0]} \n x \n {mode[1]}")
    
    # Calculate means and standard deviations
    means = [np.mean(scores) for scores in scores_list]
    stds = [np.std(scores) for scores in scores_list]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    colors = ["skyblue", "lightgreen"] # go back and customize to work dynamically depending on the type of train/test
    x_pos = np.arange(len(modes))
     # Create bars
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=[colors[0]] * 3 + [colors[1]] * 3, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Mode (train x test)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title(f"Behavior Model Performance - {mode_type}", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, rotation=45, ha='right')

     # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    
    plt.tight_layout()
    return fig

def plot_performance_swarm(mouse_day: MouseDay, modes: list[Tuple[str, str|None]], mode_type: str, figsize: Tuple[int, int]=(16, 10)):
    """
    Plots performance of different models dynamically based on the train type and test type (if test type is None, then the training data was the same type as the testing data)
    Parameters
    modes : List[Tuple[str | None]]
    """
    scores_list = []
    model_labels = []
    train_types = []
    
    for mode in modes:
        if mode[1] == None:  # default: training data class is the same as testing data class
            scores_list.append(io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type=mode[0])[0])
            model_labels.append(f"{mode[0]} \n x \n {mode[0]}")
            train_types.append(mode[0])
        else:
            scores_list.append(io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type=f"{mode[0]}_x_{mode[1]}")[0])
            model_labels.append(f"{mode[0]} \n x \n {mode[1]}")
            train_types.append(mode[0])
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    # Create color map for train types
    unique_train_types = list(set(train_types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_train_types)))
    train_color_map = {train_type: colors[i] for i, train_type in enumerate(unique_train_types)}
    
    # Create swarm plot with jitter, colored by train type
    for i, (scores, train_type) in enumerate(zip(scores_list, train_types)):
        # Add jitter for better visibility
        jitter = np.random.normal(0, 0.05, len(scores))
        x_positions = np.array([i] * len(scores)) + jitter
        ax.scatter(x_positions, scores, alpha=0.5, s=500, 
                  color=train_color_map[train_type], label=train_type if i == train_types.index(train_type) else "")
    
    # Calculate means and standard deviations
    means = [np.mean(scores) for scores in scores_list]
    stds = [np.std(scores) for scores in scores_list]
    
    # Define cohesive colors for error bars
    positive_color = '#2E7D32'  # Dark green that works well with tab10 palette
    negative_color = '#C62828'  # Dark red that works well with tab10 palette
    
    # Plot error bars with colors based on mean values
    for i, (mean, std) in enumerate(zip(means, stds)):
        error_color = positive_color if mean >= 0 else negative_color
        
        # Horizontal error bars for visual separation
        ax.errorbar([i], [mean], xerr=0.4, fmt='.', color=error_color, 
                   linewidth=5, alpha=0.8, markersize=8)
        # Vertical error bars for standard deviation
        # # ax.errorbar([i], [mean], yerr=std, fmt='.', color='k', 
        #            capsize=5, capthick=2, linewidth=2, alpha=0.8, markersize=8)
    
    # Customize the plot
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(model_labels, ha='center', fontsize=15)
    ax.set_xlabel('Mode (train x test)', fontsize=18)
    ax.set_ylabel('R² Score', fontsize=18)
    ax.set_title(f"{mode_type}", fontsize=30)
    ax.set_xlim(-0.5, len(modes) - 0.5)  # Make categories closer together
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend for train types
   #  ax.legend(title="Train Class", bbox_to_anchor=(1.05, 1), loc='best', fontsize=15)
    
    plt.tight_layout()
    return fig

def simple_crossday_performance(day1: MouseDay, day2: MouseDay, figsize=(16, 10)):
    # def a way to do this dynamically with just a list of days
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    d1_scores, d1_preds = io.load_decoded_data(day1.mouseID, day1.day, "registered_general")
    d2_scores, d2_preds = io.load_decoded_data(day2.mouseID, day2.day, "registered_general")
    d1xd2_scores, preds = io.load_decoded_data(day1.mouseID, day1.day, f"{day1.day}_x_{day2.day}")
    d2xd1_scores, preds = io.load_decoded_data(day2.mouseID, day2.day, f"{day2.day}_x_{day1.day}")

    cross_scores = [[np.mean(d1xd2_scores), np.mean(d2_scores)], 
                    [np.mean(d1_scores), np.mean(d2xd1_scores)]]
    
    im = ax.imshow(cross_scores)

    # Add axis labels
    ax.set_ylabel('Test Day', fontsize=12)
    ax.set_xlabel('Train Day', fontsize=12)
    
    # Add tick labels to make it clearer which day is which
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Day {day1.day}', f'Day {day2.day}'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f'Day {day2.day}', f'Day {day1.day}'])
    
    # Add a colorbar for the scores
    plt.colorbar(im, ax=ax, label='Mean Score')

    # Set text to indicate scores
    for i in range(0, 2):
        for j in range(0, 2):
            text = ax.text(j, i, cross_scores[i][j], ha="center", va="center", color="k")
    
    # Add title for clarity
    ax.set_title(f'Cross-day Performance\nMouse {day1.mouseID}', fontsize=14)

    return fig

def plot_decoded_data(mouse_day: MouseDay):
    """
    Just to make sure all the mice are mice-ing. 
    Runs EVERYTHING.
    Saves if we specify. 
    """

    fig = plot_interp_test(mouse_day, mouse_day.seg_keys[0])
    fig1 = plot_kin_predictions(mouse_day)
    fig2 = plot_model_performance_swarm(mouse_day)
    fig3 = plot_general_performance_by_beh(mouse_day)
    fig4 = plot_cell_performance_swarm(mouse_day)
    fig5 = plot_performance_swarm(mouse_day, modes=IN_CLASS_MODE, mode_type="In-Class")
    fig6 = plot_performance_swarm(mouse_day, modes=CROSS_CLASS_MODE, mode_type="Cross-Class")

    plt.show()
    return 0


def plot_performance_by_lag(mouse_day: MouseDay, min_lag: int, max_lag: int, figsize=(16, 10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    base_scores, p = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type="general_ridge")
    all_scores = []
    all_scores.append(base_scores)
    for i in range(min_lag, max_lag+1):
        s, p = io.load_decoded_data(mouse_day.mouseID, mouse_day.day, model_type=f"general_ridge_l{i}")
        all_scores.append(s)
    
    # Create color map for train types
    lag_list = [0] + list(range(min_lag, max_lag+1))
    colors = plt.cm.tab10(np.linspace(0, 1, len(lag_list)))
    color_map = {lag: colors[i] for i, lag in enumerate(lag_list)}
    
    # Create swarm plot with jitter, colored by train type
    for i, (scores, lag) in enumerate(zip(all_scores, lag_list)):
        # Add jitter for better visibility
        jitter = np.random.normal(0, 0.05, len(scores))
        x_positions = np.array([i] * len(scores)) + jitter
        ax.scatter(x_positions, scores, alpha=0.5, s=500, 
                  color=color_map[lag])
    
    # Calculate means and standard deviations
    means = [np.mean(scores) for scores in all_scores]
    stds = [np.std(scores) for scores in all_scores]
    
    # Define cohesive colors for error bars
    positive_color = '#2E7D32'  # Dark green that works well with tab10 palette
    negative_color = '#C62828'  # Dark red that works well with tab10 palette
    
    # Plot error bars with colors based on mean values
    for i, (mean, std) in enumerate(zip(means, stds)):
        error_color = positive_color if mean >= 0 else negative_color
        
        # Horizontal error bars for visual separation
        ax.errorbar([i], [mean], xerr=0.4, fmt='.', color=error_color, 
                   linewidth=5, alpha=0.8, markersize=8)
        # Vertical error bars for standard deviation
        # # ax.errorbar([i], [mean], yerr=std, fmt='.', color='k', 
        #            capsize=5, capthick=2, linewidth=2, alpha=0.8, markersize=8)
    
    # Customize the plot
    ax.set_xticks(range(len(lag_list)))
    ax.set_xticklabels(lag_list, ha='center', fontsize=15)
    ax.set_xlabel('Lag size (frames)', fontsize=18)
    ax.set_ylabel('R² Score', fontsize=18)
    ax.set_title(f"Performance by Lag - with 'non-lagged' labels", fontsize=30)
    ax.set_xlim(-0.5, len(lag_list) - 0.5)  # Make categories closer together
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    
    return fig


if __name__ == "__main__":

    m25_april24 = MouseDay('mouse25', '20240424')
    m25_april25 = MouseDay('mouse25', '20240425')

    plot_performance_by_lag(m25_april25, 1, 8)
    plt.show()
    