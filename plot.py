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
    print(cam1_avg)
    
    # Get calcium data
    cal_tseries = mouse_day.cal_tseries
    # TEMPORARY LIMIT TO FIRST 2.5 MINUTES (4464 FRAMES)
    max_frames = len(cal_tseries)
    cal_spikes = mouse_day.cal_spks[:, :max_frames]

    start_time = cal_tseries[0]
    end_time = cal_tseries[-1]
    
    # Get event data
    cal_event_times = mouse_day.cal_event_times
    event_labels = mouse_day.event_labels
    
    # Another temporary limit
    temp_mask = cal_event_times <= max_frames
    cal_event_times = cal_event_times[temp_mask]
    max_events = len(cal_event_times)
    event_labels = event_labels[:max_events]
    print("cal event times: ", cal_event_times)
    print(type(cal_event_times))
    print(len(cal_event_times))
    print("event labels: ", event_labels)
    print(len(event_labels))
    
    # Create color mapping for events
    unique_events = np.unique(event_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
    event_colors = dict(zip(unique_events, colors))
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Camera 1 - X and Y positions over time
    ax1 = fig.add_subplot(gs[0, 0])
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
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (pixels)')
    ax1.set_title(f'Camera 1 - Hand Position\n{mouse_day.mouseID} {mouse_day.day}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Camera 2 - X and Y positions over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cal_tseries, cam2_avg[0, :], 'b-', linewidth=1.5, label='X position', alpha=0.8)
    ax2.plot(cal_tseries, cam2_avg[1, :], 'r-', linewidth=1.5, label='Y position', alpha=0.8)
    
    # Add event markers
    for i, (frame, label) in enumerate(zip(cal_event_times, event_labels)):
        color = event_colors[label]
        frame_idx = int(frame)
        
        # Plot vertical lines for events
        # ax2.axvline(cal_tseries[frame_idx], color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        # Add markers at kinematic positions
        ax2.plot(cal_tseries[frame_idx], cam2_avg[0, frame_idx], 'o', color=color, markersize=5, 
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
        ax2.plot(cal_tseries[frame_idx], cam2_avg[1, frame_idx], 's', color=color, markersize=5, 
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
    
    ax2.set_xlim(start_time, end_time)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (pixels)')
    ax2.set_title(f'Camera 2 - Hand Position\n{mouse_day.mouseID} {mouse_day.day}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # # Subplot 3: 2D trajectory for Camera 1
    # ax3 = fig.add_subplot(gs[0, 2])
    
    # # Plot trajectory with gradient coloring by time
    # points = np.array([cam1_avg[0, :], cam1_avg[1, :]]).T
    # for i in range(len(points) - 1):
    #     alpha = i / len(points)
    #     ax3.plot([points[i, 0], points[i+1, 0]], [points[i, 1], points[i+1, 1]], 
    #             'k-', linewidth=1, alpha=alpha * 0.5 + 0.2)
    
    # # Color-code trajectory by events
    # plotted_events = set()
    # for i, (event_time, event_label) in enumerate(zip(cal_event_times, event_labels)):
    #     color = event_colors[event_label]
    #     idx = np.argmin(np.abs(cal_tseries - event_time))
        
    #     label = event_label if event_label not in plotted_events else None
    #     if label:
    #         plotted_events.add(event_label)
        
    #     ax3.plot(cam1_avg[0, idx], cam1_avg[1, idx], 'o', color=color, markersize=10, 
    #             markerfacecolor=color, markeredgecolor='white', markeredgewidth=2, 
    #             label=label)
    
    # ax3.set_xlabel('X Position (pixels)')
    # ax3.set_ylabel('Y Position (pixels)')
    # ax3.set_title('Camera 1 - 2D Trajectory')
    # ax3.set_aspect('equal', adjustable='box')
    # ax3.grid(True, alpha=0.3)
    # if plotted_events:
    #     ax3.legend(loc='best', fontsize=8)
     
    # Subplot 4: Calcium Spikes Heat Map
    ax4 = fig.add_subplot(gs[1, :2])
    
    n_neurons, n_timepoints = cal_spikes.shape

    
    
    # # Subplot 5: Average Calcium Activity
    # ax5 = fig.add_subplot(gs[1, 2])
    
    # # Calculate and plot average calcium activity
    # avg_activity = np.mean(cal_spikes, axis=0)
    # ax5.plot(cal_tseries, avg_activity, 'g-', linewidth=1.5, alpha=0.8)
    
    # # Add event markers
    # for i, (event_time, event_label) in enumerate(zip(cal_event_times, event_labels)):
    #     color = event_colors[event_label]
    #     idx = np.argmin(np.abs(cal_tseries - event_time))
        
    #     ax5.axvline(event_time, color=color, linestyle='--', linewidth=2, alpha=0.7)
    #     ax5.plot(event_time, avg_activity[idx], 'o', color=color, markersize=8, 
    #             markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
    
    # ax5.set_xlabel('Time (s)')
    # ax5.set_ylabel('Average Activity')
    # ax5.set_title('Average Calcium Activity')
    # ax5.grid(True, alpha=0.3)
    
    # # Subplot 6: Event Timeline
    # ax6 = fig.add_subplot(gs[2, :])
    
    # # Create event timeline
    # event_y_positions = {event: i for i, event in enumerate(unique_events)}
    
    # for i, (event_time, event_label) in enumerate(zip(cal_event_times, event_labels)):
    #     color = event_colors[event_label]
    #     y_pos = event_y_positions[event_label]
        
    #     ax6.scatter(event_time, y_pos, c=[color], s=100, alpha=0.8, 
    #                edgecolors='white', linewidth=1)
        
    #     # Add text labels (rotated to avoid overlap)
    #     ax6.text(event_time, y_pos + 0.1, event_label, fontsize=8, 
    #             ha='center', va='bottom', rotation=45)
    
    # ax6.set_xlabel('Time (s)')
    # ax6.set_ylabel('Event Type')
    # ax6.set_title('Event Timeline')
    # ax6.set_yticks(range(len(unique_events)))
    # ax6.set_yticklabels(unique_events)
    # ax6.grid(True, alpha=0.3)
    # ax6.set_ylim([-0.5, len(unique_events) - 0.5])
    
    # Overall title
    fig.suptitle(f'Mouse Day Analysis: {mouse_day.mouseID} - {mouse_day.day} - {event_key}', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    return fig

def plot_kinematic_heatmap(mouse_day, event_key: str, figsize: Tuple[int, int] = (12, 8)):
    """
    Create a heatmap showing kinematic activity across bodyparts over time
    
    Parameters:
    -----------
    mouse_day : MouseDay
        Instance of the MouseDay class
    event_key : str
        Specifies which 2.5min chunk to analyze
    figsize : tuple
        Figure size (width, height)
    """
    
    # Get kinematic matrices
    cam1_mat, cam2_mat = mouse_day.kin_mats[event_key]
    
    # Calculate movement velocity for each bodypart
    def calculate_velocity(kin_mat):
        x_coords = mouse_day.get_x_coords(kin_mat)
        y_coords = mouse_day.get_y_coords(kin_mat)
        
        # Calculate velocity magnitude
        dx = np.diff(x_coords, axis=1)
        dy = np.diff(y_coords, axis=1)
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Pad to maintain original length
        velocity = np.column_stack([velocity, velocity[:, -1]])
        
        return velocity
    
    cam1_velocity = calculate_velocity(cam1_mat)
    cam2_velocity = calculate_velocity(cam2_mat)
    
    # Create time series for kinematic data
    kin_tseries = mouse_day.kin_tseries
    min_frames = min(len(kin_tseries), cam1_velocity.shape[1])
    kin_tseries = kin_tseries[:min_frames]
    cam1_velocity = cam1_velocity[:, :min_frames]
    cam2_velocity = cam2_velocity[:, :min_frames]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Camera 1 heatmap
    im1 = axes[0].imshow(cam1_velocity, aspect='auto', cmap='viridis', 
                        extent=[kin_tseries[0], kin_tseries[-1], 0, len(mouse_day.BODYPARTS)])
    axes[0].set_ylabel('Bodypart')
    axes[0].set_title(f'Camera 1 - Velocity Heatmap\n{mouse_day.mouseID} {mouse_day.day}')
    axes[0].set_yticks(range(len(mouse_day.BODYPARTS)))
    axes[0].set_yticklabels(mouse_day.BODYPARTS, fontsize=8)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Velocity (pixels/frame)')
    
    # Camera 2 heatmap
    im2 = axes[1].imshow(cam2_velocity, aspect='auto', cmap='viridis',
                        extent=[kin_tseries[0], kin_tseries[-1], 0, len(mouse_day.BODYPARTS)])
    axes[1].set_ylabel('Bodypart')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title(f'Camera 2 - Velocity Heatmap')
    axes[1].set_yticks(range(len(mouse_day.BODYPARTS)))
    axes[1].set_yticklabels(mouse_day.BODYPARTS, fontsize=8)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Velocity (pixels/frame)')
    
    # Add event markers if available
    if hasattr(mouse_day, 'kin_event_times') and event_key in mouse_day.kin_event_times:
        event_times = mouse_day.kin_event_times[event_key]
        for ax in axes:
            for event_time in event_times:
                ax.axvline(event_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    return fig

def plot_correlation_analysis(mouse_day, event_key: str, figsize: Tuple[int, int] = (14, 10)):
    """
    Plot correlation between kinematic and calcium data
    
    Parameters:
    -----------
    mouse_day : MouseDay
        Instance of the MouseDay class
    event_key : str
        Specifies which 2.5min chunk to analyze
    figsize : tuple
        Figure size (width, height)
    """
    
    # Get interpolated kinematic averages
    cam1_avg, cam2_avg = mouse_day.interpolate_avgkin2cal(event_key)
    
    # Calculate kinematic features
    def calculate_features(cam_avg):
        x, y = cam_avg[0, :], cam_avg[1, :]
        
        # Position magnitude
        pos_mag = np.sqrt(x**2 + y**2)
        
        # Velocity
        dx = np.diff(x)
        dy = np.diff(y)
        velocity = np.sqrt(dx**2 + dy**2)
        velocity = np.append(velocity, velocity[-1])  # Pad to maintain length
        
        # Acceleration
        acceleration = np.diff(velocity)
        acceleration = np.append(acceleration, acceleration[-1])
        
        return pos_mag, velocity, acceleration
    
    cam1_pos, cam1_vel, cam1_acc = calculate_features(cam1_avg)
    cam2_pos, cam2_vel, cam2_acc = calculate_features(cam2_avg)
    
    # Average calcium activity
    avg_calcium = np.mean(mouse_day.cal_spks, axis=0)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Time series plots
    time = mouse_day.cal_tseries
    
    # Position vs Calcium
    axes[0, 0].plot(time, cam1_pos, 'b-', label='Cam 1 Position', alpha=0.7)
    axes[0, 0].plot(time, cam2_pos, 'r-', label='Cam 2 Position', alpha=0.7)
    ax_twin = axes[0, 0].twinx()
    ax_twin.plot(time, avg_calcium, 'g-', label='Avg Calcium', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position Magnitude')
    ax_twin.set_ylabel('Calcium Activity')
    axes[0, 0].set_title('Position vs Calcium Activity')
    axes[0, 0].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    
    # Velocity vs Calcium
    axes[0, 1].plot(time, cam1_vel, 'b-', label='Cam 1 Velocity', alpha=0.7)
    axes[0, 1].plot(time, cam2_vel, 'r-', label='Cam 2 Velocity', alpha=0.7)
    ax_twin = axes[0, 1].twinx()
    ax_twin.plot(time, avg_calcium, 'g-', label='Avg Calcium', alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity')
    ax_twin.set_ylabel('Calcium Activity')
    axes[0, 1].set_title('Velocity vs Calcium Activity')
    axes[0, 1].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    
    # Acceleration vs Calcium
    axes[0, 2].plot(time, cam1_acc, 'b-', label='Cam 1 Acceleration', alpha=0.7)
    axes[0, 2].plot(time, cam2_acc, 'r-', label='Cam 2 Acceleration', alpha=0.7)
    ax_twin = axes[0, 2].twinx()
    ax_twin.plot(time, avg_calcium, 'g-', label='Avg Calcium', alpha=0.7)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Acceleration')
    ax_twin.set_ylabel('Calcium Activity')
    axes[0, 2].set_title('Acceleration vs Calcium Activity')
    axes[0, 2].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    
    # Scatter plots for correlation
    axes[1, 0].scatter(cam1_pos, avg_calcium, alpha=0.6, s=10)
    axes[1, 0].set_xlabel('Camera 1 Position')
    axes[1, 0].set_ylabel('Avg Calcium Activity')
    axes[1, 0].set_title('Position vs Calcium Correlation')
    
    # Calculate and display correlation
    corr_pos = np.corrcoef(cam1_pos, avg_calcium)[0, 1]
    axes[1, 0].text(0.05, 0.95, f'r = {corr_pos:.3f}', transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1, 1].scatter(cam1_vel, avg_calcium, alpha=0.6, s=10)
    axes[1, 1].set_xlabel('Camera 1 Velocity')
    axes[1, 1].set_ylabel('Avg Calcium Activity')
    axes[1, 1].set_title('Velocity vs Calcium Correlation')
    
    corr_vel = np.corrcoef(cam1_vel, avg_calcium)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'r = {corr_vel:.3f}', transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1, 2].scatter(cam1_acc, avg_calcium, alpha=0.6, s=10)
    axes[1, 2].set_xlabel('Camera 1 Acceleration')
    axes[1, 2].set_ylabel('Avg Calcium Activity')
    axes[1, 2].set_title('Acceleration vs Calcium Correlation')
    
    corr_acc = np.corrcoef(cam1_acc, avg_calcium)[0, 1]
    axes[1, 2].text(0.05, 0.95, f'r = {corr_acc:.3f}', transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Kinematic-Calcium Correlation Analysis: {mouse_day.mouseID} - {mouse_day.day}', 
                fontsize=14)
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
    kin_times = mouse_day.kin_tseries
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
    cal_times = mouse_day.cal_tseries
    max_frames = len(cal_times)
    cam1_interp, cam2_interp = mouse_day.interpolate_avgkin2cal(event_key)
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


# Example usage function
def example_usage():
    """
    Example of how to use the plotting functions
    """
    # Assuming you have a MouseDay instance
    mouse_day = MouseDay('mouse25', '20240425')
    
    # Get available event keys
    event_keys = list(mouse_day.kin_event_times.keys())
    event_key = event_keys[0]  # Use first available key
    
    # Create comprehensive plot
    # fig1 = plot_mouseday_data(mouse_day, event_key)
    # plt.show()
    
    # Create kinematic heatmap
    # fig2 = plot_kinematic_heatmap(mouse_day, event_key)
    # plt.show()
    
    # # Create correlation analysis
    # fig3 = plot_correlation_analysis(mouse_day, event_key)
    # plt.show()

    # # Test Interpolation function
    # fig4 = plot_interp_test(mouse_day, event_key)
    # plt.show()
    
    print("Example usage:")
    print("mouse_day = MouseDay('mouse25', '20240425')")
    print("event_key = list(mouse_day.kin_event_times.keys())[0]")
    print("fig = plot_mouseday_data(mouse_day, event_key)")
    print("plt.show()")

if __name__ == "__main__":
    example_usage()