import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut

def interpolate_loc_to_cal_frame(mouseID, day):
    """
    Interpolates the average location of the mouse's hand during the calcium frame times. 

    Parameters
        MouseID and Day
    Returns
        Numpy NDArray (2, n_timepoints)
        Average location, interpolated to calcium time series (each timepoint is a calcium camera frame)
    """

    # calcium time per frame
    cal_tseries = io.load_tseries(mouseID, day, "calcium")

    # kinematic time per frame
    cam_tseries = io.load_tseries(mouseID, day, "cam")

    # kinematic location per frame
    camdf1, camdf2 = io.load_kinematics_df("133901event001", "mouse25", "20240425")

    # Camera 1
    bodyparts = io.get_bodyparts(camdf1)
    cam1_matrix = io.load_kinematics_matrix(camdf1, bodyparts, 0.4)

    # Get the average x and y coordinates across all bodyparts
    cam1_avg_coordinates = io.get_avg_coordinates(cam1_matrix, bodyparts) # maybe later ill modify this function to compute a weighted average
    cam1_x_avg = cam1_avg_coordinates[:, 0]
    cam1_y_avg = cam1_avg_coordinates[:, 1]

    # Resizing the camera frames to match the camera time series
    camera_frames = len(cam_tseries)
    loc_frames = len(cam1_x_avg)
    min_frames= min(camera_frames, loc_frames)

    cam1_x_avg = cam1_x_avg[:min_frames]
    cam1_y_avg = cam1_y_avg[:min_frames]
    cam_tseries = cam_tseries[:min_frames]

    # Interpolate! 
    cam1_x_avg_interp = np.interp(cal_tseries, cam_tseries, cam1_x_avg)
    cam1_y_avg_interp = np.interp(cal_tseries, cam_tseries, cam1_y_avg)

    # Smush together
    cam1_avg_interp = np.stack((cam1_x_avg_interp, cam1_y_avg_interp), axis=0)

    print(cam1_avg_interp)

# Tracks mouse data and dates imported onto this file system (in prog)
# Figure out a way to not have to do this manually (read off directory??)
mice:list[str] = ["mouse25"]
days:dict[str,list[str]] = {"mouse25": ["20240425"]}


for mouseID in mice:
    for day in days[mouseID]:
        interpolated_avg_locs = interpolate_loc_to_cal_frame(mouseID, day)
        print(interpolated_avg_locs)


# # Function Draft
# mouse = "mouse25"
# day = "20240425"

# # calcium data
# spks = io.load_spks(mouse, day)
# cal_event_times = io.load_cal_event_times(mouse, day)
# cal_tseries = io.load_tseries(mouse, day, "calcium")

# # kinematic data
# camdf1, camdf2 = io.load_kinematics_df("133901event001", "mouse25", "20240425")
# bodyparts = io.get_bodyparts(camdf1)
# cam1_matrix = io.load_kinematics_matrix(camdf1, bodyparts, 0.4)

# # Get the average x and y coordinates across all bodyparts
# cam1_avg_coordinates = io.get_avg_coordinates(cam1_matrix, bodyparts) # maybe later ill modify this function to compute a weighted average
# cam1_x_avg = cam1_avg_coordinates[:, 0]
# cam1_y_avg = cam1_avg_coordinates[:, 1]

# # Get the time frames
# cam_tseries = io.load_tseries(mouse, day, "cam")

# # Resizing the camera frames to match the camera time series
# camera_frames = len(cam_tseries)
# loc_frames = len(cam1_x_avg)
# min_frames= min(camera_frames, loc_frames)

# cam1_x_avg = cam1_x_avg[:min_frames]
# cam1_y_avg = cam1_y_avg[:min_frames]
# cam_tseries = cam_tseries[:min_frames]

# # Interpolate! 
# cam1_x_avg_interp = np.interp(cal_tseries, cam_tseries, cam1_x_avg)
# cam2_y_avg_interp = np.interp(cal_tseries, cam_tseries, cam1_y_avg)

# print(cam1_x_avg_interp)
# print(cam2_y_avg_interp)