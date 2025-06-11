import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut

"""
For every body part, interpolate the average location to the time of the cascade spikes for the specific sample. 

Parameters
    MouseID and Day
Output
    Numpy file of interpolated locations during cascade spike times. 
"""
def interpolate_mouse_data(mouseID, day) -> np.ndarray:
    """
    Neural Data
        cascade_spks in 33 ms/frame, concat overall 2.5 mins
        timestamps in unix time/frame
    """
    calcium_spks, calcium_event_times, calcium_event_labels = io.load_spks_and_events(mouseID, day)

    """
    Camera Data
        kinevamics.h5: 1 file/2.5 mins, 15 body parts with x,y locations, 5 ms/frame
        timestamps in unix time/frame
    """
    event_labels = load_event_labels(mouseID, day)
    cam_event_times = load_cam_event_times(mouseID, day)
    kinematics = "" #PLACEHOLDER - access using utils function (the dlcfiles?)

    # Should I create new files of interpolated data?
    x = arange(0, calcium_event_times[-1]) # should the x coordinates be generated seperately, or be the calcium event times themselves?
    interpolated_calcium = np.interp(x, calcium_event_times, calcium_spks)
    interpolated_camera = np.interp(x, cam_event_times, kinematics)
    interpolated_labels = np.interp(x, cam_event_times, event_labels)

# Tracks mouse data and dates imported onto this file system (in prog)
# Figure out a way to not have to do this manually (read off directory??)
mice:list[str] = ['mouse49']
days:dict[str,list[str]] = {'mouse49':['042524']}

"""
for mouseID in mice:
    for day in days[mouseID]:
        interpolate_mouse_data(mouseID, day)
"""

mouse = "mouse25"
day = "20240425"

# calcium data
spks = io.load_spks(mouse, day)
cal_event_times = io.load_cal_event_times(mouse, day)
cal_tseries = io.load_tseries(mouse, day, "calcium")

# kinematic data
camdf1, camdf2 = io.load_kinematics_df("133901event001", "mouse25", "20240425")
bodyparts = io.get_bodyparts(camdf1)
cam1_matrix = io.load_kinematics_matrix(camdf1, bodyparts, 0.4)

# Get the average x and y coordinates across all bodyparts
cam1_avg_coordinates = io.get_avg_coordinates(cam1_matrix, bodyparts) # maybe later ill modify this function to compute a weighted average
print(cam1_avg_coordinates.shape)
cam1_x_avg = cam1_avg_coordinates[:, 0]
print(cam1_x_avg.shape)
cam1_y_avg = cam1_avg_coordinates[:, 1]

# Get the time frames
cam_tseries = io.load_tseries(mouse, day, "cam")

# print("kin start time: ", cam_tseries[0])
# print("cal start time: ", cal_tseries[0])
camera_frames = len(cam_tseries)
loc_frames = len(cam1_x_avg)

print(camera_frames)
print(loc_frames)

min_frames= min(camera_frames, loc_frames)
print(min_frames)

cam1_x_avg = cam1_x_avg[:min_frames] # Resizing the camera frames to match the camera time series
cam1_y_avg = cam1_y_avg[:min_frames]
cam_tseries = cam_tseries[:min_frames]

cam1_x_avg_interp = np.interp(cal_tseries, cam_tseries, cam1_x_avg)
print(cam1_x_avg)
print(cam1_x_avg_interp)
# cam2_y_avg_interp = np.interp(cal_tseries, cam_tseries, cam1_y_avg)