import numpy as np
import IO as io
import utils as ut

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
    event_labels = io.load_event_labels(mouseID, day)
    cam_event_times = io.load_cam_event_times(mouseID, day)
    kinematics = "" #PLACEHOLDER - access using utils function (the dlcfiles?)

    # Should I create new files of interpolated data?
    x = arange(0, calcium_event_times[-1]) # should the x coordinates be generated seperately, or be the calcium event times themselves?
    interpolated_calcium = np.interp(x, calcium_event_times, calcium_spks)
    interpolated_camera = np.interp(x, cam_event_times, kinematics)
    interpolated_labels = np.interp(x, cam_event_times, event_labels)

    print("For ", mouseID, ": Day: ", day)
    print("Calcium event times: ")
    print(calcium_event_times)
    print("Event labels: ")
    print(event_labels)
    print("Camera event times: ")
    print(cam_event_times)

# Tracks mouse data and dates imported onto this file system (in prog)
# Figure out a way to not have to do this manually (read off directory??)
mice:list[str] = ['mouse49']
days:dict[str,list[str]] = {'mouse49':['042524']}

"""
for mouseID in mice:
    for day in days[mouseID]:
        interpolate_mouse_data(mouseID, day)
"""


cam_event_times = io.load_cam_event_times("mouse49", "061025")
print(cam_event_times)

event_labels = io.load_event_labels("mouse49", "061025")
print()
print(event_labels)

cal_event_times = io.load_cal_event_times("mouse49", "061025")
print(cal_event_times)