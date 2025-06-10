import numpy as np
import src.IO as io

# Tracks mouse data and dates imported onto this file system (in prog)
# Figure out a way to not have to do this manually (read off directory??)
mice:list[str] = ['mouse49']
days:dict[str,list[str]] = {'mouse49':['042524']}

"""
Interpolates calcium spike and camera (kinematic) data for given mice. Writes interpolated data to files. 
"""
def interp(mice)
    for mouseID in mice:
        for day in days[mouseID]:
            """
            Calcium Data
                x: times
                y: spks
                event_labels (?) <- how are these different from the camera labels
            """
            calcium_spks, calcium_event_times, calcium_event_labels = io.load_spks_and_events(mouseID, day)
            
            """
            Camera Data
                x: times
                y: kinematic location (average for now) (how to get????)
                event_labels
            """
            event_labels = io.load_event_labels(mouseID, day)
            cam_event_times = io.load_cam_event_times(mouseID, day)
            kinematics = "" #PLACEHOLDER - where do I access this data?

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