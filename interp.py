import numpy as np
import src.IO as io
from mouse import MouseDay

"""
This file provides examples of how to use the interpolation functions provided by the MouseDay class. 
For now, there's one specific type of interpolating going on: interpolating average kinematic location per calcium time bin.  
    s.t. you have two xy pairs per calcium time bin (one per camera). 

If I don't procrastinate, I will soon add more functions that allow us to interpolate all 15 key-points...


Within a folder called "neuro-behavior-decoder", you'll need to download the following files:
    src/IO.py   <- holds all the system functions for loading/saving data
    mouse.py    <- holds the MouseDay class

You also need to format the data according to the mouse and day in the file system.
For this example, we'll use mouse25 on april 25th. 
    mouse25/20240425
        calcium
            calcium_event_times.npy
            calcium_timestamps.pkl
            cam_event_times.pkl
            cam_timestamps.pkl
            cascade_spks.npy
            event_labels.npy
            red_labels.npy
        kinematics
            (all dlc files for this MouseDay)


To summarize, all files/data should follow this format to properly construct the MouseDay class. 
The folder, neuro-behavior-decoder, can be located anywhere on your computer. It just needs to follow this format within the folder itself. 

neuro-behavior-decoder
    mouse25
        20240425
            calcium
                calcium_event_times.npy
                calcium_timestamps.pkl
                cam_event_times.pkl
                cam_timestamps.pkl
                cascade_spks.npy
                event_labels.npy
                red_labels.npy
            kinematics
                (all dlc files for this MouseDay)
    src
        IO.py
    decode.py
    mouse.py
    interp.py

    (Optional files)
    fextract.py      <-to load the data from your downloads and into this specific folder setup
                       sorry I made it so picky I hope this file makes up for it
    requirements.txt <- if you'd like to set up a virtual environment to hold all these versions of python packages that the code uses. 
                        otherwise you need to download them manually yourself.



To run THIS file, open a terminal and navigate to the neuro-behavior-decoder directory. 
Make sure you have all the required python packages installed. 
type into the terminal: python3 interp.py
"""


if __name__ == "__main__":
    # Constructs a MouseDay object for mouse25 on april 25th, 2024
    mouseID: str = "mouse25"
    day: str = "20240425"
    mouse_day: MouseDay = MouseDay(mouseID, day)

    # Interpolates the average paw location to calcium time bins for each recording segment
    #   key = segment key (denotes the recording segment within this day)
    #   values = two arrays of floats (paw location in pixels), one per camera view
    #               within the arrays themselves, the first row is x, second row is y
    interpolated_kinematics_dict: dict[str: [np.ndarray, np.ndarray]] = mouse_day.interpolate_all("avg")

    # prints the contents of the dictionary. also helpful to see how to access the data within these strucutres. 
    print("Average locations interpolated...")
    for (seg, data) in interpolated_kinematics_dict.items():
        print()
        print("recording segment", seg)
        print("(camera 1)")
        print(" x: ", data[0][0]) # grabs the first array, first row
        print(" y: ", data[0][1])
        print("(camera 2)")
        print(" x: ", data[1][0])
        print(" y: ", data[1][1])
    print("----------------")


    # Saves the dictionary to a folder called "interpolated kinematic data". Creates the folder if it doesn't already exist. 
    path: str = io.save_interpolated_kinematics(mouseID, day, interpolated_kinematics_dict, kin_dtype="avg")
    print(f"Successfully saved data to {path}!")
    

    # To load in that data back for whatever reason (this would be done outside of this file, assuming you JUST have the interpolated kinematic files and not the mouseDay class)
    new_dict: dict[str: [np.ndarray, np.ndarray]] = io.load_interpolated_kinematics(mouseID, day, kin_dtype="avg")