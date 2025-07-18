# importing the zipfile module
from zipfile import ZipFile
import os

days = ['042024', '042124', '042224', '042324', '042424', '042824', '042924', '043024', '050124', '050224', '050324']

for day in days:
    # # loading the temp.zip and creating a zip object
    # with ZipFile(f"/home/macleanlab/Downloads/{day}-selected.zip", 'r') as zObject:

    #     # Extracting all the members of the zip into a specific location.
    #     zObject.extractall(
    #         path=f"/home/macleanlab/Documents/Maddy/neuro-behavior-decoder/mouse25/2024{day[:4]}/calcium")
    os.mkdir(f"/home/macleanlab/Documents/Maddy/neuro-behavior-decoder/mouse25/2024{day[:4]}/kinematics")
    os.mkdir(f"/home/macleanlab/Documents/Maddy/neuro-behavior-decoder/mouse25/2024{day[:4]}/tseries")