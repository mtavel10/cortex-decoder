# importing the zipfile module
from zipfile import ZipFile
import os
import glob
import shutil

MICE = ['mouse25']
DAYS = ['042024', '042124', '042224', '042324', '042424', '042824', '042924', '043024', '050124', '050224', '050324']

def load_dlc_data(mice=MICE, days=DAYS):
    # loading the dlc data into 'kinematics' folders
    res_path = f"/home/macleanlab/Downloads/dlc_results_all"
    res_contents = os.listdir(res_path)
    for mouse in mice: 
        for day in days: 
            matching_files = []
            for file_path in res_contents:
                # Check if this file matches our criteria
                if mouse in file_path and f"2024{day[:4]}" in file_path:
                    matching_files.append(file_path)
            
            dlc_path = f"/home/macleanlab/Documents/Maddy/neuro-behavior-decoder/{mouse}/2024{day[:4]}/kinematics"
            os.makedirs(dlc_path, exist_ok=True)
            for file in matching_files:
                shutil.move(f"{res_path}/{file}", dlc_path)

def load_cal_data(mice=MICE, days=DAYS):
    for mouse in mice:
        for day in days:
            src = f"/home/macleanlab/Downloads/{day}-selected.zip"
            dst = f"/home/macleanlab/Documents/Maddy/neuro-behavior-decoder/{mouse}/2024{day[:4]}/calcium"
            # make the calcium data path
            os.makedirs(dst, exist_ok=True)
            with ZipFile(src, 'r') as zObject:
                zObject.extractall(path=dst)

if __name__ == "__main__":
    load_dlc_data(['mouse25'], ['042424'])