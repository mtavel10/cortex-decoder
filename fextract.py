# importing the zipfile module
from zipfile import ZipFile
import os
import glob
import shutil

mice = ['mouse25']
days = ['042024', '042124', '042224', '042324', '042424', '042824', '042924', '043024', '050124', '050224', '050324']

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