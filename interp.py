import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut
from mouse import MouseDay

# Tracks mouse data and dates imported onto this file system (in prog)
# Figure out a way to not have to do this manually (read off directory??)
# mice:list[str] = ["mouse25"]
# days:dict[str,list[str]] = {"mouse25": ["20240425"]}


# for mouseID in mice:
#     for day in days[mouseID]:

#         interpolated_avg_locs = interpolate_loc_to_cal_frame(mouseID, day)
#         print(interpolated_avg_locs)


# Mouse Test

test_mouse = MouseDay("mouse25", "20240425")
X = test_mouse.get_trimmed_spks()
y = test_mouse.get_trimmed_avg_locs()

print(X.shape)
print(y.shape)

print(f"X shape: {X.shape}")
print(f"X.T @ X would be: {X.shape[1]} x {X.shape[1]}")
print(f"Estimated memory for X.T @ X: {X.shape[1]**2 * 8 / 1e9:.2f} GB")

# Ridge Regression
lam = 0.4

X_sqr = X.T @ X
w = np.linalg.inv(X_sqr + lam * np.eye(X_sqr.shape[0])) @ X.T @ y

# Least Squares Regression
# w = np.linalg.inv(X.T @ X) @ X.T @ y


# interpolated_data = np.load("test_interps.npy", allow_pickle=True)
# print(interpolated_data)