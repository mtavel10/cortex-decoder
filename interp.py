import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut
from mouse import MouseDay

# Tracks mouse data and dates imported onto this file system (in prog)
# Figure out a way to not have to do this manually (read off directory??)
mice:list[str] = ["mouse25"]
days:dict[str,list[str]] = {"mouse25": ["20240425"]}


# for mouseID in mice:
#     for day in days[mouseID]:

#         interpolated_avg_locs = interpolate_loc_to_cal_frame(mouseID, day)
#         print(interpolated_avg_locs)


# Mouse Test

test_mouse = MouseDay("mouse25", "20240425")
print("old interpolation...")
print(test_mouse.OLDinterpolate_avgkin2cal(test_mouse.seg_keys[0]))
print("new interpolation...")
print(test_mouse.interpolate_avgkin2cal(test_mouse.seg_keys[0]))
# cal_tstamp_dict = test_mouse.cal_tstamp_dict
# # print(cal_tstamp_dict)
# cal_tseries = test_mouse.cal_tseries
# print(cal_tseries)
# kin_tstamp_dict = test_mouse.kin_tstamp_dict
# print(kin_tstamp_dict)
# interp_test = test_mouse.interpolate_avgkin2cal("133901event001")
# print(interp_test)
# print(test_mouse.cal_tstamp_dict)
# print(test_mouse.kin_tstamp_dict)
# print(test_mouse.seg_keys)
# print("Calcium (camera) frames: ", test_mouse.cal_nframes)
# print("Calcium (timestamp) frames: ", test_mouse.cal_ntimeframes)
# print("Kinematics (camera) frames: ", test_mouse.kin_nframes)
# print("Kinematics (timestamp) frames: ", test_mouse.kin_ntimeframes)

# print(test_mouse.cal_spks[32:-32, 32:-32])

# test_cal_tstamps = test_mouse.cal_tstamps
# # print(test_cal_tstamps)
# test_kin_tstamps = test_mouse.kin_tstamps
# # print(test_cam_tstamps)