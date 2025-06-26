import numpy as np
import pickle
import glob
import pandas as pd
import src.utils
import os

def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pickle(filename,var):
    with open(filename, "wb") as file:
        pickle.dump(var, file)


def get_drive(mouseID):
    cwd = os.getcwd()
    return cwd


def get_s2p_fld(mouseID,day):
    drive = get_drive(mouseID)
    s2p_fld = f"{drive}/{mouseID}/{day}"
    return s2p_fld


def get_days(mouseID):
    drive = get_drive(mouseID)
    day_paths = sorted(glob.glob(drive+mouseID+'/*24'))
    days = [day.split('/')[-1] for day in day_paths]
    return days


# kinematic/camera data
def load_cam_event_times(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    cam_time_fn = f"{s2p_fld}/kinematics/cam_event_times.pkl"
    event_times = load_pickle(cam_time_fn)
    # get rid of None entries
    filt_event_times = {k: v for k, v in event_times.items() if v is not None}
    return filt_event_times


def load_event_labels(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    return np.load(f"{s2p_fld}/kinematics/event_labels.npy")


# calcium data
def load_spks(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    spks = np.load(f"{s2p_fld}/calcium/cascade_spks.npy")
    return spks


def load_cal_event_times(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID,day)
    event_times = np.load(f"{s2p_fld}/calcium/calcium_event_times.npy")
    return event_times


# kinematic data
def load_hdf(file):
    df = pd.read_hdf(file)
    return df


def load_kinematics_df(key,mouseID,day):
    """
    Load kinematics data from two camera views. 
    
    Parameters
        key : str
            Combined time+event identifier (e.g., '122634event005')
        mouseID : str
            Mouse identifier
        day : str
            Date in yyyymmdd
        
    Returns
        tuple
            (df_cam1, df_cam2) - DataFrames containing pose data from both cameras
    """
    time = key[:6]
    event = key[6:]
    s2p_fld = get_s2p_fld(mouseID, day)
    
    # Search for filenames with these substrings
    fn_cam1 = glob.glob(f"{s2p_fld}/kinematics/*{day}*{time}*{mouseID}*{event}*cam1*.h5")
    fn_cam2 = glob.glob(f"{s2p_fld}/kinematics/*{day}*{time}*{mouseID}*{event}*cam2*.h5")
    
    # Filter out 'filtered' files from both cameras
    fn_cam1 = [fn for fn in fn_cam1 if 'filtered' not in fn]
    fn_cam2 = [fn for fn in fn_cam2 if 'filtered' not in fn]

    # Check if no kinematic files found
    if len(fn_cam1) == 0 or len(fn_cam2) == 0:
        raise FileNotFoundError(f"No kinematic files found - cam1: {len(fn_cam1)} files, cam2: {len(fn_cam2)} files")

    # Check if more than one kinematic file found for either camera
    if len(fn_cam1) > 1 or len(fn_cam2) > 1:
        raise AssertionError(f"More than one kinematic file found - cam1: {fn_cam1}, cam2: {fn_cam2}")
    
    df_cam1 = load_hdf(fn_cam1[0])
    df_cam2 = load_hdf(fn_cam2[0])
    return df_cam1,df_cam2


# Time series retrieval - will modify file path search with glob once we have concatenated files
def load_tseries(mouseID, day, type):
    """
    Loads time stamp data.
    Parameters
        mouseID
        day
        type
            either "calcium" or "cam" to specify the frame type for this series
    Returns
        numpy np.array
            Seconds since Unix Epoch (float) per camera frame
    """
    s2p_fld = get_s2p_fld(mouseID, day)
    tseries = np.load(f"{s2p_fld}/tseries/TSeries-04252024-0944-1316_{type}_frame_timestamps.npy")
    tseries_converted = tseries.astype('datetime64[ns]').astype(float)
    # print(type, ": ", tseries_converted)
    return tseries_converted

# ALL timestamps for the day
def load_tstamp_dict(mouseID, day, type):
    """
    Parameters
        mouseID
        day
        type
            either 'calcium' or 'cam'
    Returns
        dict {"segkey": numpy.ndarray}
        Seconds since Unix Epoch (float) per camera frame (for either calcium camera or kinematics camera, as specified by type param)
    """
    s2p_fld = get_s2p_fld(mouseID, day)
    print(s2p_fld, f"/tseries/{type}_timestamps.pkl")
    tstamps = load_pickle(f"{s2p_fld}/tseries/{type}_timestamps.pkl")
    for event in tstamps:
        tstamps[event] = tstamps[event].to_numpy(dtype=float)
        if type == "calcium":
            print("num stamps in this event: ", len(tstamps[event]))
            print("stamps in this event: ", tstamps[event])
    
    return tstamps

def load_cal_tstamps(mouseID, day):
    """
    Returns
        numpy.ndarray
            Seconds since Unix Epoch (float) per calcium camera frame
    """
    s2p_fld = get_s2p_fld(mouseID, day)
    cal_tstamps = np.load(f"{s2p_fld}/tseries/calcium_timestamps.npy")
    return cal_tstamps.astype('datetime64[ns]').astype(float)