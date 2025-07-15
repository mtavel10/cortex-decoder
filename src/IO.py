import numpy as np
import pickle
import glob
import pandas as pd
import src.utils
import os

def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pickle(filepath, obj):
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


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
    cam_time_fn = f"{s2p_fld}/calcium/cam_event_times.pkl"
    event_times = load_pickle(cam_time_fn)
    # get rid of None entries
    filt_event_times = {k: v for k, v in event_times.items() if v is not None}
    return filt_event_times


def load_event_labels(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    return np.load(f"{s2p_fld}/calcium/event_labels.npy")


# calcium data
def load_spks(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    spks = np.load(f"{s2p_fld}/calcium/cascade_spks.npy")
    return spks

def load_spk_labels(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    labels = np.load(f"{s2p_fld}/calcium/red_labels.npy")
    return labels


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
    tstamps = load_pickle(f"{s2p_fld}/calcium/{type}_timestamps.pkl")
    for event in tstamps:
        tstamps[event] = tstamps[event].to_numpy(dtype=float)
    
    return tstamps

def load_cal_tstamps(mouseID, day):
    """
    Returns
        numpy.ndarray
            Seconds since Unix Epoch (float) per calcium camera frame
    """
    s2p_fld = get_s2p_fld(mouseID, day)
    cal_tstamps = np.load(f"{s2p_fld}/calcium/calcium_timestamps.npy")
    return cal_tstamps.astype('datetime64[ns]').astype(float)


def save_decoded_data(mouseID: str, day: str, scores: list[float] | None, preds: np.ndarray | None, model_type="general"):
    """ Saves decoded scores and predictions for comparison and plotting purposes """
    file_path = f"{get_drive(mouseID)}/decoded_data/{mouseID}/{day}"
    os.makedirs(file_path, mode=0o777, exist_ok=True)
    np.save(f"{file_path}/{model_type}_preds.npy", preds)
    np.save(f"{file_path}/{model_type}_scores.npy", scores)


def save_scores_by_beh(mouseID: str, day: str, scores: dict[int, np.ndarray]):
    file_path = f"{get_drive(mouseID)}/decoded_data/{mouseID}/{day}"
    file_name = f"{file_path}/general_scores_by_behavior.pkl"
    save_pickle(file_name, scores)


def load_scores_by_beh(mouseID: str, day: str):
    file_path = f"{get_drive(mouseID)}/decoded_data/{mouseID}/{day}"
    file_name = f"{file_path}/general_scores_by_behavior.pkl"
    scores = load_pickle(file_name)
    return scores

def load_decoded_data(mouseID: str, day: str, model_type="general"):
    """ Loads decoded scores and predictions for comparison and plotting purposes """
    file_path = f"{get_drive(mouseID)}/decoded_data/{mouseID}/{day}"
    preds = np.load(f"{file_path}/{model_type}_preds.npy")
    scores = np.load(f"{file_path}/{model_type}_scores.npy")
    return scores, preds

def save_model(mouseID: str, day: str, model_obj: any, model_type="general"):
    file_path = f"{get_drive(mouseID)}/decoded_data/{mouseID}/{day}"
    file_name = f"{file_path}/{model_type}_model.pkl"
    save_pickle(file_name, model_obj)

def load_model(mouseID: str, day: str, model_type="general") -> any:
    file_path = f"{get_drive(mouseID)}/decoded_data/{mouseID}/{day}/{model_type}_model.pkl"
    model_obj = load_pickle(file_path)
    return model_obj