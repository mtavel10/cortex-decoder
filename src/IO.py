import numpy as np
import pickle
import glob
import pandas as pd
import os
import mat73
from typing import Any, Dict, List, Tuple


def load_pickle(filepath) -> Any:
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pickle(filepath, obj):
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


def get_drive() -> str:
    """ Returns the current working directory. Helper for other loading functions. """
    cwd = os.getcwd()
    return cwd


def get_s2p_fld(mouseID, day) -> str:
    """ Gets the data's location within the current working directory. """
    drive = get_drive()
    s2p_fld = f"{drive}/{mouseID}/{day}"
    return s2p_fld


def get_days(mouseID) -> List[str]:
    """ Returns a list of days within this mouse's folder """
    drive = get_drive()
    day_paths = sorted(glob.glob(drive+mouseID+'/*24'))
    days = [day.split('/')[-1] for day in day_paths]
    return days


# KINEMATIC DATA

def load_cam_event_times(mouseID, day) -> Dict[Any, Any]:
    s2p_fld = get_s2p_fld(mouseID, day)
    cam_time_fn = f"{s2p_fld}/calcium/cam_event_times.pkl"
    event_times = load_pickle(cam_time_fn)
    # get rid of None entries
    filt_event_times = {k: v for k, v in event_times.items() if v is not None}
    return filt_event_times


def load_event_labels(mouseID, day) -> np.ndarray:
    s2p_fld = get_s2p_fld(mouseID, day)
    return np.load(f"{s2p_fld}/calcium/event_labels.npy")


def load_hdf(file) -> pd.DataFrame:
    df = pd.read_hdf(file)
    return df


def load_kinematics_df(key,mouseID,day) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


# CALCIUM DATA

def load_spks(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    spks = np.load(f"{s2p_fld}/calcium/cascade_spks.npy")
    return spks


def load_spk_labels(mouseID, day):
    """ Labels neurons as inhibitory (1) or excitatory (0) """
    s2p_fld = get_s2p_fld(mouseID, day)
    labels = np.load(f"{s2p_fld}/calcium/red_labels.npy")
    return labels


def load_cal_event_times(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID,day)
    event_times = np.load(f"{s2p_fld}/calcium/calcium_event_times.npy")
    return event_times


# DEPRECIATED - loading in timestamps in dictionary to mark tstamps per event
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

# DEPRECIATED - loading in timestamps per event (see tstamp dict)
def load_cal_tstamps(mouseID, day):
    """
    Returns
        numpy.ndarray
            Seconds since Unix Epoch (float) per calcium camera frame
    """
    s2p_fld = get_s2p_fld(mouseID, day)
    cal_tstamps = np.load(f"{s2p_fld}/calcium/calcium_timestamps.npy")
    return cal_tstamps.astype('datetime64[ns]').astype(float)

def load_reg_dict(mouseID, day) -> np.ndarray:
    """
    Returns a dictionary of day i's 
    """
    reg_dir= f"{get_drive()}/{mouseID}/registered_cell_pairs"
    mat_paths = os.listdir(reg_dir)
    matching_files = []
    for path in mat_paths:
        if day in path:
            matching_files.append(path)
    
    reg_dict = {}
    for file in matching_files:
        reg = mat73.loadmat(f"{reg_dir}/{file}")
        dayi = file[:8]
        dayj = file[9:-4]

        reg = reg['cell_registered_struct']['cell_to_index_map']
        
        reg_list = []
        # append every registered neuron (i.e. those greater than zero)
        # counts the sum of registered neurons
        for i in range(reg.shape[1]):
            reg_list.append(reg[:, i] > 0)
        cells_in_all = np.all(np.array(reg_list),axis=0)

        ind_list = []
        for i in range(reg.shape[1]):
            ind_list.append(reg[cells_in_all, i].astype(int)-1)#minus one because matlab using 1-indexing

        dst_day = ""
        # The first row corresponds to this mouse
        if dayi == day:
            dst_day = dayj
        # The second row corresponds to this mouse... need to switch the rows
        elif dayj == day:
            dst_day = dayi
            ind_list[0], ind_list[1] = ind_list[1], ind_list[0]

        reg_dict[dst_day] = ind_list

    return reg_dict


def save_decoded_data(mouseID: str, day: str, scores: list[float]=None, preds: np.ndarray=None, model_type="general"):
    """ Saves decoded scores and predictions for comparison and plotting purposes """
    file_path = f"{get_drive()}/decoded_data/{mouseID}/{day}"
    print("saving to...", file_path)
    os.makedirs(file_path, mode=0o777, exist_ok=True)
    np.save(f"{file_path}/{model_type}_preds.npy", preds)
    np.save(f"{file_path}/{model_type}_scores.npy", scores)


def save_scores_by_beh(mouseID: str, day: str, scores: dict[int, np.ndarray]):
    file_path = f"{get_drive()}/decoded_data/{mouseID}/{day}"
    file_name = f"{file_path}/general_scores_by_behavior.pkl"
    save_pickle(file_name, scores)


def load_scores_by_beh(mouseID: str, day: str):
    file_path = f"{get_drive()}/decoded_data/{mouseID}/{day}"
    file_name = f"{file_path}/general_scores_by_behavior.pkl"
    scores = load_pickle(file_name)
    return scores

def load_decoded_data(mouseID: str, day: str, model_type="general"):
    """ Loads decoded scores and predictions for comparison and plotting purposes """
    file_path = f"{get_drive()}/decoded_data/{mouseID}/{day}"
    preds = np.load(f"{file_path}/{model_type}_preds.npy")
    scores = np.load(f"{file_path}/{model_type}_scores.npy")
    return scores, preds

def save_model(mouseID: str, day: str, model_obj: any, model_type="general"):
    file_path = f"{get_drive()}/decoded_data/{mouseID}/{day}"
    file_name = f"{file_path}/{model_type}_model.pkl"
    save_pickle(file_name, model_obj)

def load_model(mouseID: str, day: str, model_type="general") -> any:
    file_path = f"{get_drive()}/decoded_data/{mouseID}/{day}/{model_type}_model.pkl"
    model_obj = load_pickle(file_path)
    return model_obj

def save_interpolated_kinematics(mouseID: str, day: str, data: dict[str: [np.ndarray, np.ndarray]], kin_dtype: str="avg"):
    """
    Saves a dictionary of kinematic locations for one day, interpolated so that every time bin aligns with the calcium data's frequency. 
    Parameters
        data
            key = segment key (denotes the recording segment within this day)
            values = two numpy arrays, one per camera view
        kin_dtype
            "avg" is the default - just to name the file based on whether the data is averaged across all keypoints
    """
    file_path = f"{get_drive()}/interpolated_kinematic_data"
    file_name = f"{mouseID}_{day}_interpolated_{kin_dtype}.pkl"
    os.makedirs(file_path, mode=0o777, exist_ok=True)
    save_pickle(f"{file_path}/{file_name}", data)
    return file_path

def load_interpolated_kinematics(mouseID: str, day: str, kin_dtype: str="avg"):
    file_path = f"{get_drive()}/interpolated_kinematic_data"
    file_name = f"{mouseID}_{day}_interpolated_{kin_dtype}.pkl"
    data = load_pickle(f"{file_path}/{file_name}")
    return data