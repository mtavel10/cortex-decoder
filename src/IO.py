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


# camera data
def load_cam_event_times(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    cam_time_fn = f"{s2p_fld}/camera/cam_event_times.pkl"
    event_times = load_pickle(cam_time_fn)
    # get rid of None entries
    filt_event_times = {k: v for k, v in event_times.items() if v is not None}
    return filt_event_times


def load_event_labels(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    return np.load(f"{s2p_fld}/camera/event_labels.npy")


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
    Load kinematics data from two camera views for a specific behavioral event.
    
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
    
    # searches for filenames with these substrings
    fn_cam1 = glob.glob(f"{s2p_fld}/kinematics/*{day}*{time}*{mouseID}*{event}*cam1*.h5")
    fn_cam2 = glob.glob(f"{s2p_fld}/kinematics/*{day}*{time}*{mouseID}*{event}*cam2*.h5")

    # checks if there is more than one kinematic file with these identifiers
    if (len(fn_cam1)>1) or (len(fn_cam2)>1):
        fn_cam1 = [fn for fn in fn_cam1 if not 'filtered' in fn] 
        fn_cam2 = [fn for fn in fn_cam2 if not 'filtered' in fn]
    assert (len(fn_cam1)==1) and (len(fn_cam2)==1), f'more than one kin file in {fn_cam1} or {fn_cam2}'
    
    df_cam1 = load_hdf(fn_cam1[0])
    df_cam2 = load_hdf(fn_cam2[0])
    return df_cam1,df_cam2


def get_bodyparts(df):
    # Extract level 1 (bodyparts) and get unique values
    bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()
    return sorted(bodyparts)


def get_x_y(df,bp,pcutoff):
    """
    Cuts off x and y values that are lower then a certain liklihood within a given dataframe. 
    """
    prob = df.xs(
        (bp, "likelihood"), level=(-2, -1), axis=1
    ).values.squeeze()
    mask = prob < pcutoff
    temp_x = np.ma.array(
        df.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
        mask=mask,
    )
    temp_y = np.ma.array(
        df.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
        mask=mask,
    )
    return temp_x, temp_y


def load_kinematics_matrix(df,bodyparts,pcutoff):
    """
    This function cstacks the x locations for each bodypart on top of the y locations. 
    Ex: 
                Frame: 0    1    2    3    4
    Row 0 (wrist_X): [120, 125, 130, 135, 140]
    Row 1 (elbow_X): [100, 105, 110, 115, 120] 
    Row 2 (d2tip_X): [150, 155, 160, 165, 170]
        ──────────────────────────────────────
    Row 3 (wrist_Y): [200, 205, 210, 215, 220]
    Row 4 (elbow_Y): [180, 185, 190, 195, 200]
    Row 5 (d2tip_Y): [220, 225, 230, 235, 240]

    Returns:
        numpy.ma.MaskedArray
            Masked array of shape (2*len(bodyparts), n_timepoints)
            First len(bodyparts) rows are x coordinates
            Last len(bodyparts) rows are y coordinates
            Low-confidence points are masked based on pcutoff
    """
    # Get initial for result matrix
    x_ref, y_ref = get_x_y(df,'wrist',pcutoff)
    n_timepoints = x_ref.shape[0]
    n_parts = len(bodyparts)

    kinematics_all = np.ma.masked_all([2 * n_parts, n_timepoints])
    # Fill in x, y coordinates for each bodypart
    for j, bodypart in enumerate(bodyparts):
        x, y = get_x_y(df, bodypart, pcutoff)
        kinematics_all[j,:] = x 
        kinematics_all[n_parts+j,:] = y
    
    return kinematics_all


def get_bodypart_coordinates(kinematics_matrix, bodyparts, part):
    """
    Helper function to extract x,y coordinates for a specific bodypart from kinematics matrix.
    
    Parameters:
        kinematics_matrix : numpy.ma.MaskedArray
            Output from load_kinematics_matrix
        bodyparts : list
            List of bodyparts
        part : str
            The specific bodypart you want to locate
        
    Returns:
        tuple
            (x_coords, y_coords) as masked arrays
    """
    bodypart_idx = bodyparts.index(part)
    x_coords = kinematics_matrix[bodypart_idx, :]
    y_coords = kinematics_matrix[len(bodyparts) + bodypart_idx, :]
    return x_coords, y_coords


def get_avg_coordinates(kinematics_matrix, bodyparts):
    """
    Collapses the locations of each bodypart into an "average" location.
    Idea: weight certain bodyparts over others? 
    
    Parameters: 
        kinematics_matrix : numpy.ma.MaskedArray
        bodyparts : list of bodyparts
    Returns:
        np.NDArray
            Average xy coordinates (tuple) for each timepoint
    """
    n_parts = len(bodyparts)
    
    # Extract X and Y coordinate matrices
    x_coords = kinematics_matrix[:n_parts, :]  # Shape: (n_bodyparts, n_timepoints)
    y_coords = kinematics_matrix[n_parts:, :]  # Shape: (n_bodyparts, n_timepoints)
    
    # Compute mean across bodyparts (axis=0) for each timepoint
    x_avg = np.ma.median(x_coords, axis=0)  # Shape: (n_timepoints,)
    y_avg = np.ma.median(y_coords, axis=0)  # Shape: (n_timepoints,)
    
    # Stack into (n_timepoints, 2) array
    avg_coordinates = np.column_stack((x_avg, y_avg))
    
    return avg_coordinates


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
    return tseries.astype('datetime64[s]').astype(float)