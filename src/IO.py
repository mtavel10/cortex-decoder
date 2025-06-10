import numpy as np
import pickle
import glob
import pandas as pd
import utils
import os

def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(filename,var):
    with open(filename, "wb") as file:
        pickle.dump(var, file)

# Dynamic for the system the user is working on
def get_drive(mouseID):
    cwd = os.getcwd()
    drive = cwd[:-4]
    print(drive)
    return drive

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

# camera data
def load_event_labels(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    return np.load(f"{s2p_fld}/camera/event_labels.npy")

# calcium data
def load_spks(mouseID, day):
    s2p_fld = get_s2p_fld(mouseID, day)
    spks = np.load(f"{s2p_fld}/calcium/cascade_spks.npy")
    return spks

# calcium data
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

"""
Cuts off x and y values that are lower then a certain liklihood within a given dataframe. 
"""
def get_x_y(df,bp,pcutoff):
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
    #shape: bodyparts*2 x time (x and y cat)
    x,y = get_x_y(df,'wrist',pcutoff)
    kinematics_all = np.ma.masked_all([2*len(bodyparts),x.shape[0]])
    n_parts = len(bodyparts)
    for j,k in enumerate(bodyparts):
        x,y = get_x_y(df,k,pcutoff)
        #assert (np.ma.max(x)<1450) and (np.ma.max(y)<1100), f'{np.ma.max(x)} {np.ma.max(y)}'
        kinematics_all[j,:] = x #src.utils.low_pass_filt(x,10) NOTE: lfilter doesn't handle masked arrays properly - revise if you decide to filter
        kinematics_all[n_parts+j,:] = y  #src.utils.low_pass_filt(y,10)
    #assert np.ma.max(kinematics_all)<1450, f'{np.ma.max(kinematics_all)} {np.ma.max(x)}'
    return kinematics_all

def load_kinematics_per_bodypart(df,reach_starts,duration,bodyparts,pcutoff,do_filt=False):
    #for a given h5 file based df, get an n bodyparts x n_timepoints x n_reaches matrix per bodypart
    kin_x = np.zeros([len(bodyparts),duration,reach_starts.shape[0]])
    kin_y = np.zeros([len(bodyparts),duration,reach_starts.shape[0]])
    for i,bp in enumerate(bodyparts):
        x,y = get_x_y(df,bp,pcutoff)
        for j,start in enumerate(reach_starts):
            kin_x[i,:,j] = x[start:start+duration]
            kin_y[i,:,j] = y[start:start+duration]
    return kin_x,kin_y

def load_kinematics_per_trial(df,cam_event_times,duration,bodyparts,pcutoff):
    #kinematics with one long row vector per trial (for 1 cam)
    kin_mat = load_kinematics_matrix(df,bodyparts,pcutoff)
    kin_trials = np.ma.masked_all([cam_event_times.shape[0],len(bodyparts)*2*duration])
    for s,start in enumerate(cam_event_times):
        kin_trials[s,:] = kin_mat[:,start:start+duration].flatten()
    #assert np.ma.max(kin_trials)<1450, f'{np.ma.max(kin_trials)}'
    return kin_trials
