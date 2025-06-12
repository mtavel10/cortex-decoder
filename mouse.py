import numpy as np
import pandas as pd
import src.IO as io

# Idea: class hierarchy of mice, then each mouse has a day (yippeee)
class mouse_day: 
    """
    Imagine one day you woke up from brain surgery as a mouse. 
    You're hooked up to a dark apparatus and in front of you is a crumb. 
    You reach out to the crumb, grab it, then another appears. 
    You reach again.
    Two cameras film your hand movements.
    Another tracks calcium spikes through photons in your brain. 
    Here is the data. 

        mouseID : str
            ex- "mouse25"
        day : str
            YYYYMMDD
        
        _bodyparts : list[str]
            list of the bodyparts recorded by the kinematic cameras

        _cal_tseries : np.ndarray
        _kin_tseries : np.ndarray

        _cal_spks : numpy.ndarray 

        # ask Gabriella how we can change the storage
        _kin_dfs : tuple[list[pandas.core.frame.DataFrame], list[pandas.core.frame.DataFrame])
            Each df covers 2.5 minutes
        _kin_mats : tuple[list[numpy.ma.MaskedArray], list[numpy.ma.MaskedArray]]
            Matrix versions of the dataframes, better for computations
        
        _cal_event_times : numpy.ndarray
        _kin_event_times : numpy.ndarray
        _event_labels : numpy.ndarray

    """

    def __init__(self, mouseID, day, start_time):
        self.mouseID : str = mouseID
        self.day : str = day

        # Load all the data
        self._cal_tseries = io.load_tseries(mouseID, day, "calcium")
        self._kin_tseries = io.load_tseries(mouseID, day, "cam")

        self._cal_spks = io.load_spks(mouseID, day)
        # This is where we use the start time
        # How do we determinue the number of events (i.e. number of kinematic data frames per mouse)
        # PLACEHOLDER: LOAD IN KIN DATA FRAMES AND MATRICES
        self._

        
        self._bodyparts = get_bodyparts(kin_df1)

        self._cal_event_times = io.load_cal_event_times(mouseID, day)
        self._kin_event_times = io.load_cam_event_times(mouseID, day)
        self._event_labels = io.load_event_labels(mouseID, day)

    # Getter methods for the data to be accessed outside of the class
    @property
    def bodyparts(self) -> list[str]:
        """Get the list of bodyparts tracked in kinematics data"""
        return self._bodyparts

    @property
    def cal_tseries(self) -> np.ndarray:
        """Get the calibration time series data."""
        return self._cal_tseries
    
    @property
    def kin_tseries(self) -> np.ndarray:
        """Get the kinematic time series data."""
        return self._kin_tseries
    
    @property
    def cal_spks(self) -> np.ndarray:
        """Get the calibration spikes data."""
        return self._cal_spks
    
    @property
    def kin_df1(self) -> list[pd.DataFrame]:
        """Get the first kinematic dataframes list. Each df covers 2.5 minutes."""
        return self._kin_df1
    
    @property
    def kin_df2(self) -> list[pd.DataFrame]:
        """Get the second kinematic dataframes list."""
        return self._kin_df2
    
    @property
    def kin_mat1(self) -> list[np.ma.MaskedArray]:
        """Get the first kinematic masked arrays list. Used more frequently for computations."""
        return self._kin_mat1
    
    @property
    def kin_mat2(self) -> list[np.ma.MaskedArray]:
        """Get the second kinematic masked arrays list."""
        return self._kin_mat2
    
    @property
    def cal_event_times(self) -> np.ndarray:
        """Get the calibration event times."""
        return self._cal_event_times
    
    @property
    def kin_event_times(self) -> np.ndarray:
        """Get the kinematic event times."""
        return self._kin_event_times
    
    @property
    def event_labels(self) -> np.ndarray:
        """Get the event labels."""
        return self._event_labels


    def get_bodyparts(df):
        # Extract level 1 (bodyparts) and get unique values
        bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()
        return sorted(bodyparts)


    def get_x_y(df,bp,pcutoff):
        """
        Masks x and y values that are lower then a certain liklihood within a given dataframe. 
        Helper function to mask certain values when loading the kinematics matrix. 
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


    def create_kinematics_matrix(df,bodyparts,pcutoff):
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

    # Unused atm
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

    # Helper function for interpolating
    def get_avg_coordinates(kinematics_matrix) -> np.ndarray:
        """
        Collapses the locations of each bodypart into an "average" location.
        Idea: weight certain bodyparts over others? 
        
        Parameters: 
            kinematics_matrix : numpy.ma.MaskedArray
            bodyparts : list of bodyparts
        Returns:
            np.NDArray (n_timepoints, 2)
                Average xy coordinates (tuple) for each timepoint
        """
        n_parts = len(self._bodyparts)
        
        # Extract X and Y coordinate matrices
        x_coords = kinematics_matrix[:n_parts, :]  # Shape: (n_bodyparts, n_timepoints)
        y_coords = kinematics_matrix[n_parts:, :]  # Shape: (n_bodyparts, n_timepoints)
        
        # Compute mean across bodyparts (axis=0) for each timepoint
        x_avg = np.ma.median(x_coords, axis=0)  # Shape: (n_timepoints,)
        y_avg = np.ma.median(y_coords, axis=0)  # Shape: (n_timepoints,)
        
        # Stack into (n_timepoints, 2) array
        avg_coordinates = np.column_stack((x_avg, y_avg))
        
        return avg_coordinates


    # Interpolate method
    def interpolate_kin2cal(i : int) -> [np.ndarray, np.ndarray]:
        """
        Interpolates the average location of the mouse's hand during the calcium frame times. 
        Parameters
            If we can't concatenate all the kinematic matrices during one day, need to specify which 

        Returns 
            2 Numpy NDArrays (2, n_timepoints) (one for each camera)
            Average location, interpolated to calcium time series (each timepoint is a calcium camera frame)
        """
        # List of kinematic matrices to process
        # THIS ONLY WORKS FOR ONE kinematic matrix at a time (why we're grabbing the ith matrix)
        ith_kin_mats = [self._kin_mats[0][i], self._kin_mats[1][i]]
        cam_avg_interps = []
        
        for i, kin_mat in enumerate(ith_kin_mats):
            # Get the average x and y coordinates across all bodyparts
            cam_avg_coordinates = get_avg_coordinates(ith_kin_mat)  # maybe later ill modify this function to compute a weighted average
            cam_x_avg = cam_avg_coordinates[:, 0]
            cam_y_avg = cam_avg_coordinates[:, 1]
            
            # Resizing the kinematics-camera frames to match the kinematics time series
            cam_frames = len(self._kin_tseries)
            kin_frames = len(cam_x_avg)
            min_frames = min(cam_frames, kin_frames)
            
            cam_x_avg = cam_x_avg[:min_frames]
            cam_y_avg = cam_y_avg[:min_frames]
            cam_tseries = self._kin_tseries[:min_frames]
            
            # Interpolate!
            cam_x_avg_interp = np.interp(self._cal_tseries, cam_tseries, cam_x_avg)
            cam_y_avg_interp = np.interp(self._cal_tseries, cam_tseries, cam_y_avg)
            
            # Smush together
            cam_avg_interp = np.stack((cam_x_avg_interp, cam_y_avg_interp), axis=0)
            cam_avg_interps.append(cam_avg_interp)
        
        return cam_avg_interps[0], cam_avg_interps[1]