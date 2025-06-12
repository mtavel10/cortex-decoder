import numpy as np
import pandas as pd
import src.IO as io

# Idea: class hierarchy of mice, then each mouse has a day (yippeee)
class MouseDay: 
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

        _cal_event_times : numpy.ndarray
        _kin_event_times : numpy.ndarray
        _event_labels : numpy.ndarray

        _cal_spks : numpy.ndarray 

        _kin_dfs : list[tuple[pandas.core.frame.DataFrame, pandas.core.frame.DataFrame]]
            List of 2.5 minute chunks, each chunk has 2 dfs for the 2 camera views
        _kin_mats : list[tuple[numpy.nparray, numpy.nparray]]
            Matrix versions of the dataframes, better for computations

    """
    bodyparts = ['d1middle', 'd2tip', 'd2middle', 'd2knuckle', 'd3tip', 'd3middle',	'd3knuckle', 'd4tip', 'd4middle', 'wrist', 'wrist_outer', 'elbow', 'elbow_crook', 'pellet', 'pedestal', 'p2d1tip']


    def __init__(self, mouseID, day):
        self.mouseID : str = mouseID
        self.day : str = day

        # Load all the data
        self._cal_tseries = io.load_tseries(mouseID, day, "calcium")
        self._kin_tseries = io.load_tseries(mouseID, day, "cam")

        self._cal_event_times = io.load_cal_event_times(mouseID, day)
        self._kin_event_times = io.load_cam_event_times(mouseID, day)
        self._event_labels = io.load_event_labels(mouseID, day)

        self._cal_spks = io.load_spks(mouseID, day)

        self._kin_dfs = {}
        for key in self.kin_event_times:
            self._kin_dfs[key] = io.load_kinematics_df(key, mouseID, day)
        

        self._kin_mats = []
        for df1, df2 in self.kin_dfs.values():
            self._kin_mats.append( (self.create_kinematics_matrix(df1, self.bodyparts, 0.4), 
                                    self.create_kinematics_matrix(df2, self.bodyparts, 0.4)))
            

    # Getter methods for the data to be accessed outside of the class
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
    def kin_dfs(self) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Get the first kinematic dataframes list. Each df covers 2.5 minutes."""
        return self._kin_dfs       

    @property
    def kin_mats(self) -> list[tuple[np.ma.MaskedArray, np.ma.MaskedArray]]:
        """Get the first kinematic masked arrays list. Used more frequently for computations."""
        return self._kin_mats

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

    # Not in use, changed the list of boydparts to a static variable
    def get_bodyparts(self, df):
        # Extract level 1 (bodyparts) and get unique values
        bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()
        return sorted(bodyparts)

    # Used to load in kinematics matrix from dataframes
    def get_x_y(self, df, bp, pcutoff):
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

    def create_kinematics_matrix(self, df, bodyparts, pcutoff):
        """
        This function stacks the x locations for each bodypart on top of the y locations. 
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
        x_ref, y_ref = self.get_x_y(df,'wrist',pcutoff)
        n_timepoints = x_ref.shape[0]
        n_parts = len(bodyparts)

        kinematics_all = np.ma.masked_all([2 * n_parts, n_timepoints])
        # Fill in x, y coordinates for each bodypart
        for j, bodypart in enumerate(bodyparts):
            x, y = self.get_x_y(df, bodypart, pcutoff)
            kinematics_all[j,:] = x 
            kinematics_all[n_parts+j,:] = y
        
        return kinematics_all

    # Unused atm
    def get_bodypart_coordinates(self, kinematics_matrix, bodyparts, part):
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
    def get_avg_coordinates(self, kinematics_matrix) -> np.ndarray:
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
        n_parts = len(self.bodyparts)
        
        # Extract X and Y coordinate matrices
        x_coords = kinematics_matrix[:n_parts, :]  # Shape: (n_bodyparts, n_timepoints)
        y_coords = kinematics_matrix[n_parts:, :]  # Shape: (n_bodyparts, n_timepoints)
        
        # Compute mean across bodyparts (axis=0) for each timepoint
        x_avg = np.ma.median(x_coords, axis=0)  # Shape: (n_timepoints,)
        y_avg = np.ma.median(y_coords, axis=0)  # Shape: (n_timepoints,)
        
        # Stack into (n_timepoints, 2) array
        avg_coordinates = np.column_stack((x_avg, y_avg))
        
        return avg_coordinates


    def interpolate_kin2cal(self, i : int) -> [np.ndarray, np.ndarray]:
        """
        Interpolates the average location of the mouse's hand for calcium frame times. 
        Limited to the ith 2.5 minute chunk of kinematics data (for now). 

        Parameters
            If we can't concatenate all the kinematic matrices during one day, need to specify which 

        Returns 
            2 Numpy NDArrays (2, n_timepoints) (one for each camera)
            Average location, interpolated to calcium time series (each timepoint is a calcium camera frame)
        """
        # List of kinematic matrices to process
        # THIS ONLY WORKS FOR ONE kinematic chunk at a time (why we're grabbing the ith matrices)
        ith_kin_mats = [self._kin_mats[0][i], self._kin_mats[1][i]]
        cam_avg_interps = []
        
        for i, ith_kin_mat in enumerate(ith_kin_mats):
            # Get the average x and y coordinates across all bodyparts
            cam_avg_coordinates = self.get_avg_coordinates(ith_kin_mat)  # maybe later ill modify this function to compute a weighted average
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
            
            # Stack x ontop of y
            cam_avg_interp = np.stack((cam_x_avg_interp, cam_y_avg_interp), axis=0)
            cam_avg_interps.append(cam_avg_interp)
        
        return cam_avg_interps[0], cam_avg_interps[1]