from __future__ import annotations
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
    Here is the data from that day. 

        class vars
            BODYPARTS : list[str]
                list of the bodyparts recorded by the kinematic cameras
                static

            N_PARTS : int
                number of bodyparts
            
            CUTOFF : int
                minimum confidence to keep kinematic data

        protected vars
            _mouseID : str
                    ex- "mouse25"
            _day : str
                YYYYMMDD

            _seg_keys : list[str]
                A list of keys for 2.5 min segments of data collected during this day
                Each key is formatted by time and event e.g. "133901event001"

            _cal_tstamps : np.ndarray[float]
                Time (ns) since Unix Epoch per calcium camera frame
            _cal_tstamp_dict : dict { "segkey", np.ndarray[int]}
                For each 2.5 minute chunk, time (ns) since Unix Epoch per calcium camera frame
            _kin_tstamp_dict : dict { "segkey", np.ndarray[int]}
                For each 2.5 minute chunk, time (ns) since Unix Epoch per kinematic camera frame

            _cal_spks : numpy.ndarray 
                Spike probabilities for each neuron at each timepoint

            _kin_dfs : dict { "segkey , tuple[pandas.DataFrame, pandas.DataFrame]}
                List of 2.5 minute chunks, each chunk has 2 dfs for the 2 camera views
            _kin_mats : dict { "segkey", tuple[numpy.ndarray, numpy.ndarray]}
                Matrix versions of the dataframes, better for computations
            
            _cal_event_frames : numpy.ndarray
                Indicates the CALCIUM CAMERA FRAME when each event occurred (indexed by event)
            
            _kin_event_frames : dict { "segkey", np.ndarray }
                For each 2.5 minute chunk, indicates the KINEMATIC CAMERA FRAME when each event occurred
            _event_labels : numpy.ndarray
                Indicate the type of event when each event occurred (indexed by event)

    """
    BODYPARTS = ['d1middle', 'd2tip', 'd2middle', 'd2knuckle', 'd3tip', 'd3middle',	'd3knuckle', 'd4tip', 'd4middle', 'wrist', 'wrist_outer', 'elbow', 'elbow_crook', 'pellet', 'pedestal', 'p2d1tip']
    N_PARTS = len(BODYPARTS)
    CUTOFF = 0.4
    BEHAVIOR_LABELS = {0: 'reach', 1: 'grasp', 2: 'carry', 3: 'non_movement_or_kept_jumping', 4: 'fidget', 5: 'eating', 6: 'grooming', -1: 'non_behavior_event'}

    def __init__(self, mouseID, day):
        self.mouseID : str = mouseID
        self.day : str = day                

        # Load all the data
        # # Keeping these for debugging purposes as I scale up to all time series
        # self._cal_tseries = io.load_tseries(mouseID, day, "calcium")
        # self._kin_tseries = io.load_tseries(mouseID, day, "cam")

        self._cal_tstamp_dict = io.load_tstamp_dict(mouseID, day, "calcium")
        self._kin_tstamp_dict = io.load_tstamp_dict(mouseID, day, "cam")

        # self._cal_tstamps = io.load_cal_tstamps(mouseID, day)
        self._cal_event_frames = io.load_cal_event_times(mouseID, day)
        self._kin_event_frames = io.load_cam_event_times(mouseID, day)
        self._event_labels = io.load_event_labels(mouseID, day)

        self._cal_spks = io.load_spks(mouseID, day)
        self._cell_labels = io.load_spk_labels(mouseID, day)

        self._kin_dfs = {}
        for key in self.seg_keys:
            self._kin_dfs[key] = io.load_kinematics_df(key, mouseID, day)

        self._kin_mats = {}
        for key in self.seg_keys:
            df1, df2 = self._kin_dfs[key]  # Assuming load_kinematics_df returns a tuple of two dataframes
            self._kin_mats[key] = (self.create_kinematics_matrix(df1), self.create_kinematics_matrix(df2))
        
        self._interpolated_kin_avgs = self.interpolate_all("avg")

    @property
    def cal_tstamps(self) -> np.ndarray:
        """Get the calcium time series data."""
        full_tstamps = []
        for seg in self.cal_tstamp_dict:
            full_tstamps = np.append(full_tstamps, self.cal_tstamp_dict[seg])
        return full_tstamps

    def check_caltime_latency(self):
        """
        Checks the differences between each timestamp in the kinematic timeseries. Counts the number of differences that are greater than 35 ms
        """
        count = 0
        for i in range(1, len(self.cal_tstamps)):
            if (self.cal_tstamps[i] - self.cal_tstamps[i-1]) > 35e6:
                count += 1
        print("Number of timestamp gaps greater than 35 ms: ", count)
    
    @property
    def cal_tstamp_dict(self) -> dict [str, np.ndarray]:
        """Get the calcium time series data."""
        return self._cal_tstamp_dict

    @property
    def kin_tstamp_dict(self) -> dict [str, np.ndarray]:
        """Get the kinematic time series data."""
        return self._kin_tstamp_dict
    
    @property
    def cal_spks(self) -> np.ndarray:
        """Get the calibration spikes data."""
        return self._cal_spks
    
    @property
    def cell_labels(self) -> np.ndarray:
        """ True for inhibitory, False for excitatory """
        return self._cell_labels
    
    @property
    def kin_dfs(self) -> dict [str : tuple[pd.DataFrame, pd.DataFrame]]:
        """Get the first kinematic dataframes list. Each df covers 2.5 minutes."""
        return self._kin_dfs       

    @property
    def kin_mats(self) -> dict [str : tuple[np.ndarray, np.ndarray]]:
        """Get the first kinematic masked arrays list. Used more frequently for computations."""
        return self._kin_mats

    @property
    def cal_event_frames(self) -> np.ndarray:
        """Get the calibration event times."""
        return self._cal_event_frames
    
    @property
    def kin_event_frames(self) -> np.ndarray:
        """Get the kinematic event times."""
        return self._kin_event_frames
    
    @property
    def event_labels(self) -> np.ndarray:
        """Get the event labels."""
        return self._event_labels

    # The list of event keys for this day
    @property
    def seg_keys(self) -> list[str]:
        return list(self.kin_event_frames.keys())

    # Calcium camera frames differ from number of timestamps
    @property
    def cal_nframes(self) -> int:
        return len(self.cal_spks[0])
    
    @property
    def cal_ntimestamps(self) -> int:
        return len(self.cal_tstamps)
    
    @property
    def n_samples(self) -> int:
        """
        MOST IMPORTANT - the number of valid samples in this dataset
        Calcium probability estimation algorithm doesn't compute for the first and last 32 frames, so this is the number of valid timepoints of data
        """
        return self.cal_nframes - 64
    
    # Number of frames varies per recording segment... should these even be properties?
    # Assumes the frames are uniform across cameras
    def get_kin_nframes(self, key) -> int:
        return len(self.kin_mats[key][0][0])
    
    def get_kin_ntimeframes(self, key) -> int:
        return len(self.kin_tstamp_dict[key])

    def get_trimmed_cal_tstamps(self) -> np.ndarray:
        return self.cal_tstamps[32:-32]

    # Not in use, changed the list of boydparts to a static variable
    def get_bodyparts(self, df):
        # Extract level 1 (bodyparts) and get unique values
        bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()
        return sorted(bodyparts)

    # Helper function for create_kinematics_matrix. 
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

    def create_kinematics_matrix(self, df):
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
        x_ref, y_ref = self.get_x_y(df,'wrist',self.CUTOFF)
        n_timepoints = x_ref.shape[0]

        kinematics_all = np.ma.masked_all([2 * self.N_PARTS, n_timepoints])
        # Fill in x, y coordinates for each bodypart
        for j, bodypart in enumerate(self.BODYPARTS):
            x, y = self.get_x_y(df, bodypart, self.CUTOFF)
            kinematics_all[j,:] = x 
            kinematics_all[self.N_PARTS+j,:] = y
        
        return kinematics_all
    
    def get_x_coords(self, kinematics_matrix : np.ndarray) -> np.ndarray:
        """ Returns the top half of a kinematics matrix (the x coordinates of each bodypart) """
        # Shape: (n_bodyparts, n_timepoints)
        return kinematics_matrix[:self.N_PARTS, :]

    def get_y_coords(self, kinematics_matrix: np.ndarray) -> np.ndarray:
        """ Returns the bottom half of a kinematics matrix (the y coordinates of each bodypart) """
        # Shape: (n_bodyparts, n_timepoints)
        return kinematics_matrix[self.N_PARTS:, :]

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
            maybe i should change to a tuple of n_timepoints??
        """
        
        # Extract X and Y coordinate matrices
        x_coords = self.get_x_coords(kinematics_matrix)
        y_coords = self.get_y_coords(kinematics_matrix)
        
        # Compute mean across bodyparts (axis=0) for each timepoint
        x_avg = np.ma.median(x_coords, axis=0)  # Shape: (n_timepoints,)
        y_avg = np.ma.median(y_coords, axis=0)  # Shape: (n_timepoints,)
        
        # Stack into (n_timepoints, 2) array
        avg_coordinates = np.column_stack((x_avg, y_avg))
        
        return avg_coordinates

    def interpolate_avgkin2cal(self, key) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolates the average location of the mouse's hand for calcium frame times. 
        Limited to the 2.5 minute chunk of kinematics data specified by the key. 
        For later: Maybe save this as new "interpolated" kinematics matrices to simplify decoder (can just perform operations on the interpolated data)

        Parameters
            If we can't concatenate all the kinematic matrices during one day, need to specify which 

        Returns 
            2 Numpy NDArrays (2, n_timepoints) (one for each camera)

            (calcium frames)   0   1   2   3   4 ...
                    x_avg      
                    y_avg

            Average location, interpolated to calcium time series (each timepoint is a calcium camera frame)
        """
        # List of kinematic matrices to process
        # THIS ONLY WORKS FOR ONE 2.5 min chunk at a time
        curr_kin_mats = self.kin_mats[key]
        curr_kin_tstamps = self.kin_tstamp_dict[key]
        # print(curr_kin_tstamps)
        curr_cal_tstamps = self.cal_tstamp_dict[key]
        # print(curr_cal_tstamps)
        avg_interps = []
    
        for cam in curr_kin_mats:
            # Get the average x and y coordinates across all bodyparts
            avg_coordinates = self.get_avg_coordinates(cam) # maybe later ill modify this function to compute a weighted average ("true centroid")
            # print(avg_coordinates)
            x_avg = avg_coordinates[:, 0]
            y_avg = avg_coordinates[:, 1]
            
            # Resizing the kinematics-camera frames to match the kinematics time series
            max_frames = min(self.get_kin_ntimeframes(key), self.get_kin_nframes(key))
            x_avg = x_avg[:max_frames]
            y_avg = y_avg[:max_frames]
            cam_tstamps = curr_kin_tstamps[:max_frames]
            # Interpolate!
            x_avg_interp = np.interp(curr_cal_tstamps, cam_tstamps, x_avg)
            y_avg_interp = np.interp(curr_cal_tstamps, cam_tstamps, y_avg)
            
            # Stack x ontop of y
            avg_interp = np.stack((x_avg_interp, y_avg_interp), axis=0)
            avg_interps.append(avg_interp)
        
        return tuple(avg_interps)
    
    
    def interpolate_all(self, features : str) -> {str : tuple[np.ndarray, np.ndarray]}:
        kin_avg_interp = {}
        for seg in self.seg_keys:
            if features == "avg":
                kin_avg_interp[seg] = self.interpolate_avgkin2cal(seg)
        
        # print(kin_avg_interp)
        # np.save("{mouseID}/interpolated_avgs.npy", kin_avg_interp)
        return kin_avg_interp
    
    def get_all_avg_locations(self):
        """
        Stiches together all average kinematic positions from each recording segment. 
        Returns
            all_avg_locations (n_timepoints, 4) : ndarray
                x1 y1 x2 y2
                one coordinate pair per camera view

        """
        all_cam1 = []
        all_cam2 = []
        for seg, cam_locs in self._interpolated_kin_avgs.items():
            cam1_locs, cam2_locs = cam_locs
            all_cam1.append(cam1_locs.T)
            all_cam2.append(cam2_locs.T)

         # Concatenate all segments (row-wise) so they're stacked ontop of each other
        all_cam1 = np.concatenate(all_cam1, axis=0)
        all_cam2 = np.concatenate(all_cam2, axis=0)

        all_avg_locations = np.hstack([all_cam1, all_cam2])
        return all_avg_locations

    def get_trimmed_spks(self) -> np.ndarray:
        """
        Returns a numpy array of size (n_timepoints x n_neurons)
        Represents the estimated spike probabilities across all timepoints
        """
        trimmed_arr = self.cal_spks[:, 32:-32].T
        return trimmed_arr

    def get_trimmed_avg_locs(self):
        """
        Returns a numpy array of size (n_timepoints x 4 locations)
        Represents the average x and y locations of the mouse's hand for two camera views
        """
        # First and last 32 frames are NaN because of spike probability estimate algorithm
        trimmed1 = self.get_all_avg_locations()[32:-32]
        return trimmed1
    
    # FOR MONDAY: TEST THIS FUNCTION
    def get_beh_labels(self):
        """ 
        Returns a 1D numpy array of behavior labels for all valid timepoints (calcium frames) for the day. 
        Each event frame indicates the "start" of a behavior, so it sets the subsequent 8 frames to that label (unless interrupted by another behavior)
        Handles "non-event" timepoints by setting the label to -1. 
        """
        max_beh_frames = 8
        # Counter variable tracks whether the frame is during a behavior (>0)
        beh_frame_count = 0
        curr_beh_label = -1
        beh_labels = []
        for frame in range(0, int(self.cal_nframes)):

            event_idx_list = np.where(self.cal_event_frames == frame)[0]
            # There is no event label for this frame
            if not event_idx_list:
                beh_labels.append(curr_beh_label)
                # An ongoing behavior
                if beh_frame_count > 0:
                    beh_frame_count += 1
            # A new event starts at this frame
            else:
                event_idx = event_idx_list[0]
                curr_beh_label = self.event_labels[event_idx]
                beh_labels.append(curr_beh_label)
                # Checks whether we're interrupting an ongoing event
                if beh_frame_count == 0:
                    beh_frame_count += 1
                else:
                    beh_frame_count = 1
    
            if beh_frame_count == max_beh_frames:
                curr_beh_label = -1
                beh_frame_count = 0
        
        return np.array(beh_labels)
    
    def get_trimmed_beh_labels(self):
        return self.get_beh_labels()[32:-32]