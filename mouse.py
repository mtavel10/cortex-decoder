from __future__ import annotations
import numpy as np
import pandas as pd
import src.IO as io
from typing import Dict, List, Tuple, Optional


class MouseDay: 
    """
    Mouse neural and behavioral data for one recording day. 

    Combines calcium imaging (neural spike probabilities) with kinematic tracking
    across multiple 2.5-minute recording segments. 
    """

    BODYPARTS = [
        'd1middle', 'd2tip', 'd2middle', 'd2knuckle', 'd3tip', 'd3middle',	
        'd3knuckle', 'd4tip', 'd4middle', 'wrist', 'wrist_outer', 'elbow', 
        'elbow_crook', 'pellet', 'pedestal', 'p2d1tip'
    ]
    N_PARTS = len(BODYPARTS)
    CUTOFF = 0.4
    BEHAVIOR_LABELS = {
        0: 'reach', 1: 'grasp', 2: 'carry', 3: 'non_movement_or_kept_jumping', 
        4: 'fidget', 5: 'eating', 6: 'grooming', -1: 'non_behavior_event'
    }

    def __init__(self, mouseID: str, day: str, register_cells: bool=False):
        self.mouseID = mouseID
        self.day = day    

        # Load data
        self._load_timestamps()
        self._load_neural_data()
        self._load_behavior_events()
        self._load_kinematics()
        
        if register_cells:
            self._reg_dict = io.load_reg_dict(mouseID, day)           


    # Data Loading
    def _load_timestamps(self):
        """Load calcium and kinematic timestamps."""
        self._cal_tstamp_dict = io.load_tstamp_dict(self.mouseID, self.day, "calcium")
        self._kin_tstamp_dict = io.load_tstamp_dict(self.mouseID, self.day, "cam")
    
    def _load_neural_data(self):
        """Load calcium imaging data."""
        self._cal_spks = io.load_spks(self.mouseID, self.day)
        self._cell_labels = io.load_spk_labels(self.mouseID, self.day)

    def _load_behavior_events(self):
        """Load behavioral event data."""
        self._cal_event_frames = io.load_cal_event_times(self.mouseID, self.day)
        self._kin_event_frames = io.load_cam_event_times(self.mouseID, self.day)
        self._event_labels = io.load_event_labels(self.mouseID, self.day)

    def _load_kinematics(self):
        """Load and process kinematic data."""
        self._kin_dfs = {}
        self._kin_mats = {}

        for key in self.seg_keys:
            # Load dataframes
            self._kin_dfs[key] = io.load_kinematics_df(key, self.mouseID, self.day)
            
            # Convert to matrices
            df1, df2 = self._kin_dfs[key]
            self._kin_mats[key] = (
                self._create_kinematics_matrix(df1),
                self._create_kinematics_matrix(df2)
            )
        
        # Pre-compute interpolated average positions
        self._interpolated_kin_avgs = self._interpolate_all_segments()

    # Properties used within this class
    @property
    def seg_keys(self) -> List[str]:
        """Recording segment keys."""
        return list(self._kin_event_frames.keys())
        
    @property
    def cal_spks(self) -> np.ndarray:
        """Neural spike probability matrix (neurons, bins)."""
        return self._cal_spks

    @property
    def cell_labels(self) -> np.ndarray:
        """Cell type labels (True=inhibitory, False=excitatory)."""
        return self._cell_labels

    @property
    def n_samples(self) -> int:
        """Number of valid neural data samples (excludes first/last 32 timebins)."""
        return self._cal_spks.shape[1] - 64

    @property
    def reg_dict(self) -> Optional[Dict[str, np.ndarray]]:
        """Cell registration dictionary."""
        return getattr(self, '_reg_dict', None)

    @property
    def cal_tstamps(self) -> np.ndarray:
        """Calicum data time stamps (Unix Time per bin)."""
        full_tstamps = []
        for seg in self._cal_tstamp_dict:
            full_tstamps = np.append(full_tstamps, self.cal_tstamp_dict[seg])
        return full_tstamps
    
    @property
    def cal_event_frames(self) -> np.ndarray:
        """Calcium bin numbers where behavioral events occur."""
        return self._cal_event_frames
    
    @property
    def kin_event_frames(self) -> np.ndarray:
        """Kinematic bin numbers where behavioral events occur."""
        return self._kin_event_frames
    
    @property
    def event_labels(self) -> np.ndarray:
        """Behavioral event labels."""
        return self._event_labels

    # For future data screening: On certain session days, 
    # recorded calcium bins differ from number of timestamps. 
    # Causes alignment issues. 
    @property
    def cal_nframes(self) -> int:
        """Number of total calicum data bins"""
        return self.cal_spks[0].shape[0]
    
    @property
    def cal_ntimestamps(self) -> int:
        """Number of total calcium time stamps"""
        return self.cal_tstamps.shape[0]
    
    # Debugging functions for external data screening / validation
    def get_kin_nframes(self, key) -> int:
        return min(len(self.kin_mats[key][0][0]), len(self.kin_mats[key][1][0]))
    
    def get_kin_ntimeframes(self, key) -> int:
        return self.kin_tstamp_dict[key].shape[0]

    def get_trimmed_cal_tstamps(self) -> np.ndarray:
        return self.cal_tstamps[32:-32]

    def check_bin_tstamp_alignment(self):
        """
        Looking into the number of bins vs tstamps per seg key across calcium and camera data
        """
        print("MouseDay: ", self.mouseID, " ", self.day)
    
        print("Caclium Data Comparison")
        for event_key in self.cal_tstamp_dict.keys():
            num_cal_tstamps = self.cal_tstamp_dict[event_key].shape[0]
            print(f"{event_key}: {num_cal_tstamps} tstamps")
        
        print("total cal tstamps: ", self.cal_ntimestamps)

        print("total cal bins: ", self.cal_nframes)
        print()

        print("Kinematic Data Comparison")
        for event_key in self.kin_tstamp_dict.keys(): 
            num_kin_tstamps = self.get_kin_ntimeframes(event_key)
            num_kin_bins = self.get_kin_nframes(event_key)
            print(f"{event_key}: {num_kin_tstamps} tstamps")
            print(f"number of kin bins = (1) {len(self._kin_mats[event_key][0][0])}, (2) {len(self._kin_mats[event_key][1][0])}")
        print()

    def check_caltime_latency(self):
        """
        Checks the differences between each timestamp in the kinematic timeseries. 
        Counts the number of differences that are greater than 35 ms
        """
        count = 0
        for i in range(1, len(self.cal_tstamps)):
            if (self.cal_tstamps[i] - self.cal_tstamps[i-1]) > 35e6:
                count += 1
        print("Number of timestamp gaps greater than 35 ms: ", count)
    
    def latency_check(mouse_day: MouseDay):
        print("# of timestamps (calcium): ", mouse_day.cal_ntimestamps)
        print("# of datapoints (calcium): ", mouse_day.cal_nframes)
        mouse_day.check_caltime_latency()

    def dimensions_check(mouse_day: MouseDay):
        test_locs = mouse_day.get_trimmed_avg_locs()
        test_spikes = mouse_day.get_trimmed_spks()
        test_labels = mouse_day.get_trimmed_beh_labels()

        test_untrimmedlocs = mouse_day.get_all_avg_locations()
        test_untrimmedspks = mouse_day.cal_spks.T
        test_untrimmed_labels = mouse_day.get_beh_labels()

        print("No Trim Locs: ", test_untrimmedlocs.shape)
        print("No Trim Spikes: ", test_untrimmedspks.shape)
        print("No Trim Labels: ", len(test_untrimmed_labels))

        print("Trimmed Locs: ", test_locs.shape)
        print("Trimmed Spikes: ", test_spikes.shape)
        print("Trimmed labels: ", len(test_labels))


    # Core data processing methods
    def _create_kinematics_matrix(self, df: pd.DataFrame) -> np.ma.MaskedArray:
        """
        Convert kinematic DataFrme to matrix format. 

        Returns:
            Masked array of shape (2*n_bodyparts, n_timepoints)
            First n_bodyparts rows: x coordinates
            last n_bodyparts rows: y coordinates

            Ex: 
                        Frame: 0    1    2    3    4
            Row 0 (wrist_X): [120, 125, 130, 135, 140]
            Row 1 (elbow_X): [100, 105, 110, 115, 120] 
            Row 2 (d2tip_X): [150, 155, 160, 165, 170]
                ──────────────────────────────────────
            Row 3 (wrist_Y): [200, 205, 210, 215, 220]
            Row 4 (elbow_Y): [180, 185, 190, 195, 200]
            Row 5 (d2tip_Y): [220, 225, 230, 235, 240]
        """
        # Get reference dimensions to initialize result matrix
        x_ref, y_ref = self._get_x_y(df, 'wrist', self.CUTOFF)
        n_timepoints = x_ref.shape[0]

        kinematics_all = np.ma.masked_all([2 * self.N_PARTS, n_timepoints])
        
        # Fill in coordinates for each bodypart
        for j, bodypart in enumerate(self.BODYPARTS):
            x, y = self._get_x_y(df, bodypart, self.CUTOFF)
            kinematics_all[j,:] = x 
            kinematics_all[self.N_PARTS+j,:] = y
        
        return kinematics_all

    def _get_x_y(self, df: pd.DataFrame, bodypart: str, cutoff: float):
        """Extract and mask coordinates based on likelihood cutoff."""
        prob = df.xs((bodypart, "likelihood"), level=(-2, -1), axis=1).values.squeeze()
        mask = prob < cutoff

        x = np.ma.array(
            df.xs((bodypart, "x"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask,
        )

        y = np.ma.array(
            df.xs((bodypart, "y"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask,
        )

        return x, y


    def _get_avg_coordinates(self, kinematics_matrix) -> np.ndarray:
        """Computes average paw-positions across all bodyparts."""
        x_coords = kinematics_matrix[:self.N_PARTS, :]
        y_coords = kinematics_matrix[self.N_PARTS:, :]
        
        x_avg = np.ma.median(x_coords, axis=0)
        y_avg = np.ma.median(y_coords, axis=0)
        
        return np.column_stack((x_avg, y_avg))

    def _interpolate_segment(self, seg_key: str) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate kinematic averages to calcium timestamps for one segment."""
        kin_mats = self._kin_mats[seg_key]
        kin_tstamps = self._kin_tstamp_dict[seg_key]
        cal_tstamps = self._cal_tstamp_dict[seg_key]

        avg_interps = []

        for cam_matrix in kin_mats:
            avg_coords = self._get_avg_coordinates(cam_matrix)

            # Ensure consistent dimensions
            max_frames = min(len(kin_tstamps), cam_matrix.shape[1])
            x_avg = avg_coords[:max_frames, 0]
            y_avg = avg_coords[:max_frames, 1]
            kin_tstamps_trimmed = kin_tstamps[:max_frames]

            # Interpolate from kinematic to calcium timestamps
            x_interp = np.interp(cal_tstamps, kin_tstamps_trimmed, x_avg)
            y_interp = np.interp(cal_tstamps, kin_tstamps_trimmed, y_avg)

            avg_interps.append(np.stack((x_interp, y_interp), axis=0))

        return tuple(avg_interps)
    
    
    def _interpolate_all_segments(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Interpolate all recording segments."""
        return {seg: self._interpolate_segment(seg) for seg in self.seg_keys}
    
    
    # Public data access methods
    def get_neural_data(self, trim: bool=True, 
                        registered_key: Optional[str]=None) -> np.ndarray:
        """
        Get neural spike data. 
        
        Parameters
            trim: Remove first/last 32 frames (always NaN due to spike estimation algorithm)
            registered_key: If provided, return only registered cells. 
        """
        data = self._cal_spks.T

        if trim: 
            data = data[32:-32]
        
        if registered_key and self.reg_dict:
            registered_neurons = self.reg_dict[registered_key][0]
            data = data[:, registered_neurons]

        return data


    def get_locations(self, trimmed: bool = True) -> np.ndarray:
        """
        Get average paw-locations (x1, y1, x2, y2)

        Returns:
            Array of shape (n_timepoints, 4) with coordinates from both cameras. 
        """
        # Stitch segments together
        all_cam1 = []
        all_cam2 = []

        for cam1_data, cam2_data in self._interpolated_kin_avgs.values():
            all_cam1.append(cam1_data.T)
            all_cam2.append(cam2_data.T)

        all_cam1 = np.concatenate(all_cam1, axis=0)
        all_cam2 = np.concatenate(all_cam2, axis=0)
        all_locs = np.hstack([all_cam1, all_cam2])
        
        if trimmed:
            all_locs = all_locs[32:-32]
        
        return all_locs


    def get_behavior_labels(self, trim: bool=True) -> np.ndarray:
        """ 
        Get behavior labels for each frame. 

        Each behavior event lasts for 8 frames unless interrupted. 
        Non-behavior periods are labeled as -1. 
        """
        max_beh_frames = 8
        # Counter variable tracks whether the frame is during a behavior (>0)
        beh_frame_count = 0
        curr_beh_label = -1
        beh_labels = []

        for frame in range(self._cal_spks.shape[1]):

            event_idx_list = np.where(self.cal_event_frames == frame)[0]
            # There is no event label for this frame
            if len(event_idx_list) == 0:
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
        
        labels = np.array(beh_labels)
        return labels[32:-32] if trim else labels


if __name__ == "__main__":
    mouse_day = MouseDay("mouse25", "20240425")

    neural_data = mouse_day.get_neural_data()
    locations = mouse_day.get_locations()
    behaviors = mouse_day.get_behavior_labels()
    
    print(f"Neural data shape: {neural_data.shape}")
    print(f"Locationsshape: {locations.shape}")
    print(f"Behavior labels shape: {behaviors.shape}")
