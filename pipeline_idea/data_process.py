import numpy as np
import logging
from typing import Tuple, Optional
from mouse import MouseDay

logger = logging.getLogger(__name__)

class DataProcess:
    """
    Handles data loading and basic preprocessing
    
    We apply a lag to the calcium and behavior labels to mimic the biological 
    """
    
    @staticmethod
    def load_data(mouse_day: MouseDay, lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess neural and behavioral data.
        
        Args:
            mouse_day: MouseDay object
            lag: Optional calcium lag in frames
            
        Returns:
            Tuple of (neural_data, behavioral_targets, behavior_labels)
        """
        logger.info(f"Loading data for {mouse_day.mouseID} day {mouse_day.day}")
        
        X = mouse_day.get_trimmed_spks()
        y = mouse_day.get_trimmed_avg_locs()
        labels = mouse_day.get_trimmed_beh_labels()
        
        if lag is not None:
            logger.info(f"Applying calcium lag of {lag} frames")
            X, y, labels = X[lag:], y[:-lag], labels[lag:]
        
        logger.debug(f"Data shapes - Neural: {X.shape}, Behavioral: {y.shape}, Labels: {len(labels)}")
        
        return X, y, labels