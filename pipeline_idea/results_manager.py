import src.IO as io
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ResultsManager:
    """Handles saving and loading of model results"""
    
    @staticmethod
    def save_results(mouse_id: str, 
                    day: str,
                    scores: List[float],
                    predictions: np.ndarray,
                    model: object,
                    model_type: str = "general",
                    lag: Optional[int] = None):
        """
        Save model results and trained model.
        
        Args:
            mouse_id: Mouse identifier
            day: Day identifier  
            scores: Cross-validation scores
            predictions: Model predictions
            model: Trained model object
            model_type: Type identifier for saving
            lag: Optional lag value for filename
        """
        save_label = ResultsManager._create_save_label(model_type, lag)
        
        logger.info(f"Saving results as '{save_label}' for {mouse_id} day {day}")
        
        io.save_decoded_data(mouse_id, day, scores, predictions, model_type=save_label)
        io.save_model(mouse_id, day, model, model_type=save_label)
    
    @staticmethod
    def _create_save_label(model_type: str, lag: Optional[int]) -> str:
        """Create standardized save label"""
        if lag is not None:
            return f"{model_type}_l{lag}"
        return model_type