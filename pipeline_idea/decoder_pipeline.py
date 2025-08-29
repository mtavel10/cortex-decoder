from typing import Tuple, List
import numpy as np
import logging
from mouse import MouseDay
from config import DecoderConfig, ModelConfig, CrossValidationConfig, DataConfig
from data_process import DataProcess
from pipeline_idea.results_manager import ResultsManager
from pipeline_idea.cross_validatorross_validator import CrossValidator
from pipeline_idea.models import ModelFactory

logger = logging.getLogger(__name__)

class DecoderPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: DecoderConfig):
        self.config = DecoderConfig
        self.data_loader = DataProcess()
        self.model_factory = ModelFactory()
        self.cross_validator = CrossValidator(config.cv)
        self.results_manager = ResultsManager()
        
    def decode_general(self, mouse_day: MouseDay) -> Tuple[List[float], np.ndarray]:
        """
        Decodes general neural data and predicts all mouse-paw positions for
        a mouse's daily training session. 

        Args:
            mouse_day: MouseDay object containing neural and behavioral data
            
        Returns:
            Tuple of (cross_validation_scores, predictions)
        """
        try:
            # 1. Load and preprocess data
            X, y, labels = self.data_loader.load_data(
                mouse_day, self.config.lag
            )
            
            # 2. Create model
            model = self.model_factory.create_model(self.config.model)
            
            # 3. Cross-validate
            scores, predictions = self.cross_validator.validate_model(
                model, X, y, labels
            )
            
            # 4. Save results if requested
            if self.config.save_results:
                self.results_manager.save_results(
                    mouse_day.mouseID,
                    mouse_day.day,
                    scores,
                    predictions,
                    model,
                    model_type="general",
                    lag=self.config.lag
                )
            
            logger.info("General decoding pipeline completed successfully")
            return scores, predictions
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create configuration
    config = DecoderConfig(
        model=ModelConfig(name="ridge", alphas=[0.1, 1.0, 10.0, 100.0]),
        cv=CrossValidationConfig(n_splits=10),
        data=DataConfig(mouseID="mouse25", day="20240425")
    )
    
    # Create and run pipeline
    pipeline = DecoderPipeline(config)
    
    # Load your mouse data
    mouse_day = MouseDay("mouse25", "20240425")
    
    # Run the pipeline
    scores, predictions = pipeline.decode_general(mouse_day)
    print("scores: ", scores)