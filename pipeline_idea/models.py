from sklearn.linear_model import RidgeCV, MultiTaskLassoCV
from sklearn.base import BaseEstimator
from config import ModelConfig
from typing import List, Tuple
import numpy as np
import logging as logger

class ModelFactory:
    @staticmethod
    def create_model(config: ModelConfig) -> BaseEstimator:
        logger.info(f"Creating {config.name} model with alphas {config.alphas}")
        
        models = {
            "ridge": lambda: RidgeCV(alphas=config.alphas, 
                                    fit_intercept=config.fit_intercept),
            "lasso": lambda: MultiTaskLassoCV(alphas=config.alphas)
        }
        
        if config.name not in models:
            logger.warning(f"Unknown model {config.name}, defaulting to ridge")
            config.name = "ridge"
            
        return models[config.name]()