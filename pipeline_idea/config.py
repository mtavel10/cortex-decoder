from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    mouseID: str
    day: str
    lag: Optional[int]

@dataclass
class ModelConfig:
    name: str = "ridge" # or "lasso"
    alphas: List[float] = None
    fit_intercept: bool = True
    
    def __post_init__(self):
        if self.alphas is None:
            self.alphas = [0.1, 1.0, 10.0, 100.0]


@dataclass
class CrossValidationConfig:
    # two splitting modes: stratify or balance
    # covariate balancing by behavior
    balance_behaviors: str
    # in class or cross class affects splitting logic, implement this later...
    mode: int
    n_splits: int = 10
    shuffle: bool = True
    random_state: int = 42
    test_size: float = 0.30

@dataclass
class DecoderConfig:
    model: ModelConfig
    cv: CrossValidationConfig
    data: DataConfig
    save_res: bool