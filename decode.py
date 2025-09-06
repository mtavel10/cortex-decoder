"""
Motor Cortex Decoder for Kinematic Behavior Analysis
==================================================

A comprehensive pipeline for decoding paw positions from calcium imaging data
across different behavioral contexts and interneuron cell types. 

Author: Maddy Tavel
Summer Research Project - Maclean Lab @ University of Chicago
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV, MultiTaskLassoCV

import src.IO as io
from mouse import MouseDay

TEST_SIZE = .30
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

BEHAVIOR_CLASSES = {
    "all": [0, 1, 2, 3, 4, 5], 
    "learned": [0, 1, 2], 
    "natural": [3, 4, 5], 
    "reach": [0], 
    "grasp": [1], 
    "carry": [2], 
    "non_movement": [3], 
    "fidget": [4], 
    "eating": [5]
}

LEARNED_BEHAVIORS = ["reach", "grasp", "carry"]
NATURAL_BEHAIORS = ["non_movement", "fidget", "eating"]

logger = logging.getLogger(__name__)

class CortexDecoder:
    """Predicts mouse-paw positions from calcium spike data."""

    def __init__(self, log_level: logging._LevelLEVEL=logging.INFO, 
                 alphas: List[float]=None, random_state: int=42):
        """
        Initialize decoder with hyperparameters. 

        Parameters
        ----------
        alphas: List[float], Optional
            Regularization parameters for model cross-validation
        random_state: int
            Random seed for reproducibility
        """
        logger.setLevel(log_level)
        self.alphas = alphas or ALPHAS
        self.random_state = random_state

    def _apply_calcium_lag(self, X: np.ndarray, y: np.ndarray, 
                        beh_labels: np.ndarray, lag: int) -> Tuple[np.ndarray, 
                                                                    np.ndarray, 
                                                                    np.ndarray]:
        """Apply temporal lag to account for calcium delay in neural spiking."""
        if lag is None:
            return X, y, beh_labels
        
        logger.info(f"Applying calcium lag of {lag} bins...")
        return X[lag:], y[:-lag], beh_labels[lag:]
        
    def _create_model(self, model_type: str) -> Any:
        """Factory method for initializing regression models."""
        if model_type.lower() == "lasso":
            return MultiTaskLassoCV(alphas=self.alphas)
        else:
            return RidgeCV(alphas=self.alphas, fit_intercept=True)


    def decode_general_population(self, mouse_day: MouseDay, model_type="ridge",
                                lag: Optional[int]=None, n_trials: int=10, 
                                save_results=False) -> Tuple[List[float], np.ndarray]:
        """
        Predict paw positions using the entire neural population. 

        Parameters
        ----------
        mouse_day: MouseDay
            Data class for single experimental session. 
        model_type: str
            Regression model type ("ridge" or "lasso")
        lag: int, Optional
            Calcium lag in frames
        n_trials: int
            Number of cross-validation folds
        save_results: bool
            Whether to save model outputs
        
        Returns
        -------
        scores: List[float]
            Cross-validation R^2 scores
        predictions: np.ndarray
            Position predictions for all timepoints
        """
        # Load and preprocess data
        X = mouse_day.get_trimmed_spks()
        y = mouse_day.get_trimmed_avg_locs()
        behavior_labels = mouse_day.get_trimmed_beh_labels()

        X, y, behavior_labels = self._apply_calcium_lag(X, y, behavior_labels, lag)

        logger.debug(f"Data shapes - Neural spikes: {X.shape}, Positions: {y.shape}")

        # Cross-validation
        splitter = StratifiedKFold(n_splits=n_trials, shuffle=True, random_state=self.random_state)
        model = self._create_model(model_type="ridge")

        scores = []
        fold_predictions = []
        
        for fold, (train_idcs, test_idcs) in enumerate(splitter.split(X, behavior_labels)):
            logger.info(f"Processing fold {fold+1}/{n_trials}")
            
            X_train, X_test = X[train_idcs], X[test_idcs]
            y_train, y_test = y[train_idcs], y[test_idcs]

            # Train and evaluate model
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))

            # Store predictions and indices for reconstruction
            y_pred_fold = model.predict(X_test)
            fold_predictions.append((test_idcs, y_pred_fold))
        
        # Reconstruct full prediction array
        y_predictions = np.zeros_like(y)
        for test_idcs, preds in fold_predictions:
            y_predictions[test_idcs] = preds
        
        # Save results if requested
        if save_results:
            save_label = f"general_{model_type}" + (f"_lag{lag}" if lag else "")
            io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_predictions, model_type=save_label)
            io.save_model(mouse_day.mouseID, mouse_day.day, model, model_type=save_label)

        return scores, y_predictions


    def decode_by_behavior(self, mouse_day: MouseDay, model_type: str="ridge", 
                           lag: Optional[int]=None, n_trials: int=10, 
                           save_results: bool=False) -> Tuple[List[List[float]], List[np.ndarray]]:
        """
        Train separate decoders for each behavior. 

        Returns
        -------
        all_scores: List[List[float]]
            R^2 scores for each behavior
        all_predictions: List[np.ndarray]
            Predictions for each behavior
        """
        X = mouse_day.get_trimmed_spks()
        y = mouse_day.get_trimmed_avg_locs()
        behavior_labels = mouse_day.get_trimmed_beh_labels()

        X, y, behavior_labels = self._apply_calcium_lag(X, y, behavior_labels, lag)

        # Filter out behaviors with small sample sizes (grooming, for now)
        valid_behaviors = {k: v for k, v in mouse_day.BEHAVIOR_LABELS.items() if k != 6}
       
        all_scores = []
        all_predictions = []

        for behavior_id, behavior_name in valid_behaviors.items():
            logger.info(f"Training decoder for {behavior_name}...")
            
            # Extract samples for current behavior
            behavior_mask = behavior_labels == behavior_id
            X_behavior = X[behavior_mask]
            y_behavior = y[behavior_mask]

            if len(behavior_mask) == 0:
                logger.warning(f"Warning: No data found for behavior {behavior_name}")
                continue

            # Cross-validation within behavior
            splitter = KFold(n_splits=n_trials, shuffle=True, random_state=self.random_state)
            model = self._create_model(model_type="ridge")
            
            behavior_scores = []
            for fold, (train_idcs, test_idcs) in enumerate(splitter.split(X_behavior)):
                logger.info(f"Processing fold {fold+1}/{n_trials}")
                X_train, X_test = X_behavior[train_idcs], X_behavior[test_idcs]
                y_train, y_test = y_behavior[train_idcs], y_behavior[test_idcs]

                # Train and evaluate model
                model.fit(X_train, y_train)
                behavior_scores.append(model.score(X_test, y_test))

            # Generate predictions on full dataset using final model
            y_pred = model.predict(X)

            all_scores.append(behavior_scores)
            all_predictions.append(y_pred)

            # Save individual behavior results
            if save_results:
                # saves the scores and predictions for plotting
                save_label = f"{behavior_name}_{model_type}" + (f"_lag{lag}" if lag else "")
                io.save_decoded_data(mouse_day.mouseID, mouse_day.day, behavior_scores, y_pred, model_type=save_label)
                io.save_model(mouse_day.mouseID, mouse_day.day, model, model_type=save_label)

        return all_scores, all_predictions


    def decode_cross_behaviors(self, mouse_day: MouseDay, n_trials: int=10, 
                               save_results: bool=False) -> Dict[int, List[float]]:
        """
        Train on general population, test on individual behaviors. 
        Measures model generalization across behavioral contexts. 

        Returns
        -------
        scores_by_behavior: Dict[int, List[float]]
            R^2 scores for each behavior across CV folds
        """
        X = mouse_day.get_trimmed_spks()
        y = mouse_day.get_trimmed_avg_locs()
        behavior_labels = mouse_day.get_trimmed_beh_labels()
        
        # 1. Create balanced test sets for each behavior
        valid_behaviors = {k: v for k, v in mouse_day.BEHAVIOR_LABELS.items() if k != 6}
        behavior_data = {}
        min_samples = float('inf')

        # Organize data by behavior and find minimum sample count
        for behavior_id in valid_behaviors.keys():
            mask = behavior_labels == behavior_id
            behavior_data[behavior_id] = {
                'X': X[mask],
                'y': y[mask],
                'indices': np.where(mask)[0],
                'n_samples': len(X[mask])
            }
            min_samples = min(min_samples, behavior_data[behavior_id]['n_samples'])

        # Calculate balanced split sizes
        test_samples_per_behavior = int(TEST_SIZE * min_samples)
        train_samples_per_behavior = min_samples - test_samples_per_behavior

        logger.debug(f"Using {train_samples_per_behavior} training \
                    and {test_samples_per_behavior} test samples per behavior")

        scores_by_behavior = {behavior_id: [] for behavior_id in valid_behaviors.keys()}

        # 2. Cross-validate a model with behavior-balanced sets
        for fold in range(n_trials):
            logger.info(f"Processing fold {fold+1}/{n_trials}")
            np.random.seed(42 + fold)
            
            # Create balanced training set
            X_train_list = []
            y_train_list = []
            test_sets = {}

            for behavior_id, data in behavior_data.items():
                # Shuffle and split data
                indices = np.random.permulation(behavior_data[behavior_id]['n_samples'])

                train_end = train_samples_per_behavior
                test_end = train_end + test_samples_per_behavior

                # Add to training set
                X_train_list.append(data['X'][indices[:train_end]])
                y_train_list.append(data['y'][indices[:train_end]])
                
                # Store test set
                test_sets[behavior_id] = {
                    'X': data['X'][indices[train_end:test_end]],
                    'y': data['y'][indices[train_end:test_end]]
                }

            # Combine and shuffle training data
            X_train = np.vstack(X_train_list)
            y_train = np.vstack(y_train_list)
            train_indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[train_indices], y_train[train_indices]

            # Train model on general population
            model = self._create_model(model_type="ridge")
            model.fit(X_train, y_train)

            # Test on each behavior
            for behavior_id, test_data in test_sets.items():
                score = model.score(test_data['X'], test_data['y'])
                scores_by_behavior[behavior_id].append(score)

        if save_results:
            io.save_scores_by_beh(mouse_day.mouseID, mouse_day.day, scores_by_behavior)
        
        return scores_by_behavior


def decode_by_cell_type(self, mouse_day: MouseDay, n_trials: int=10, 
                        save_results: bool=False) -> Tuple[List[float], List[float],
                                                           np.ndarray, np.ndarray]:
    """
    Compare decoder performance between excitatory and inhibitory neurons. 

    Returns
    -------
    inhibitory_scores: List[float]
        R^2 scores for inhibitory neurons
    excitatory_scores: List[float]
        R^2 scores for excitatory neurons
    inhibitory_predictions: np.ndarray
        Position predictions from inhibitory neurons
    excitatory_predictions: np.ndarray
        Position predictions from excitatory neurons
    """
    cell_labels = mouse_day.cell_labels
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    behavior_labels = mouse_day.get_trimmed_beh_labels()

    # Split neural data by cell type
    X_inhibitory = X[:, cell_labels]
    X_excitatory = X[: , ~cell_labels]

    # Balance feature spaces for fair comparison
    min_features = min(X_inhibitory.shape[1], X_excitatory.shape[1])
    logger.info(f"Balancing to {min_features} neurons per cell type")
    
    if X_excitatory.shape[1] > min_features: # Always more excitatory
        np.random.seed(self.random_state)
        selected_features = np.random.choice(X_excitatory.shape[1], min_features, replace=False)
        X_excitatory = X_excitatory[:, selected_features]

    cell_type_data = {
        'inhibitory': {'X': X_inhibitory, 'scores': [], 'predictions': None}, 
        'excitatory': {'X': X_excitatory, 'scores': [], 'predictions': None}
    }

    # Train and evaluate each cell type's population
    for cell_type, data in cell_type_data.items():
        logger.info(f"Decoding {cell_type} interneuron activity...")

        splitter = StratifiedShuffleSplit(
            n_splits=n_trials, test_size=TEST_SIZE, 
            train_size=(1-TEST_SIZE), random_state=self.random_state
        )

        fold_predictions = []

        for fold, (train_idcs, test_idcs) in enumerate(splitter.split(data['X'], behavior_labels)):
            logger.info(f"Processing fold {fold+1}/{n_trials}")
            
            X_train, X_test = data['X'][train_idcs], data['X'][test_idcs]
            y_train, y_test = y[train_idcs], y[test_idcs]

            model = self._create_model(model_type="ridge")
            model.fit(X_train, y_train)

            data['scores'].append(model.score(X_test, y_test))

            y_pred_fold = model.predict(X_test)
            fold_predictions.append((test_idcs, y_pred_fold))
        
        # Reconstruct full predictions
        y_predictions = np.zeros_like(y)
        for test_idcs, preds in y_predictions:
            y_predictions[test_idcs] = preds
        data['predictions'] = y_predictions
        
        if save_results:
            io.save_decoded_data(mouse_day.mouseID, mouse_day.day, data['scores'],
                                  data['predictions'], model_type=cell)
            io.save_model(mouse_day.mouseID, mouse_day.day, model, model_type=cell_type)

        return (cell_type_data['inhibitory']['scores'], cell_type_data['excitatory']['scores'],
                cell_type_data['inhibitory']['predictions'], cell_type_data['excitatory']['predictions'])


    def decode_behavioral_class(self, mouse_day: MouseDay, behavior_class: str, 
                                lag: Optional[int]=None, n_trials: int=10, 
                                save_results: bool=False) -> Tuple[List[float], np.ndarray]:
        """
        Decodes neural activity during behavioral classes (learned vs natural behaviors)

        Parameters
        ----------
        behavior_class: str
            Either "learned" or "natural"
        """
        X = mouse_day.get_trimmed_spks()
        y = mouse_day.get_trimmed_avg_locs() 
        behavior_labels = mouse_day.get_trimmed_beh_labels()

        X, y, behavior_labels = self._apply_calcium_lag(X, y, behavior_labels, lag)

        # Sort behaviors based on class
        if behavior_class == "learned":
            class_mask = (behavior_labels in BEHAVIOR_CLASSES['learned'])
        elif behavior_class == "natural":
            class_mask = (behavior_labels in BEHAVIOR_CLASSES['natural'])
        else:
            raise ValueError("behavior_class must be 'learned' or 'natural'")
    
        X_class = X[class_mask]
        y_class = y[class_mask]
        behavior_labels_class = behavior_labels[class_mask]

        logger.info(f"Decoding {behavior_class} behaviors: {X_class.shape[0]} samples")

        # Cross-validation
        splitter = StratifiedShuffleSplit(
            n_splits=n_trials, test_size=TEST_SIZE,
            train_size=1-TEST_SIZE, random_state=self.random_state
        )

        model = self._create_model(model_type="ridge")
        scores = []
        fold_predictions = []

        for fold, (train_idcs, test_idcs) in enumerate(splitter.split(X_class, behavior_labels_class)):
            logger.info(f"Processing fold {fold+1}/{n_trials}")

            X_train, X_test = X_class[train_idcs], X_class[test_idcs]
            y_train, y_test = y_class[train_idcs], y_class[test_idcs]

            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))

            y_pred_fold = model.predict(X_test)
            fold_predictions.append((test_idcs, y_pred_fold))
        
        # Reconstruct full predictions
        y_predictions = np.zeros_like(y_class)
        for test_idcs, preds in fold_predictions:
            y_predictions[test_idcs] = preds
        
        if save_results:
            save_label = f"{behavior_class}_class" + (f"_lag{lag}" if lag else "")
            io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_predictions, model_type=save_label)
            io.save_model(mouse_day.mouseID, mouse_day.day, model, model_type=save_label)

        return scores, y_predictions


    def get_minimum_behavior_samples(self, mouse_day: MouseDay, 
                                     behavior_list: List[int] = None) -> Tuple[int, List[int]]:
        """
        Find minimum sample count across behaviors for balanced analysis.
        
        Parameters
        ----------
        mouse_day : MouseDay
            Experimental session data
        behavior_list : List[int], optional
            Specific behaviors to analyze. If None, uses all behaviors.
            
        Returns
        -------
        min_samples : int
            Smallest sample count across behaviors
        sample_counts : List[int]
            Sample count for each behavior
        """
        if behavior_list is None:
            behavior_list = BEHAVIOR_CLASSES["all"]
        
        behavior_labels = mouse_day.get_trimmed_beh_labels()
        sample_counts = []
        min_samples = float('inf')
        
        for behavior_id in behavior_list:
            count = np.sum(behavior_labels == behavior_id)
            sample_counts.append(count)
            min_samples = min(min_samples, count)
        
        return int(min_samples), sample_counts


def decode_within_vs_across_classes(self, mouse_day: MouseDay, 
                                    train_behaviors: List[int], 
                                    test_behaviors: List[int], 
                                    analysis_mode: str, 
                                    n_trials: int=10, 
                                    save_results: bool=False) -> Tuple[List[float], np.ndarray]:
    """
    Compare within-class vs cross-class behavioral decoding. 

    Parameters
    ----------
    train_behaviors: List[int]
        Behaviors IDs to train on
    test_behaviors: List[int]
        Behaviors IDs to test on
    analysis_mode: str
        Either "in_class" or "cross_class"
    n_trials: int
        Number of cross-validation trials
    save_results: bool
        Whether to save results

    Returns
    -------
    scores: List[float]
        Cross-validation R^2 scores
    predictions: np.ndarray
        Full prediction array
    """
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs() 
    behavior_labels = mouse_day.get_trimmed_beh_labels()

    behavior_mapping: dict[int, str] = mouse_day.BEHAVIOR_LABELS
    logger.info(f"Training on: {[behavior_mapping[b] for b in train_behaviors]}")
    logger.info(f"{[behavior_mapping[b] for b in test_behaviors]}")

    # Calculate balanced sample sizes
    min_train_samples, train_sample_counts = self.get_minimum_behavior_samples(
        mouse_day, train_behaviors
    )

    # Organize data by behavior
    train_behavior_data = {}
    for behavior_id in train_behaviors:
        mask = behavior_labels == behavior_id
        train_behavior_data[behavior_id] = {
            'X': X[mask],
            'y': y[mask],
            'indices': np.where(mask)[0]
        }

    # Determine test set configuration
    if analysis_mode == "in_class":
        # Test on held-out samples from the same class
        test_behavior = test_behaviors[0] # For now, just testing on single behaviors
        if test_behavior not in train_behaviors:
            raise ValueError("For within_class mode, test behavior must be in train behavior class")
        
        # Test set sizes are balanced between all in_class behaviors
        test_samples_per_behavior = int(0.3 * min_train_samples)
        train_samples_per_behavior = min_train_samples = test_samples_per_behavior

    elif analysis_mode == "cross_class":
        # Test on different behavioral class
        all_behavior_min, _ = self.get_minimum_behavior_samples(mouse_day, BEHAVIOR_CLASSES["all"])
        test_samples_per_behavior = all_behavior_min
        train_samples_per_behavior = min_train_samples

    else:
        raise ValueError("analysis_mode must be 'in_class' or 'cross_class'")

    logger.debug(f"Using {train_samples_per_behavior} training samples per behavior")
    logger.debug(f"Using {test_samples_per_behavior} test samples")

    # Cross-validation
    model = self._create_model(model_type="ridge")
    scores = []
    fold_predictions = []
        
    for fold in range(n_trials):
        logger.info(f"Processing fold {fold+1}/{n_trials}")
        np.random.seed(self.random_state + fold)

        # Build training set with balanced sampling
        X_train_parts = []
        y_train_parts = []

        # Handle test set creation based on analysis mode
        if analysis_mode == "in_class":
            # Hold out test samples from target behavior
            test_behavior = test_behaviors[0]
            test_behavior_data = train_behavior_data[test_behavior]

            # Shuffle and split test behavior data
            n_test_behavior_samples = len(test_behavior_data['X'])
            test_behavior_indices = np.random.permutation(n_test_behavior_samples)

            train_indices = test_behavior_indices[:train_samples_per_behavior]
            test_indices = test_behavior_indices[train_samples_per_behavior:train_samples_per_behavior + test_samples_per_behavior]

            # Create test sets
            X_test = test_behavior_data['X'][test_indices]
            y_test = test_behavior_data['y'][test_indices]
            
            # Create train sets
            # Add test behavior portion, then other train behaviors
            X_train_parts.append(test_behavior_data['X'][train_indices])
            y_train_parts.append(test_behavior_data['y'][train_indices])
            
            for behavior_id in train_behaviors:
                if behavior_id != test_behavior:
                    data = train_behavior_data[behavior_id]
                    indices = np.random.permuation(len(data['X']))
                    X_train_parts.append(data['X'][indices[:train_samples_per_behavior]])
                    y_train_parts.append(data['y'][indices[:train_samples_per_behavior]])
            
        elif analysis_mode == "cross_class":
            # Train on all training behaviors
            for behavior_id in train_behaviors:
                data = train_behavior_data[behavior_id]
                indices = np.random.permutation(len(data['X']))
                
                X_train_parts.append(data['X'][indices[:train_samples_per_behavior]])
                y_train_parts.append(data['y'][indices[:train_samples_per_behavior]])
                
            # Test on outside behavior
            test_mask = np.isin(behavior_labels, test_behaviors)
            test_indices = np.where(test_mask)[0]
            np.random.shuffle(test_indices)

            X_test = X[test_indices]
            y_test = y[test_indices]
            test_indices_global = test_indices



            test_idcs = np.where(np.isin(beh_per_bin, test_class))[0]
            test_idcs = test_idcs[:test_size]
            X_test = spikes[test_idcs]
            y_test = locs[test_idcs]
            # X_test = X_test[:test_size]
            # y_test = y_test[:test_size]
        
        print("test size: ", test_size, ", ", len(X_test)) # these numbers should be the same


        # Generates a training split... 
        # covariate balancing:  randomly pulling the smallest sample size from each behavior's data
        for i, train_beh in enumerate(train_class):
            # shuffle and limit the size of each sample
            train_samples = samples_by_beh[i]
            train_locs = locs_by_beh[i]

            idcs = np.arange(0, len(train_samples))
            np.random.shuffle(idcs)
            train_samples = train_samples[idcs]
            train_samples = train_samples[:num_covars]
            train_locs = train_locs[idcs]
            train_locs = train_locs[:num_covars]

            X_train = np.vstack((X_train, train_samples))
            y_train = np.vstack((y_train, train_locs))

        # give the training sets a lil trim and shuffle
        X_train = X_train[1:]
        y_train = y_train[1:]
        idcs = np.arange(0, len(X_train))
        np.random.shuffle(idcs)
        X_train = X_train[idcs]
        y_train = y_train[idcs]

        # or reset the test_class_samples for the next fold (if the test set is in the train class)
        if mode == "in_class":
            samples_by_beh[test_class_idx] = test_class_samples
            locs_by_beh[test_class_idx] = test_class_locs
        
        # linreg = LinearRegression()
        # linreg.fit(X_train, y_train)
        # score = linreg.score(X_test, y_test)
        # scores.append(score)

        ridge.fit(X_train, y_train)
        score = ridge.score(X_test, y_test)
        scores.append(score)

        y_pred_fold = ridge.predict(X_test)
        y_preds.append((test_idcs, y_pred_fold))
    

    # Reconstruct predictions based on all the test indicies
    y_pred_full = np.zeros_like(locs)
    for test_idcs, y_pred_fold in y_preds:
        y_pred_full[test_idcs] = y_pred_fold
    
    if (save_res):
        train_class_type = [key for key, value in BEH_CLASSES.items() if value==train_class][0]
        test_class_type = [key for key, value in BEH_CLASSES.items() if value==test_class][0]
        print(train_class_type)
        print(test_class_type)
        io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_pred_full, model_type=f"{train_class_type}_x_{test_class_type}")
       
    return scores, y_pred_full


def decode_crossday_general(train_day: MouseDay, test_day: MouseDay, cross_test: bool=False, ntrials: int=10, save_res=False):
    """
    Decoding paw positions from the general population of REGISTERED neurons.
    Model is trained on the train_day's registered neurons. 
    If cross-test is true, we test on the test_day. Otherwise test on train_day's holdout. 
    """
    
    X = train_day.get_trimmed_spks(reg_key=test_day.day)
    y = train_day.get_trimmed_avg_locs()
    beh_labels = train_day.get_trimmed_beh_labels()

    if (cross_test):
        X_cross_day = test_day.get_trimmed_spks(reg_key=train_day.day)
        y_cross_day = test_day.get_trimmed_avg_locs()

    scores = []

    splitter = StratifiedShuffleSplit(n_splits=ntrials, test_size=TEST_SIZE, train_size=1-TEST_SIZE, random_state=42)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000], fit_intercept=True)

    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, beh_labels)):
        print("Fold: ", i)
        X_train = X[train_idcs]
        y_train = y[train_idcs]

        if (cross_test):
            X_test = X_cross_day[test_idcs]
            y_test = y_cross_day[test_idcs]
        else:
            X_test = X[test_idcs]
            y_test = y[test_idcs]

        ridge.fit(X_train, y_train)

        score = ridge.score(X_test, y_test)
        scores.append(score)

    if (cross_test):
        y_preds = ridge.predict(X_cross_day)
    else:
        y_preds = ridge.predict(X)

    if (save_res):
        # SAVES WITHIN THE TRAIN DAY'S FOLDER
        if (cross_test):
            save_label = f"{train_day.day}_x_{test_day.day}"
        else:
            save_label = f"registered_general"
        io.save_decoded_data(train_day.mouseID, train_day.day, scores, y_preds, save_label)
        io.save_model(train_day.mouseID, train_day.day, ridge, save_label)

    return scores, y_preds

      

def latency_check(mouse_day: MouseDay):
    print("# of timestamps (calcium): ", mouse_day.cal_ntimestamps)
    print("# of datapoints (calcium): ", mouse_day.cal_nframes)
    mouse_day.check_caltime_latency()
    return 0

def dimensions_check(mouse_day: MouseDay):
    # Go back and figure out how the lengths differ...and why this func takes a hot sec
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
    return 0

def md_run(mouse_day: MouseDay, save_status=False):
    """
    Just to make sure all the mice are mice-ing. 
    Runs EVERYTHING.
    Saves if we specify. 
    """
    # latency_check(mouse_day)
    # dimensions_check(mouse_day)

    # fig = myplot.plot_interp_test(mouse_day, mouse_day.seg_keys[0])
    # plt.show()

    decode_general(mouse_day, save_res=save_status)
    # fig1 = myplot.plot_kin_predictions(mouse_day)

    decode_behaviors(mouse_day, save_res=save_status)
    # fig2 = myplot.plot_model_performance_swarm(mouse_day)

    decode_behaviors_with_general(mouse_day, save_res=save_status)
    # fig3 = myplot.plot_general_performance_by_beh(mouse_day)

    decode_by_cell(mouse_day, save_res=save_status)
    # fig4 = myplot.plot_cell_performance_swarm(mouse_day)

    for beh in LEARNED:
        scores, preds = decode_behaviors_with_class(mouse_day, train_class=BEH_CLASSES["learned"], test_class=BEH_CLASSES[beh], mode="in_class", save_res=save_status)
        scores1, preds1 = decode_behaviors_with_class(mouse_day, train_class=BEH_CLASSES["natural"], test_class=BEH_CLASSES[beh], mode="cross_class", save_res=save_status)

    for beh in NATURAL:
        scores, preds = decode_behaviors_with_class(mouse_day, train_class=BEH_CLASSES["natural"], test_class=BEH_CLASSES[beh], mode="in_class", save_res=save_status)
        scores1, preds1 = decode_behaviors_with_class(mouse_day, train_class=BEH_CLASSES["learned"], test_class=BEH_CLASSES[beh], mode="cross_class", save_res=save_status)

    # fig5 = myplot.plot_performance_swarm(mouse_day, modes=myplot.IN_CLASS_MODE, mode_type="In-Class")
    # fig6 = myplot.plot_performance_swarm(mouse_day, modes=myplot.CROSS_CLASS_MODE, mode_type="Cross-Class")

    # plt.show()
    return 0

def decode_across_days(mouse_days: list[MouseDay]):
    for curr_day in mouse_days:
        for cross_day in mouse_days:

            if curr_day != cross_day:
                    s, p = decode_crossday_general(train_day=curr_day, test_day=cross_day, cross_test=True, save_res=True)
                    print(f"{curr_day.day} x {cross_day.day} scores: ", s)
            else:
                # just train a general model on the day's registered neurons
                s, p = decode_crossday_general(train_day=curr_day, test_day=cross_day, cross_test=False, save_res=True)
                print(f"{curr_day.day}'s registered neuron scores: ", s)


def decode_gen_with_lag(mouse_day: MouseDay):
    for i in range(1, 9):
        s, p = decode_general(mouse_day, lag=i, save_res=True)
        print(s)

def decode_beh_with_lag(mouse_day: MouseDay):
    for i in range(1, 9):
        s, p  = decode_behaviors(mouse_day, lag=i, save_res=True)
        print(s)

def decode_class_with_lag(mouse_day: MouseDay, class_type: str):
    for i in range(1, 9):
        s, p = decode_by_class(mouse_day, beh_class=class_type, lag=i, save_res=True)
        print(s)

if __name__ == "__main__":

    mouseIDs = ['mouse25']
    days = ['20240420', '20240421', '20240422', '20240423', '20240424', '20240425', '20240428', '20240429', '20240430', '20240501' ,'20240502', '20240503']
    test_day = MouseDay("mouse25", "20240425")
    decode_beh_with_lag(test_day)
    decode_class_with_lag(test_day, "learned")
    decode_class_with_lag(test_day, "natural")