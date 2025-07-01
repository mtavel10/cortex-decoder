import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut
import math
from mouse import MouseDay
from typing import Optional, Tuple
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, brier_score_loss
import plot as myplot

def general_ridge(mouse_day: MouseDay, n_trials: int=10):
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()
    scores = []
    weights = []
    # Splitter object
    splitter = StratifiedKFold(n_splits=n_trials, shuffle=True, random_state=42)

    # splitter.split() randomly generates test/training indices for X based on a stratifier (the beh labels)
    # Loops n_splits times
    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, beh_per_frame)):
        print("Training Split: ", i)
        X_train, X_test = X[train_idcs], X[test_idcs]
        y_train, y_test = y[train_idcs], y[test_idcs]
        
        # Cross-validate to find the best alpha
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=False)

        # Train the model
        ridge.fit(X_train, y_train)

        # Evaluate based on R^2
        scores.append(ridge.score(X_test, y_test))
        weights.append(ridge.coef_)
    
    # Find the average weights based on each trial's score
    avg_w = np.zeros(weights[0].shape)
    score_sum = np.sum(scores)
    for i, w in enumerate(weights):
        avg_w += (w * scores[i])
    avg_w *= (1/score_sum)
    
    # Use average weights to make predictions on the kinematics data
    y_pred = X @ avg_w.T
    
    return avg_w, scores, y, y_pred

def ridge_by_beh(mouse_day: MouseDay, n_trials: int=10):
    """
    Separates the data by behavior label, trains a seperate model per behavior and returns a list of those weights (indexed by behavior label)
    """
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()
    labels = mouse_day.BEHAVIOR_LABELS
    all_weights = []
    all_scores = []

    for label in labels.keys():
        beh_frames = np.where(beh_per_frame == label)
        curr_X = X[beh_frames]
        curr_y = y[beh_frames]
        scores = []
        weights = []
        
        # KFold CV on the current behavior's data
        splitter = KFold(n_splits=n_trials, shuffle=True, random_state=42)
        for i, (train_idcs, test_idcs) in enumerate(splitter.split(curr_X)):
            X_train, X_test = curr_X[train_idcs], curr_X[test_idcs]
            y_train, y_test = curr_y[train_idcs], curr_y[test_idcs]
            
            # CV to find best alpha
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])

            # Train the model
            ridge.fit(X_train, y_train)

            # Evaluate based on R^2
            scores.append(ridge.score(X_test, y_test))
            weights.append(ridge.coef_)
        
        # Find the average weights based on each trial's score
        avg_w = np.zeros(weights[0].shape)
        score_sum = np.sum(scores)
        for i, w in enumerate(weights):
            avg_w += (w * scores[i])
        avg_w *= (1/score_sum)
        print(f"Average weights for {label}th behavior: ", avg_w)
        print(f"Scores for {label}th behavior: ", scores)
        
        all_weights.append(avg_w)
        all_scores.append(scores)
    
    return all_weights, all_scores
      
def latency_check(mouse_day: MouseDay):
    print("# of timestamps (calcium): ", test_mouse.cal_ntimestamps)
    print("# of datapoints (calcium): ", test_mouse.cal_nframes)
    mouse_day.check_caltime_latency()

def dimensions_check(mouse_day: MouseDay):
    test_locs = mouse_day.get_trimmed_avg_locs()
    test_untrimmedlocs = mouse_day.get_all_avg_locations()
    test_spikes = mouse_day.get_trimmed_spks()
    test_labels = mouse_day.get_trimmed_beh_labels()
    test_untrimmed_labels = mouse_day.get_beh_labels()

    print("No Trim Locs: ", test_untrimmedlocs.shape)
    print("No Trim Spikes: ", mouse_day.cal_spks.T.shape)
    print("No Trim Labels: ", len(test_untrimmed_labels))

    print("Trimmed Locs: ", test_locs.shape)
    print("Trimmed Spikes: ", test_spikes.shape)

    print("Num labels: ", len(test_labels))

if __name__ == "__main__":
    test_mouse = MouseDay("mouse25", "20240425")
    # print("number of caltstamps: ", test_mouse.cal_ntimestamps)
    # print("number of cal frames: ", test_mouse.cal_nframes)
    # print("caltstamps (full): ", test_mouse.cal_tstamps)
    # print("length of that array: ", len(test_mouse.cal_tstamps))

    # for label in test_mouse.get_beh_labels():
    #     print(label)
    
    # latency_check(test_mouse)
    # dimensions_check(test_mouse)

    # gen_weights, gen_scores = general_ridge(test_mouse)
    # print(avg_weights)
    # print(scores)
    # beh_weights, beh_scores = weights_by_beh = ridge_by_beh(test_mouse)
    # myplot.plot_r2_scores(test_mouse.BEHAVIOR_LABELS, beh_scores, gen_scores)
