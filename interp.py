import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut
import math
from mouse import MouseDay
from typing import Optional, Tuple
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
import plot as myplot

def general_ridge(mouse_day: MouseDay, n_trials: int=10):
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()
    scores = []
    weights = []
    y_preds = []
    # Splitter object
    splitter = StratifiedKFold(n_splits=n_trials, shuffle=True, random_state=42)

    # splitter.split() randomly generates test/training indices for X based on a stratifier (the beh labels)
    # Loops n_splits times
    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, beh_per_frame)):
        print("Training Split: ", i)
        X_train, X_test = X[train_idcs], X[test_idcs]
        y_train, y_test = y[train_idcs], y[test_idcs]
        
        # Cross-validate to find the best alpha
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=True)

        # Train the model
        ridge.fit(X_train, y_train)

        # Evaluate based on R^2
        scores.append(ridge.score(X_test, y_test))

        # Only predict on test data for this fold
        y_pred_fold = ridge.predict(X_test)
        y_preds.append((test_idcs, y_pred_fold))
    
    # Reconstruct full predictions
    y_pred = np.zeros_like(y)
    for test_idcs, pred in y_preds:
        y_pred[test_idcs] = pred
    
    # saves the preds and scores for plotting purposes
    # io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, model_type="general")
    # save the ridge objects (just in case)

    return scores, y, y_pred


def ridge_by_beh(mouse_day: MouseDay, n_trials: int=10):
    """
    Separates the data by behavior label, trains a seperate model per behavior and returns a list of those weights (indexed by behavior label)
    """
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()
    labels = mouse_day.BEHAVIOR_LABELS
    all_preds = []
    all_scores = []

    for label in labels.keys():
        beh_frames = np.where(beh_per_frame == label)
        curr_X = X[beh_frames]
        curr_y = y[beh_frames]
        scores = []
        y_preds = []
        
        # KFold CV on the current behavior's data
        splitter = KFold(n_splits=n_trials, shuffle=True, random_state=42)
        for i, (train_idcs, test_idcs) in enumerate(splitter.split(curr_X)):
            X_train, X_test = curr_X[train_idcs], curr_X[test_idcs]
            y_train, y_test = curr_y[train_idcs], curr_y[test_idcs]
            
            # CV to find best alpha
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=False)

            # Train the model
            ridge.fit(X_train, y_train)

            # Evaluate based on R^2
            scores.append(ridge.score(X_test, y_test))

            # Predict locations for this test fold
            # Only predict on test data for this fold
            y_pred_fold = ridge.predict(X_test)
            y_preds.append((test_idcs, y_pred_fold))
        
        # Reconstruct full predictions FOR THIS BEHAVIOR MODEL
        y_pred = np.zeros_like(y)
        for test_idcs, pred in y_preds:
            y_pred[test_idcs] = pred

        all_preds.append(y_pred)
        all_scores.append(scores)

    # save scores
    # for label, scores in zip(labels.values(), all_scores):
    #     io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, model_type=label)

    return all_scores, y, y_preds


def ridge_test_by_beh(mouse_day: MouseDay, ntrials: int=10):
    """
    Training a general model, holding out certain samples that are representative of behavior... then testing that general model by samples grouped by behavior. 
    """
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()

    # Holding testing data by behavior
    X_beh_test: dict[int, np.ndarray] = {}
    y_beh_test: dict[int, np.ndarray] = {}

    X_gen = np.zeros((1, 420))
    y_gen = np.zeros((1, 4))
    beh_per_frame_gen = []

    # 1: need to hold out samples of each behavior
    behaviors = mouse_day.BEHAVIOR_LABELS
    for label in behaviors.keys():
        beh_frames = np.where(beh_per_frame == label)
        beh_X = X[beh_frames]
        beh_y = y[beh_frames]
        
        indices = np.arange(len(beh_X))
        np.random.shuffle(indices)
        beh_X = beh_X[indices]
        beh_y = beh_y[indices]

        # 70/30 split: 70% of these samples will go towards training the general model, the 30% will be tested on
        # vibes based we can see how performance changes if we adjust this split
        holdout_idx = int(.3 * len(beh_X))
        holdout_X, holdout_y = beh_X[:holdout_idx], beh_y[:holdout_idx]
        using_X, using_y = beh_X[holdout_idx:], beh_y[holdout_idx:]

        X_beh_test[label] = holdout_X
        y_beh_test[label] = holdout_y
        
        beh_per_frame_gen += [label] * len(using_y)
    
        X_gen = np.vstack((X_gen, using_X))
        y_gen = np.vstack((y_gen, using_y))

    X_gen = X_gen[1:]
    y_gen = y_gen[1:]

    # 2: train a model
    scores_by_beh: dict[int, list[float]] = {}
    preds_by_beh: dict[int, list[float]] = {}
    for label in behaviors.keys():
        scores_by_beh[label] = []
        preds_by_beh[label] = []

    splitter = StratifiedKFold(n_splits=ntrials, shuffle=True, random_state=42)
    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X_gen, beh_per_frame_gen)):
        print("Training Split: ", i)
        X_train = X_gen[train_idcs]
        y_train = y_gen[train_idcs]
        
        # Cross-validate to find the best alpha
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=True)

        # Train the model
        ridge.fit(X_train, y_train)

        # Evaluate fold model performance and predict on each behavior data
        for label in X_beh_test.keys():
            X_test = X_beh_test[label]
            y_test = y_beh_test[label]

            scores_by_beh[label].append(ridge.score(X_test, y_test)) # 10 scores per behavior (10 folds)
            preds_by_beh[label].append(ridge.predict(X_test)) # every fold is going to have different predictions for the same testing data...
            
    
    return scores_by_beh, y_beh_test, preds_by_beh

      
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

    # gen_scores, y, y_pred = general_ridge(test_mouse)
    # print(gen_scores)
    # print(y_pred)

    # print(avg_weights)
    # print(scores)
    # all_weights, all_scores, y, y_preds = ridge_by_beh(test_mouse)
    # myplot.plot_r2_scores(test_mouse.BEHAVIOR_LABELS, beh_scores, gen_scores)
    
    scores_by_beh, y_beh_test, preds_by_beh = ridge_test_by_beh(test_mouse)
    print(scores_by_beh)
