import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut
import math
from mouse import MouseDay
from typing import Optional, Tuple
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.metrics import r2_score
import plot as myplot
import matplotlib.pyplot as plt

TEST_SIZE = .30 # 70/30 split duhhh

def general_ridge(mouse_day: MouseDay, n_trials: int=10):
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()
    scores = []
    y_preds = []
    # Splitter object
    splitter = StratifiedShuffleSplit(n_splits=n_trials, test_size=0.3, train_size=0.7, random_state=42)

    # Cross-validate to find the best alpha
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=True)
    
    # splitter.split() randomly generates test/training indices for X based on a stratifier (the beh labels)
    # Loops n_splits times
    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, beh_per_frame)):
        print("Training Split: ", i)
        X_train, X_test = X[train_idcs], X[test_idcs]
        y_train, y_test = y[train_idcs], y[test_idcs]

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
    
    # saves the scores and predictions for plotting
    io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_pred, model_type="general")
    # saves the last training iteration just in case
    io.save_model(mouse_day.mouseID, mouse_day.day, ridge)

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
        print(f"Training models on {labels[label]} data... ")
        beh_frames = np.where(beh_per_frame == label)
        curr_X = X[beh_frames]
        curr_y = y[beh_frames]
        scores = []
        y_preds = []

        # CV to find best alpha
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=True)
        
        # KFold CV on the current behavior's data
        splitter = KFold(n_splits=n_trials, shuffle=True, random_state=42)
        for i, (train_idcs, test_idcs) in enumerate(splitter.split(curr_X)):
            print("training split: ", i)
            X_train, X_test = curr_X[train_idcs], curr_X[test_idcs]
            y_train, y_test = curr_y[train_idcs], curr_y[test_idcs]

            # Train the model
            ridge.fit(X_train, y_train)

            # Evaluate based on R^2
            trial_score = ridge.score(X_test, y_test)
            scores.append(trial_score)

        #     # Predict locations for this test fold
        #     # Only predict on test data for this fold
        #     y_pred_fold = ridge.predict(X_test)
        #     y_preds.append((test_idcs, y_pred_fold))
        
        # # Reconstruct full predictions (but only for timepoints of this behavior??)
        # y_pred = np.zeros_like(y)
        # for test_idcs, pred in y_preds:
        #     y_pred[test_idcs] = pred
        # print(y_pred)

        # Make an overall prediction on all data using the last iteration of this behavior model
        # Not that precise of a location prediction but pred data more for sanity check than anything
        y_pred = ridge.predict(X)

        all_preds.append(y_pred)
        all_scores.append(scores)
        print("scores: ", scores)

        # save model
        io.save_model(mouse_day.mouseID, mouse_day.day, ridge, labels[label])
        io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_pred, model_type=labels[label])

    return all_scores, y_preds


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

    print(X.shape)
    print(y.shape)
    X_gen = np.zeros((1, X.shape[1]))
    y_gen = np.zeros((1, y.shape[1]))
    beh_per_frame_gen = []

    # need to limit the size of testing samples later on
    test_sizes = []

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

        test_sizes.append(len(holdout_X))

        X_beh_test[label] = holdout_X
        y_beh_test[label] = holdout_y
        
        beh_per_frame_gen += [label] * len(using_y)
    
        X_gen = np.vstack((X_gen, using_X))
        y_gen = np.vstack((y_gen, using_y))

    X_gen = X_gen[1:]
    y_gen = y_gen[1:]
    
    min_test_size = 157
    print(test_sizes)
    print(min_test_size)

    # 2: train a model
    scores_by_beh: dict[int, list[float]] = {}
    for label in behaviors.keys():
        scores_by_beh[label] = []

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

            # make sure all the test data samples are the same across behaviors
            indices = np.arange(len(X_test))
            np.random.shuffle(indices)
            X_test, y_test = X_test[indices], y_test[indices]
            X_test, y_test = X_test[:min_test_size], y_test[:min_test_size]

            trial_scores = ridge.score(X_test, y_test)
            print(f"General Model Score on {behaviors[label]} data: ", trial_scores)
            scores_by_beh[label].append(trial_scores)# 10 scores per behavior (10 folds)
            
    io.save_scores_by_beh(mouse_day.mouseID, mouse_day.day, scores_by_beh)

    return scores_by_beh

def ridge_by_cell(mouse_day: MouseDay, ntrials: int=10):
    """
    Running 2 ridge regressions on different feature spaces: excitatory and inhibitory interneurons. 
    Motivation: Is it easier to predict on excitatory or inhibitory neural data?
    """
    cell_labels = mouse_day.cell_labels
    data = mouse_day.get_trimmed_spks()

    X_inhibitory = data[:, cell_labels]
    X_excitatory = data[: , ~cell_labels]

    # Need to train on equal feature spaces...
    min_features = len(X_inhibitory[0])

    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()

    ex_scores = []
    ex_preds = []

    in_scores = []
    in_preds = []
    
    split_data: dict[str, list[any]] = {"inhibitory": [X_inhibitory, in_scores, in_preds], "excitatory": [X_excitatory, ex_scores, ex_preds]}

    for cell, cell_data in split_data.items():
        print(f"Decoding {cell} data...")

        splitter = StratifiedShuffleSplit(n_splits=ntrials, test_size=TEST_SIZE, train_size=(1-TEST_SIZE), random_state=42)
        # Cross-validate to find the best alpha
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=True)

        X = cell_data[0]

        # Need to limit the feature space for excitatory data
        if cell == "excitatory":
            np.random.seed(42)
            idcs = np.random.choice(X.shape[1], min_features, replace=False)
            X = X[:, idcs]
        
        y_preds = []
        for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, beh_per_frame)):
            print("Training Split: ", i)
            X_train, X_test = X[train_idcs], X[test_idcs]
            y_train, y_test = y[train_idcs], y[test_idcs]

            # Train the model
            ridge.fit(X_train, y_train)

            # Evaluate based on R^2
            cell_data[1].append(ridge.score(X_test, y_test))

            # Only predict on test data for this fold
            y_pred_fold = ridge.predict(X_test)
            y_preds.append((test_idcs, y_pred_fold))
        
        # Reconstruct full predictions
        y_pred = np.zeros_like(y)
        for test_idcs, pred in y_preds:
            y_pred[test_idcs] = pred
        cell_data[2] = y_pred

        print("Scores: ", cell_data[1])
        
        # saves the scores and predictions for plotting
        io.save_decoded_data(mouse_day.mouseID, mouse_day.day, cell_data[1], cell_data[2], model_type=cell)
        # saves the last training iteration just in case
        io.save_model(mouse_day.mouseID, mouse_day.day, ridge, model_type=cell)

    return in_scores, in_preds, ex_scores, ex_preds

def ridge_by_class(mouse_day: MouseDay, beh_class: str, ntrials: int=10):
    spikes = mouse_day.get_trimmed_spks()
    locs = mouse_day.get_trimmed_avg_locs() 
    beh_per_frame = mouse_day.get_trimmed_beh_labels()
    
    # separate out data by behavior class
    if beh_class == "learned":
        # decode based on reach, carry, and grasp data
        class_beh_frames = np.where((beh_per_frame >= 0) & (beh_per_frame <= 2))
    else:
        # decode based on "natural" behaviors: non-movement, fidget, eating, and grooming
        class_beh_frames = np.where((beh_per_frame >= 3) & (beh_per_frame <= 6))
    X = spikes[class_beh_frames]
    y = locs[class_beh_frames]
    beh_per_frame = beh_per_frame[class_beh_frames]

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=True)
    splitter = StratifiedShuffleSplit(n_splits=ntrials, test_size=TEST_SIZE, train_size = 1-TEST_SIZE)

    scores = []
    y_preds = []
    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, beh_per_frame)):
        print("Training split: ", i)
        X_train, X_test = X[train_idcs], X[test_idcs]
        y_train, y_test = y[train_idcs], y[test_idcs]

        ridge.fit(X_train, y_train)
        scores.append(ridge.score(X_test, y_test))

        y_preds_fold = ridge.predict(X_test)
        y_preds.append((test_idcs, y_preds_fold))
    
     # Reconstruct full predictions
    y_pred = np.zeros_like(y)
    for test_idcs, pred in y_preds:
        y_pred[test_idcs] = pred
    
    io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_pred, model_type=f"{beh_class}_class")
    io.save_model(mouse_day.mouseID, mouse_day.day, ridge, model_type=f"{beh_class}_class")

    return scores, y_pred


def decode_cross_beh_class(mouse_day: MouseDay, train_class: list[int], test_class: list[int], mode: str, save_res=True, ntrials: int=10):
    """
    Train and test classes contain behavior labels belonging to certain "classes". Can select which behaviors to train/test on dynamically. 
        mode
            either "cross_class" or "in_class"
    """
    behavior_labels: dict[int, str] = mouse_day.BEHAVIOR_LABELS
    print("Function with naive splitting: ")
    print("training data: ", ", ".join(behavior_labels[beh] for beh in train_class))
    print("testing data: ", ", ".join(behavior_labels[beh] for beh in test_class))

    spikes: np.ndarray = mouse_day.get_trimmed_spks()
    locs: np.ndarray = mouse_day.get_trimmed_avg_locs() 
    beh_per_frame:np.ndarray = mouse_day.get_trimmed_beh_labels()
    
    # separate out data by behavior class
    # Problem: For decoding behaviors with a model trained on their own behavior class, wouldn't there be a bias towards behaviors with more samples?
    # Should each training dataset be segmented into representative sample-sizes?
    # # yes: do "covariate balancing"

    training_beh_frames = np.where(np.isin(beh_per_frame, train_class))[0]
    X = spikes[training_beh_frames]
    y = locs[training_beh_frames]

    testing_beh_frames = np.where(np.isin(beh_per_frame, test_class))[0]
    X_test = spikes[testing_beh_frames]
    y_test = locs[testing_beh_frames]


    min_test_size = 100000000000000000000
    # limiting it to the smallest sample size so that no group is biased
    if (mode == "cross_class"):
        classes = BEH_CLASSES["learned"] + BEH_CLASSES["natural"]
    elif (mode == "in_class"):
        # find the minimum size within this class
        classes = train_class
    for beh in classes:
        num_beh_frames = np.sum(beh == beh_per_frame)
        if num_beh_frames < min_test_size:
            min_test_size = num_beh_frames
        
    print("min test size: ", min_test_size)

    # to sort training data by behavior group to ensure equal distribution
    training_bpf = beh_per_frame[training_beh_frames]
    
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], fit_intercept=True)
    splitter = StratifiedShuffleSplit(n_splits=ntrials, test_size=TEST_SIZE, train_size = 1-TEST_SIZE)

    np.random.seed(42)

    scores = []
    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, training_bpf)):
        print("Training split: ", i)
        X_train = X[train_idcs]
        y_train = y[train_idcs]

        ridge.fit(X_train, y_train)

        # 10 shuffles/tests on the tested behavior/behavior class. sample size limited to the smallest behavior class sample size
        test_indices = np.arange(len(X_test))
        np.random.shuffle(test_indices)

        X_test_fold, y_test_fold = X_test[test_indices], y_test[test_indices]
        X_test_fold, y_test_fold = X_test[:min_test_size], y_test[:min_test_size]
        
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot(y_test_fold, label="y_test_fold")
        # ax.plot(y_train, label="y_train")
        scores.append(ridge.score(X_test_fold, y_test_fold))
        
    # make predictions on cross-testing class - not robust, since using the most recent ridge object, just for sanity check
    y_pred = ridge.predict(X_test)

    # put these predictions into a matrix of the same size as the original y, zero out where behavior isn't here
    y_pred_full = np.zeros_like(locs)
    for frame in range(0, len(y_pred_full)):
        if frame in testing_beh_frames:
            y_pred_full[frame] = y_pred[np.where(testing_beh_frames == frame)]
        else:
            y_pred_full[frame] = 0

    if (save_res):
        train_class_type = [key for key, value in BEH_CLASSES.items() if value==train_class][0]
        test_class_type = [key for key, value in BEH_CLASSES.items() if value==test_class][0]
        io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_pred_full, model_type=f"{train_class_type}_x_{test_class_type}")

    return scores, y_pred_full


def simple_decode_by_class(mouse_day: MouseDay, train_class: list[int], test_class: list[int], mode: str, save_res=True, ntrials: int=10):
    """
    Wtf is going on with these R^2 scores
    """
    
    behavior_labels: dict[int, str] = mouse_day.BEHAVIOR_LABELS
    print("training data: ", ", ".join(behavior_labels[beh] for beh in train_class))
    print("testing data: ", ", ".join(behavior_labels[beh] for beh in test_class))

    spikes: np.ndarray = mouse_day.get_trimmed_spks()
    locs: np.ndarray = mouse_day.get_trimmed_avg_locs() 
    beh_per_bin: np.ndarray = mouse_day.get_trimmed_beh_labels()

    # balance the covariates in the training class
    min_samples = 1000000000000000000000
    train_sample_sizes = []

    samples_by_beh = []
    locs_by_beh = []
    bins_by_beh = [] # holds the overall timebins where the behaviors occur (for reconstructing predictions later)
    for i, train_beh in enumerate(train_class):
        num_beh_spikes = np.sum(beh_per_bin == train_beh)
        # print("behavior: ", train_beh, ", nsamples: ", num_beh_spikes)
        # finds the smallest sample size amongst all training behaviors
        if min_samples >= num_beh_spikes:
            min_samples = num_beh_spikes
        train_sample_sizes.append(num_beh_spikes)
        
        # pulls out all samples for all training behaviors
        beh_idcs = np.where(beh_per_bin == train_beh)
        samples_by_beh.append(spikes[beh_idcs])
        locs_by_beh.append(locs[beh_idcs])
        bins_by_beh.append(beh_idcs)

    scores = []
    y_preds = []

    # calculate the holdout size and update the min_sample size if needed
    if mode == "in_class":
        test_class_idx = np.where(np.isin(train_class, test_class))[0][0]

        test_class_samples = samples_by_beh[test_class_idx]
        test_class_locs = locs_by_beh[test_class_idx]

        train_size = int((1-TEST_SIZE) * len(test_class_samples))

        # update the minimum sample size if needed
        if (min_samples > train_size):
            min_samples = train_size
    elif mode == "cross_class":
        min_samples = 40 # change this to be dynamic across mice (the min behavior label size across all behaviors)

    print("min test size: ", min_samples)
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
    for trial in range(ntrials):
        print("training split: ", trial)
        np.random.seed(42 + trial)

        X_train = np.zeros((1, spikes.shape[1]))
        y_train = np.zeros((1, locs.shape[1]))

        # going to be set based on the mode
        X_test = ""
        y_test = ""
        test_idcs = ""
        
        # Holdout a test set from the training samples if needed
        if mode == "in_class":
            # to ensure random pull per fold
            idcs = np.arange(0, len(test_class_samples)) 
            
            np.random.shuffle(idcs)
            test_class_samples = test_class_samples[idcs]
            test_class_locs = test_class_locs[idcs]
            # pull out 70% of this sample set to put into the model
            using_spikes, using_locs = test_class_samples[train_size:], test_class_locs[train_size:]
            # hold 30% of this sample set to use for testing
            holdout_spikes, holdout_locs = test_class_samples[:train_size], test_class_locs[:train_size]

            samples_by_beh[test_class_idx] = using_spikes
            locs_by_beh[test_class_idx] = using_locs

            # assign the test values
            X_test = holdout_spikes
            y_test = holdout_locs

            # save the indicies for predictions later
            # saving these wrong: based off of the sample indicies not the overall indicies
            test_idcs_in_sample = idcs[train_size:]
            test_idcs = bins_by_beh[test_class_idx][0][test_idcs_in_sample]


        # covariate balancing: creates a new training dataset by randomly pulling the smallest sample size from each behavior's data
        for i, train_beh in enumerate(train_class):
            # shuffle and limit the size of each sample
            train_samples = samples_by_beh[i]
            train_locs = locs_by_beh[i]

            idcs = np.arange(0, len(train_samples))
            np.random.shuffle(idcs)
            train_samples = train_samples[idcs]
            train_samples = train_samples[:min_samples]
            train_locs = train_locs[idcs]
            train_locs = train_locs[:min_samples]

            X_train = np.vstack((X_train, train_samples))
            y_train = np.vstack((y_train, train_locs))

        # give the training sets a lil trim and shuffle
        X_train = X_train[1:]
        y_train = y_train[1:]
        idcs = np.arange(0, len(X_train))
        np.random.shuffle(idcs)
        X_train = X_train[idcs]
        y_train = y_train[idcs]

        if mode == "cross_class":
            test_idcs = np.where(np.isin(beh_per_bin, test_class))[0]
            X_test = spikes[test_idcs]
            y_test = locs[test_idcs]
        elif mode == "in_class":
            # reset the test_class_samples for the next fold
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
        print(y_pred_fold.shape)
        print(test_idcs.shape)
        # y_pred_full[test_idcs] = y_pred_fold
       
    return scores, y_pred_full
      

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
    mouseID = "mouse25"
    day = "20240425"
    test_mouse = MouseDay(mouseID, day)

    # hold out grooming data
    BEH_CLASSES = {"learned": [0, 1, 2], "natural": [3, 4, 5], "reach": [0], "grasp": [1], "carry": [2], "non_movement": [3], "fidget": [4], "eating": [5], "grooming": [6]}
    LEARNED = ["reach", "grasp", "carry"]
    NATURAL = ["non_movement", "fidget", "eating"]

    scores, preds = simple_decode_by_class(test_mouse, train_class=BEH_CLASSES["learned"], test_class=BEH_CLASSES["reach"], mode="in_class")
    print("score: ", np.mean(scores))

    # # Wtf is going on with the R2 scores
    # for beh in LEARNED:
    #     scores, preds = simple_decode_by_class(test_mouse, train_class=BEH_CLASSES["learned"], test_class=BEH_CLASSES[beh], mode="in_class")
    #     print("score: ", np.mean(scores))
    #     print()

    #     class_scores, class_preds = decode_cross_beh_class(test_mouse, train_class=BEH_CLASSES["learned"], test_class=BEH_CLASSES[beh], mode="in_class")
    #     print("score: ", np.mean(class_scores))
    #     print()
    #     print("--------------------------")

    # for beh in NATURAL:
    #     scores, preds = simple_decode_by_class(test_mouse, train_class=BEH_CLASSES["natural"], test_class=BEH_CLASSES[beh], mode="in_class")
    #     print("score: ", np.mean(scores))
    #     print()

    #     class_scores, class_preds = decode_cross_beh_class(test_mouse, train_class=BEH_CLASSES["natural"], test_class=BEH_CLASSES[beh], mode="in_class")
    #     print("score: ", np.mean(class_scores))
    #     print()
    #     print("--------------------------")

    # all_beh_scores, all_beh_preds = ridge_by_beh(test_mouse)
    # print("all score: ", all_beh_scores)

    # # testing on natural classes
    # for beh in ["eating"]:
    #     print()
    #     print(f"(IN-CLASS) Natural model on {beh} data: ")
    #     scores, preds = decode_cross_beh_class(test_mouse, train_class=BEH_CLASSES["natural"], test_class = BEH_CLASSES[beh], mode="in_class")
    #     print("scores: ", scores)

    #     print()
    #     print(f"(CROSS-CLASS) Learned model on {beh} data: ")
    #     scores, preds = decode_cross_beh_class(test_mouse, train_class=BEH_CLASSES["learned"], test_class = BEH_CLASSES[beh], mode="cross_class")
    #     print("scores: ", scores)

    # # testing on learned classes
    # for beh in LEARNED:
    #     print(f"(IN-CLASS) Learned model on {beh} data: ")
    #     scores, preds = decode_cross_beh_class(test_mouse, train_class=BEH_CLASSES["learned"], test_class = BEH_CLASSES[beh], mode="in_class")
    #     print("scores: ", scores)

    #     print(f"(CROSS-CLASS) Natural model on {beh} data: ")
    #     scores, preds = decode_cross_beh_class(test_mouse, train_class=BEH_CLASSES["natural"], test_class = BEH_CLASSES[beh], mode="cross_class")
    #     print("scores: ", scores)

    # # s, p = decode_cross_beh_class(test_mouse, train_class=BEH_CLASSES["natural"], test_class=BEH_CLASSES["learned"], mode="cross_class")
    # # s1, p1 = decode_cross_beh_class(test_mouse, train_class=BEH_CLASSES["learned"], test_class=BEH_CLASSES["natural"], mode="cross_class")