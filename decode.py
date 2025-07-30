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
BEH_CLASSES = {"all": [0, 1, 2, 3, 4, 5], "learned": [0, 1, 2], "natural": [3, 4, 5], "reach": [0], "grasp": [1], "carry": [2], "non_movement": [3], "fidget": [4], "eating": [5]}
LEARNED = ["reach", "grasp", "carry"]
NATURAL = ["non_movement", "fidget", "eating"]


def decode_general(mouse_day: MouseDay, n_trials: int=10, save_res=False):
    """
    Decodes all the samples across the entire population of neurons. 
        Train: General, Test: General
    """
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()
    scores = []
    y_preds = []

    # Splitter object
    splitter = StratifiedKFold(n_splits=n_trials, shuffle=True, random_state=42)

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
    
    if (save_res):
        # saves the scores and predictions for plotting
        io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_pred, model_type="general")
        # saves the last training iteration just in case
        io.save_model(mouse_day.mouseID, mouse_day.day, ridge)

    return scores, y_pred


def decode_behaviors(mouse_day: MouseDay, n_trials: int=10, save_res=False):
    """
    Decodes behaviors with models trained on that specific behavior. 
        Train: By Behavior, Test: By Behavior
    """
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_per_frame = mouse_day.get_trimmed_beh_labels()
    # Shitty workaround until i fix the behavior labels in the mouseday class
    labels = {key: value for key, value in mouse_day.BEHAVIOR_LABELS.items() if key != 6}
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

        # Make an overall prediction on all data using the last iteration of this behavior model
        # Not that precise of a location prediction but pred data more for sanity check than anything
        y_pred = ridge.predict(X)

        all_preds.append(y_pred)
        all_scores.append(scores)
        print("scores: ", scores)

        if (save_res):
            io.save_model(mouse_day.mouseID, mouse_day.day, ridge, labels[label])
            io.save_decoded_data(mouse_day.mouseID, mouse_day.day, scores, y_pred, model_type=labels[label])

    return all_scores, y_preds


def decode_behaviors_with_general(mouse_day: MouseDay, ntrials: int=10, save_res=False):
    """
    Decodes specific behavior samples using a model trained on the general population. 
        Train: General, Test: By Behavior
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
    # Shitty workaround until i fix the behavior labels in the mouseday class
    behaviors = {key: value for key, value in mouse_day.BEHAVIOR_LABELS.items() if key != 6}
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

def decode_by_cell(mouse_day: MouseDay, ntrials: int=10, save_res=False):
    """
    Decodes neural activity by cell type (inhibitory vs excitatory). Splits data into two groups of neurons and trains/tests two models on their own cell's data. 
        Train: By Cell, Test: By cell
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

def decode_by_class(mouse_day: MouseDay, beh_class: str, ntrials: int=10, save_res=False):
    """
    Decodes specific classes of behavior (either "natural" or "learned") and tests within that class. 
        Train: BehClass, Test: BehClass
    """
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


def smallest_sample_size(mouse_day: MouseDay, beh_list: list[int]) -> Tuple[int, list[int]]:
    """
    Helper function for decode by class to generate uniform test set sizes across all behaviors. 
    Finds the smallest amount of total samples amongst these behaviors, given all the samples in the mouseDay. 
    """
    min_samples = float('inf')
    sample_sizes = []
    beh_per_bin: np.ndarray = mouse_day.get_trimmed_beh_labels()
    for i, beh in enumerate(beh_list):
        num_samples = np.sum(beh_per_bin == beh)
        if num_samples <= min_samples:
            min_samples = num_samples
        sample_sizes.append(num_samples)
    
    return int(min_samples), sample_sizes



def decode_behaviors_with_class(mouse_day: MouseDay, train_class: list[int], test_class: list[int], mode: str, ntrials: int=10, save_res=False):
    """
    Decodes a specific behavior using an "in-class" or "cross-class" model. 
    """
    
    behavior_labels: dict[int, str] = mouse_day.BEHAVIOR_LABELS
    print("training data: ", ", ".join(behavior_labels[beh] for beh in train_class))
    print("testing data: ", ", ".join(behavior_labels[beh] for beh in test_class))

    spikes: np.ndarray = mouse_day.get_trimmed_spks()
    locs: np.ndarray = mouse_day.get_trimmed_avg_locs() 
    beh_per_bin: np.ndarray = mouse_day.get_trimmed_beh_labels()

    # to balance the covariates in the training class
    num_covars, all_sample_sizes = smallest_sample_size(mouse_day, train_class)

    samples_by_beh = []
    locs_by_beh = []
    bins_by_beh = [] # holds the overall timebins where the behaviors occur (for reconstructing predictions later)
    for i, train_beh in enumerate(train_class):
        # for debugging... delete later
        num_beh_spikes = np.sum(beh_per_bin == train_beh)
        print("behavior: ", train_beh, ", nsamples: ", num_beh_spikes)
        
        # pulls out all samples for all training behaviors
        beh_idcs = np.where(beh_per_bin == train_beh)
        samples_by_beh.append(spikes[beh_idcs])
        locs_by_beh.append(locs[beh_idcs])
        bins_by_beh.append(beh_idcs)


    test_size = -1 # splits for the test class set ONLY (nothing to do with how we divide the train class)
    train_size = -1
    # calculate the train/test sizes depending on the mode
    # For IN_CLASS: 
    #   test_size = 30% of the smallest behavior samples within the behavior class (so the performance is comparable across in_class behaviors)
    #   train_size = what we're pulling from the test dataset and putting into the model (affects covar balancing if smaller than smallest sample size)
    if mode == "in_class":
        test_class_idx = np.where(np.isin(train_class, test_class))[0][0]
        test_class_samples = samples_by_beh[test_class_idx]
        test_class_locs = locs_by_beh[test_class_idx]

        # num_covars being used differently here, but we want 30% of the smallest sample size within the training set to be able to compare across in_class behaviors
        test_size = int(TEST_SIZE * num_covars)

        # the number of samples pulled from the testing dataset to be put into the model
        train_size = len(test_class_samples) - test_size

        # update the minimum sample size if needed to maintain balanced covariates in the training set
        if (num_covars > train_size):
            num_covars = train_size
    elif mode == "cross_class":
        # min samples (of each train behavior in the test set) should remain the same
        # buttt testing size should be uniform across ALL behaviors (so we can compare cross testing performance across all behaviors)
        # train size isn't relevant
        test_size = smallest_sample_size(mouse_day, beh_list = BEH_CLASSES["all"])[0]
    
    print("NUM COVARIATE SAMPLES: ", num_covars)
        
    scores = []
    y_preds = []
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
        
        # Generating a test split... 
        # Holdout a test set from the training samples
        if mode == "in_class":
            # to ensure random pull per fold
            idcs = np.arange(0, len(test_class_samples)) 
            np.random.shuffle(idcs)
            test_class_samples = test_class_samples[idcs]
            test_class_locs = test_class_locs[idcs]
            # pull out 70% of this sample set to put into the model
            using_spikes, using_locs = test_class_samples[:train_size], test_class_locs[:train_size]
            # hold 30% of this sample set to use for testing
            holdout_spikes, holdout_locs = test_class_samples[train_size:], test_class_locs[train_size:]

            samples_by_beh[test_class_idx] = using_spikes
            locs_by_beh[test_class_idx] = using_locs

            # create the final test sets - limiting to test sizes to make it uniform across behaviors
            X_test = holdout_spikes[:test_size]
            y_test = holdout_locs[:test_size]

            # save the indicies for predictions later
            test_idcs_in_sample = idcs[train_size:]
            test_idcs_in_sample = test_idcs_in_sample[:test_size]
            test_idcs = bins_by_beh[test_class_idx][0][test_idcs_in_sample]
        # Test set for cross-class analysis
        elif mode == "cross_class":
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
    test_untrimmedlocs = mouse_day.get_all_avg_locations()
    test_spikes = mouse_day.get_trimmed_spks()
    test_labels = mouse_day.get_trimmed_beh_labels()
    test_untrimmed_labels = mouse_day.get_beh_labels()

    print("No Trim Locs: ", test_untrimmedlocs.shape)
    print("No Trim Spikes: ", mouse_day.cal_spks.T.shape)
    print("No Trim Labels: ", len(test_untrimmed_labels))

    print("Trimmed Locs: ", test_locs.shape)
    print("Trimmed Spikes: ", test_spikes.shape)

    print("Untrimmed labels: ", len(test_labels))
    return 0

def md_run(mouse_day: MouseDay, save_status=False):
    """
    Just to make sure all the mice are mice-ing. 
    Runs EVERYTHING.
    Saves if we specify. 
    """
    latency_check(mouse_day)
    dimensions_check(mouse_day)

    fig = myplot.plot_interp_test(mouse_day, mouse_day.seg_keys[0])
    plt.show()

    decode_general(mouse_day, save_res=save_status)
    fig1 = myplot.plot_kin_predictions(mouse_day)

    decode_behaviors(mouse_day, save_res=save_status)
    fig2 = myplot.plot_model_performance_swarm(mouse_day)

    decode_behaviors_with_general(mouse_day, save_res=save_status)
    fig3 = myplot.plot_general_performance_by_beh(mouse_day)

    decode_by_cell(mouse_day, save_res=save_status)
    fig4 = myplot.plot_cell_performance_swarm(mouse_day)

    for beh in LEARNED:
        scores, preds = decode_behaviors_with_class(mouse_day, train_class=BEH_CLASSES["learned"], test_class=BEH_CLASSES[beh], mode="in_class", save_res=save_status)
        scores1, preds1 = decode_behaviors_with_class(mouse_day, train_class=BEH_CLASSES["natural"], test_class=BEH_CLASSES[beh], mode="cross_class", save_res=save_status)

    for beh in NATURAL:
        scores, preds = decode_behaviors_with_class(mouse_day, train_class=BEH_CLASSES["natural"], test_class=BEH_CLASSES[beh], mode="in_class", save_res=save_status)
        scores1, preds1 = decode_behaviors_with_class(mouse_day, train_class=BEH_CLASSES["learned"], test_class=BEH_CLASSES[beh], mode="cross_class", save_res=save_status)

    fig5 = myplot.plot_performance_swarm(mouse_day, modes=myplot.IN_CLASS_MODE, mode_type="In-Class")
    fig6 = myplot.plot_performance_swarm(mouse_day, modes=myplot.CROSS_CLASS_MODE, mode_type="Cross-Class")

    plt.show()
    return 0


if __name__ == "__main__":
    mouseID = "mouse25"
    april25 = MouseDay(mouseID, "20240425")
    april24 = MouseDay(mouseID, "20240424")

    # s, p = decode_crossday_general(train_day=april24, test_day=april25, cross_test=True, save_res=True)
    # print("scores: ", s)
