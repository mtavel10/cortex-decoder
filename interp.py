import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut
import math
from mouse import MouseDay
from typing import Optional, Tuple
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, brier_score_loss

def ridge_regression(mouse_day: MouseDay):
    X = mouse_day.get_trimmed_spks()
    y = mouse_day.get_trimmed_avg_locs()
    beh_labels = mouse_day.get_beh_labels()

    # Splitter object
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.3, random_state=42)
    # splitter.split() randomly generates test/training indices for X based on a stratifier (the beh labels)
    # Loops n_splits times
    # Figure out how to do a 10 fold cross-validation
    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, beh_labels)):
        print("Split: ", i)
        X_train, X_test = X[train_idcs], X[test_idcs]
        y_train, y_test = y[train_idcs], y[test_idcs]
        beh_train, beh_test = beh_labels[train_idcs], beh_labels[test_idcs]

        # Cross-validate to find the best alpha
        ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        ridge_cv.fit(X_train, y_train)

        # Train the model
        ridge = Ridge(alpha=ridge_cv.alpha_)
        ridge.fit(X_train, y_train)
        print(ridge.score(X_test, y_test))

        # # Make predictions
        # y_pred = ridge.predict(X_test)

        # print("Mean scored error: ", mean_squared_error(y_test, y_pred))

def latency_check(mouse_day: MouseDay):
    print("# of timestamps (calcium): ", test_mouse.cal_ntimestamps)
    print("# of datapoints (calcium): ", test_mouse.cal_nframes)
    mouse_day.check_caltime_latency()

def dimensions_check(mouse_day: MouseDay):
    test_locs = mouse_day.get_trimmed_avg_locs()
    test_untrimmedlocs = mouse_day.get_all_avg_locations()
    test_spikes = mouse_day.get_trimmed_spks()
    test_labels = mouse_day.get_beh_labels()

    print("No trim location shape: ", test_untrimmedlocs.shape)
    print("No Trim Spikes shape: ", mouse_day.cal_spks.T.shape)
    print("Trimmed Location shape: ", test_locs.shape)
    print("Trimmed Spikes shape: ", test_spikes.shape)
    
    # print(mouse_day.cal_spks[:, 32:-32])
    print("Num labels: ", len(test_labels))

if __name__ == "__main__":
    test_mouse = MouseDay("mouse25", "20240425")

    # ridge_regression(test_mouse)
    latency_check(test_mouse)
   #  dimensions_check(test_mouse)

    
    # print(X.shape)
    # print(y.shape)

    # print(f"X shape: {X.shape}")
    # print(f"X.T @ X would be: {X.shape[1]} x {X.shape[1]}")
    # print(f"Estimated memory for X.T @ X: {X.shape[1]**2 * 8 / 1e6} MB")

    # # Ridge Regression
    # lam = 0.4

    # X_sqr = X.T @ X
    # w = np.linalg.inv(X_sqr + lam * np.eye(X_sqr.shape[0])) @ X.T @ y

    # print(w.shape)
    # y_hat = X @ w