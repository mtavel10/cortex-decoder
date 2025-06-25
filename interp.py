import numpy as np
import pandas as pd
import src.IO as io
import src.utils as ut
import math
from mouse import MouseDay
from typing import Optional, Tuple
from sklearn.model_selection import StratifiedShuffleSplit

if __name__ == "__main__":
    test_mouse = MouseDay("mouse25", "20240425")
    X = test_mouse.get_trimmed_spks()
    y = test_mouse.get_trimmed_avg_locs()
    beh_labels = test_mouse.get_beh_labels()
    print(beh_labels)
    print(len(beh_labels))
    print(test_mouse.cal_event_frames)
    print(len(test_mouse.cal_event_frames))

    # Splitter object
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.3, random_state=42)
    # splitter.split() randomly generates test/training indices for X based on a stratifier (the beh labels)
    # Loops n_splits times
    for i, (train_idcs, test_idcs) in enumerate(splitter.split(X, beh_labels)):
        print("Split: ", i)
        X_train, X_test = X[train_idcs], X[test_idcs]
        y_train, y_test = y[train_idcs], y[test_idcs]
        beh_train, beh_test = beh_labels[train_idcs], beh_labels[test_idcs]

    
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