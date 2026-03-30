"""
utils.py
--------
Shared utility functions used across all k-NN implementations
(baseline sequential, MapReduce, and Spark variants).

Provides:
  - Euclidean distance computation
  - Accuracy evaluation
  - Train/test splitting
  - Synthetic dataset generation
  - A wall-clock timer context manager
"""

import time
import contextlib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def euclidean_distance(a, b):
    """
    Compute the Euclidean (L2) distance between two 1-D numpy arrays.

    Parameters
    ----------
    a : np.ndarray  -- first point, shape (n_features,)
    b : np.ndarray  -- second point, shape (n_features,)

    Returns
    -------
    float  -- non-negative scalar distance
    """
    # Subtract element-wise, square each difference, sum, then take the root
    return np.sqrt(np.sum((a - b) ** 2))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy as the fraction of correct predictions.

    Parameters
    ----------
    y_true : array-like  -- ground-truth labels, length n
    y_pred : array-like  -- predicted labels, length n

    Returns
    -------
    float  -- accuracy in [0.0, 1.0]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Boolean mask of correct predictions, then take the mean
    correct = y_true == y_pred
    return float(np.mean(correct))


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split arrays into random train and test subsets.

    Thin wrapper around sklearn's train_test_split to provide a consistent
    interface across all project modules.

    Parameters
    ----------
    X            : np.ndarray -- feature matrix, shape (n_samples, n_features)
    y            : np.ndarray -- label vector, shape (n_samples,)
    test_size    : float      -- proportion of the dataset for the test split
    random_state : int        -- random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test : four np.ndarrays
    """
    # Delegate entirely to sklearn; shuffle=True is the default
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def load_synthetic_dataset(n_samples, n_features=10, n_classes=3, random_state=42):
    """
    Generate a synthetic multi-class classification dataset.

    Uses sklearn's make_classification to produce a dataset with controlled
    properties (balanced classes, redundant/informative features).

    Parameters
    ----------
    n_samples    : int -- total number of samples to generate
    n_features   : int -- total number of features per sample
    n_classes    : int -- number of distinct class labels
    random_state : int -- random seed for reproducibility

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features) -- feature matrix
    y : np.ndarray, shape (n_samples,)            -- integer class labels
    """
    # n_informative must be >= n_classes; use at most n_features informative
    n_informative = max(n_classes, n_features // 2)

    # n_redundant features are linear combinations of the informative ones
    n_redundant = max(0, n_features - n_informative - 1)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )
    return X, y


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timer(label="Elapsed"):
    """
    Context manager that measures and prints wall-clock elapsed time.

    Usage
    -----
    with timer("my operation"):
        do_something()
    # prints: my operation: 0.1234 s

    Parameters
    ----------
    label : str -- prefix printed alongside the elapsed time
    """
    start = time.perf_counter()   # high-resolution wall-clock start
    try:
        yield                      # run the body of the with-block
    finally:
        end = time.perf_counter()  # wall-clock end
        elapsed = end - start
        print(f"{label}: {elapsed:.4f} s")
