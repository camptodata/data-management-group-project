"""
baseline_knn.py
---------------
Sequential baseline implementing exact brute-force k-Nearest Neighbours (k-NN)
classification.

Algorithm
---------
For every test point, compute the Euclidean distance to *all* training points,
select the k nearest neighbours by sorting those distances, and assign the
majority-vote class label among those neighbours.

Purpose in this project
-----------------------
1. Correctness reference: Spark/MapReduce variants are validated against this
   implementation to confirm they produce identical (or near-identical) results.
2. Speedup baseline: wall-clock time measured here is the denominator when
   computing parallel speedup ratios.

Time complexity  : O(n_test * n_train * n_features)
Space complexity : O(n_train) per test point (distance vector)
"""

import numpy as np
from collections import Counter

# Import shared utilities from the same package directory
from utils import (
    euclidean_distance,
    compute_accuracy,
    train_test_split_data,
    load_synthetic_dataset,
    timer,
)


# ---------------------------------------------------------------------------
# Core k-NN predictor
# ---------------------------------------------------------------------------

def knn_predict(X_train, y_train, X_test, k):
    """
    Predict class labels for every point in X_test using exact brute-force k-NN.

    For each test point the function:
      1. Computes the Euclidean distance to every training point.
      2. Identifies the k training points with the smallest distances (argsort).
      3. Takes a majority vote over the k neighbours' labels.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features) -- training feature matrix
    y_train : np.ndarray, shape (n_train,)            -- training labels
    X_test  : np.ndarray, shape (n_test,  n_features) -- test feature matrix
    k       : int                                     -- number of neighbours

    Returns
    -------
    predictions : np.ndarray, shape (n_test,) -- predicted integer class labels
    """
    predictions = []  # collect one predicted label per test point

    # --- outer loop: iterate over each test point ---
    for test_point in X_test:

        # 1. Distance loop ---------------------------------------------------
        #    Compute Euclidean distance from this test point to every training
        #    point.  Result is a 1-D array of length n_train.
        distances = np.array([
            euclidean_distance(test_point, train_point)
            for train_point in X_train
        ])

        # 2. argsort ---------------------------------------------------------
        #    np.argsort returns the *indices* that would sort the distances
        #    array in ascending order (smallest distance first).
        #    We only need the first k indices to get the k nearest neighbours.
        sorted_indices = np.argsort(distances)   # full sort; O(n log n)
        k_nearest_indices = sorted_indices[:k]   # slice to keep only k nearest

        # 3. Majority vote ---------------------------------------------------
        #    Retrieve the labels of the k nearest training points, then pick
        #    the most common label (ties broken by the first encountered in
        #    Counter, which is deterministic for a given input order).
        k_nearest_labels = y_train[k_nearest_indices]
        vote_counts = Counter(k_nearest_labels)           # {label: count, ...}
        majority_label = vote_counts.most_common(1)[0][0] # label with max count

        predictions.append(majority_label)

    return np.array(predictions)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run a demonstration and basic correctness check of the baseline k-NN."""

    print("=" * 60)
    print("Baseline Sequential k-NN — Brute-Force Exact Search")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Full-scale toy experiment
    #    200 training samples, 50 test samples, 3 classes, 10 features
    # ------------------------------------------------------------------
    print("\n[Experiment] 200-train / 50-test, k=3, 3 classes, 10 features")

    # Generate a reproducible synthetic dataset with 250 total samples
    X, y = load_synthetic_dataset(
        n_samples=250,
        n_features=10,
        n_classes=3,
        random_state=42
    )

    # Split into 200 train / 50 test (test_size = 50/250 = 0.20)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.20, random_state=42
    )

    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Test samples     : {X_test.shape[0]}")
    print(f"  Features         : {X_train.shape[1]}")

    # Run prediction under the timer context manager to capture wall-clock time
    with timer("  Runtime"):
        y_pred = knn_predict(X_train, y_train, X_test, k=3)

    # Compute and display accuracy
    accuracy = compute_accuracy(y_test, y_pred)
    print(f"  Accuracy (k=3)   : {accuracy:.4f}  ({accuracy*100:.1f} %)")

    # ------------------------------------------------------------------
    # 2. Tiny correctness check
    #    20 training points, 5 test points, k=1
    #    With k=1 the nearest neighbour IS the label — easy to verify manually.
    # ------------------------------------------------------------------
    print("\n[Correctness check] 20-train / 5-test, k=1")

    # Use a different random state so the tiny set is distinct
    X_small, y_small = load_synthetic_dataset(
        n_samples=25,
        n_features=10,
        n_classes=3,
        random_state=7
    )

    # First 20 rows as training, last 5 as test
    X_tr_small = X_small[:20]
    y_tr_small = y_small[:20]
    X_te_small = X_small[20:]
    y_te_small = y_small[20:]

    # k=1 NN: predicted label == label of the single closest training point
    y_pred_small = knn_predict(X_tr_small, y_tr_small, X_te_small, k=1)

    print(f"  True labels      : {y_te_small.tolist()}")
    print(f"  Predicted labels : {y_pred_small.tolist()}")

    # Compute accuracy for the tiny check
    tiny_accuracy = compute_accuracy(y_te_small, y_pred_small)
    print(f"  Accuracy (k=1)   : {tiny_accuracy:.4f}  ({tiny_accuracy*100:.1f} %)")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
