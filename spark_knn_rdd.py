"""
spark_knn_rdd.py
----------------
RDD-based Spark implementation of exact k-Nearest Neighbours (k-NN)
classification, faithful to the MapReduce design described in:

    Maillo et al. (2015) "A MapReduce Approach to k-Nearest Neighbour
    Classification for Big Data".

Design overview (mirrors the paper's MapReduce phases)
------------------------------------------------------
1. **Broadcast test set** – the full test set (with integer test_id keys) is
   serialised once and broadcast to every executor via sc.broadcast().  Every
   partition (= mapper) can then access every test point without any shuffle.

2. **mapPartitions → local top-k** – each Spark partition holds a shard of the
   training set.  The `local_topk` function iterates over that shard, computes
   Euclidean distances from every training point to every test point, and emits
   (test_id, [(dist, label), ...]) pairs containing only the local top-k
   candidates for each test point.

3. **reduceByKey → candidate merging** – `merge_candidates` is applied across
   partitions to combine local shortlists.  Because the true global top-k
   *must* appear in at least one partition's local top-k (the training set is
   partitioned exhaustively), merging local shortlists yields an exact result.

4. **Majority vote** – once each test point has its global top-k candidates,
   `majority_vote` picks the most frequent label.

Why RDD over DataFrames?
------------------------
Spark DataFrames (and Datasets) abstract away partitioning decisions and do not
expose a per-partition iterator with controlled early stopping.  RDD's
`mapPartitions` gives explicit partition-level control that maps 1-to-1 onto
the paper's mapper concept: one RDD partition == one mapper task.  This fidelity
is essential for a faithful implementation of the Maillo et al. algorithm.
"""

import heapq
import numpy as np
from collections import Counter

from pyspark import SparkContext, SparkConf

# Import shared project utilities
from utils import (
    compute_accuracy,
    load_synthetic_dataset,
    train_test_split_data,
    timer,
)

# Import baseline for correctness comparison
from baseline_knn import knn_predict


# ---------------------------------------------------------------------------
# Map phase: local top-k per partition
# ---------------------------------------------------------------------------

def local_topk(partition_iter, test_data_broadcast, k):
    """
    Mapper function applied to one Spark partition via mapPartitions().

    For every test point this mapper finds the k nearest training points
    *within the current partition* and yields a (test_id, candidates) pair.
    Across all partitions, these local shortlists are later reduced to obtain
    the exact global top-k.

    Parameters
    ----------
    partition_iter       : iterator -- yields (features_list, label) tuples
                           for training rows assigned to this partition
    test_data_broadcast  : pyspark.Broadcast -- wraps a list of
                           (test_id, features_list) pairs (the full test set)
    k                    : int -- number of nearest neighbours requested

    Yields
    ------
    (test_id, [(dist, label), ...]) -- local top-k candidates for each test
    point; the list has at most k entries, sorted ascending by distance
    """
    # --- Partition materialisation ------------------------------------------
    # Convert the partition iterator to a list so we can iterate over training
    # rows multiple times (once per test point).  An empty partition is handled
    # gracefully: the loop below simply does not execute.
    train_rows = list(partition_iter)

    # Early exit: nothing to do if this partition is empty
    if not train_rows:
        return

    # Unpack training features and labels into separate numpy arrays for
    # vectorised distance computation (avoids per-row Python overhead).
    train_features = np.array([row[0] for row in train_rows], dtype=np.float64)
    train_labels   = [row[1] for row in train_rows]

    # --- Retrieve broadcast test set ----------------------------------------
    # .value dereferences the broadcast variable; this is a list of
    # (test_id, features_list) pairs shared across all partitions.
    test_data = test_data_broadcast.value  # [(test_id, [f1, f2, ...]), ...]

    # --- Per-test-point distance computation and top-k selection ------------
    for test_id, test_features in test_data:

        # Convert test point to a numpy array for vectorised subtraction
        test_vec = np.array(test_features, dtype=np.float64)

        # Vectorised Euclidean distance: subtract train matrix row-wise,
        # square element-wise, sum across feature axis, then take the root.
        # Result shape: (n_train_in_partition,)
        diffs     = train_features - test_vec          # broadcast subtraction
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))  # L2 norm per row

        # --- Top-k selection ------------------------------------------------
        # heapq.nsmallest runs in O(n log k) and is more memory-efficient than
        # a full sort when k << partition_size.  Each element is a
        # (distance, label) tuple so the heap is ordered by distance first.
        candidates = heapq.nsmallest(
            k,
            zip(distances.tolist(), train_labels),
            key=lambda x: x[0]           # sort key: distance (first element)
        )

        # Yield the (test_id, local_candidates) pair for the reduce phase.
        # The list is already sorted ascending by distance (nsmallest contract).
        yield (test_id, candidates)


# ---------------------------------------------------------------------------
# Reduce phase: merge two local shortlists
# ---------------------------------------------------------------------------

def merge_candidates(list1, list2, k):
    """
    Merge two local top-k candidate lists into a single global top-k list.

    Correctness argument (from Maillo et al.):
        The training set is *partitioned* (each point lives in exactly one
        partition).  Therefore the true global k nearest neighbours of any
        test point must appear in *some* partition's local shortlist.  Merging
        all local shortlists and retaining the k smallest distances is
        therefore equivalent to a global search — the result is exact.

    Parameters
    ----------
    list1 : list of (float, label) -- local top-k from one partition/group
    list2 : list of (float, label) -- local top-k from another partition/group
    k     : int                    -- desired number of neighbours to keep

    Returns
    -------
    list of (float, label) -- merged, sorted, length <= k
    """
    # Concatenate the two shortlists (each of length at most k), then keep
    # only the k entries with the smallest distances.  sorted() is O(2k log 2k)
    # which is effectively O(k log k) — cheap because k is small.
    merged = sorted(list1 + list2, key=lambda x: x[0])  # sort by distance
    return merged[:k]  # retain only the global top-k after merging


# ---------------------------------------------------------------------------
# Prediction: majority vote
# ---------------------------------------------------------------------------

def majority_vote(candidates):
    """
    Derive a class prediction from a list of (distance, label) neighbour pairs.

    Parameters
    ----------
    candidates : list of (float, label) -- top-k nearest neighbours,
                 sorted ascending by distance

    Returns
    -------
    label -- the most frequently occurring label among the candidates.
             Tie-breaking: Counter.most_common() returns the label that was
             encountered first (in insertion order, i.e. the one that is
             closer to the test point when the list is distance-sorted), which
             provides a distance-weighted tie-break consistent with the paper.
    """
    # Extract labels from (distance, label) pairs; distances are not needed here
    labels = [label for _, label in candidates]

    # Counter tallies frequencies; most_common(1) returns [(label, count)] for
    # the label with the highest count.  When counts are equal, Python's Counter
    # preserves insertion order (Python 3.7+), so the nearest neighbour's label
    # wins — a natural distance-based tie-break.
    return Counter(labels).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Main Spark k-NN driver
# ---------------------------------------------------------------------------

def spark_knn_rdd(X_train, y_train, X_test, k, n_partitions=4):
    """
    Run exact k-NN classification using the Spark RDD MapReduce algorithm.

    This function encapsulates the full Spark lifecycle (context creation,
    RDD construction, distributed computation, context teardown) so it can be
    called as an ordinary Python function from test scripts or notebooks.

    Parameters
    ----------
    X_train      : np.ndarray, shape (n_train, n_features)
    y_train      : np.ndarray, shape (n_train,)
    X_test       : np.ndarray, shape (n_test,  n_features)
    k            : int -- number of nearest neighbours
    n_partitions : int -- number of Spark partitions (= number of mappers)

    Returns
    -------
    np.ndarray, shape (n_test,) -- predicted class labels in the same row
    order as X_test
    """
    # --- SparkContext setup -------------------------------------------------
    # local[*] spawns one thread per logical CPU core on the local machine.
    # A fresh SparkContext is created (and later stopped) inside this function
    # so that the caller does not need to manage Spark lifecycle state and so
    # that repeated calls in a test suite each get a clean context.
    conf = SparkConf().setAppName("MapReduce_kNN").setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sc.setLogLevel("WARN")  # suppress INFO noise so output is readable

    try:
        # --- Broadcast the full test set ------------------------------------
        # We attach a sequential test_id to each test point so we can
        # re-assemble predictions in the original row order after the reduce
        # phase.  Broadcasting avoids shipping the test set repeatedly over
        # the network — it is serialised once and cached in each executor's JVM.
        test_list = list(zip(
            range(len(X_test)),       # test_id: 0, 1, 2, ...
            X_test.tolist()           # convert numpy rows to plain Python lists
        ))
        test_bc = sc.broadcast(test_list)  # broadcast to all executors

        # --- Build training RDD ---------------------------------------------
        # Each element is a (features_list, label) tuple.  We convert numpy
        # rows to plain Python lists so Spark's default serialiser (pickle)
        # handles them without requiring numpy on the worker side.
        train_data = list(zip(
            X_train.tolist(),         # features as plain Python lists
            y_train.tolist()          # labels as plain Python scalars
        ))
        # Parallelise and repartition to exactly n_partitions shards so the
        # number of mappers matches the experiment's configuration parameter.
        train_rdd = sc.parallelize(train_data, numSlices=n_partitions)

        # --- Map phase: local top-k per partition ---------------------------
        # mapPartitions calls local_topk once per partition, passing an
        # iterator over that partition's (features, label) tuples.
        # The lambda captures test_bc and k from the enclosing scope.
        local_topk_rdd = train_rdd.mapPartitions(
            lambda it: local_topk(it, test_bc, k)
        )  # produces (test_id, [(dist, label), ...]) pairs

        # --- Reduce phase: merge shortlists across partitions ---------------
        # reduceByKey groups all (test_id, candidates) pairs that share the
        # same test_id (one per non-empty partition) and repeatedly applies
        # merge_candidates to produce a single global top-k list per test
        # point.  The lambda captures k so merge_candidates stays pure.
        global_topk_rdd = local_topk_rdd.reduceByKey(
            lambda a, b: merge_candidates(a, b, k)
        )  # produces (test_id, global_top_k_candidates) pairs

        # --- Collect results and apply majority vote ------------------------
        # Collect pulls all (test_id, candidates) pairs back to the driver.
        # For a k-NN inference workload the output is tiny (n_test items)
        # so collect() is safe here.
        results = global_topk_rdd.collect()  # list of (test_id, candidates)

        # Build a {test_id: predicted_label} dict via majority vote
        predictions_dict = {
            test_id: majority_vote(candidates)
            for test_id, candidates in results
        }

        # Align predictions to the original X_test row order using test_id
        # as the index.  This guarantees the returned array matches X_test
        # row-for-row even if Spark returns results in non-sequential order.
        predictions = np.array([
            predictions_dict[i] for i in range(len(X_test))
        ])

    finally:
        # --- Tear down SparkContext -----------------------------------------
        # Always stop the context, even if an exception occurred above, to
        # release JVM threads and port bindings cleanly.
        sc.stop()

    return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Demonstration and correctness check for the RDD-based Spark k-NN.

    Experiment 1 — full toy dataset
        200 training samples, 50 test samples, 10 features, 3 classes.
        Runs spark_knn_rdd(k=3, n_partitions=4) and reports accuracy vs.
        ground truth and vs. the sequential baseline.

    Experiment 2 — tiny correctness check
        20 training samples, 5 test samples, k=3.
        Compares spark_knn_rdd predictions to baseline_knn.knn_predict
        label-by-label to confirm the distributed implementation is exact.
    """
    print("=" * 65)
    print("RDD-based Spark k-NN  (MapReduce, Maillo et al. 2015)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Experiment 1: full-scale toy dataset
    # ------------------------------------------------------------------
    print("\n[Experiment 1] 200-train / 50-test | k=3 | 4 partitions")

    # Generate 250 samples; split 200/50
    X, y = load_synthetic_dataset(
        n_samples=250,
        n_features=10,
        n_classes=3,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.20, random_state=42
    )

    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Test samples     : {X_test.shape[0]}")
    print(f"  Features         : {X_train.shape[1]}")

    # Run Spark k-NN and time it
    with timer("  Spark RDD runtime"):
        spark_preds = spark_knn_rdd(
            X_train, y_train, X_test,
            k=3,
            n_partitions=4
        )

    # Accuracy against ground truth labels
    spark_acc = compute_accuracy(y_test, spark_preds)
    print(f"  Spark accuracy (k=3) : {spark_acc:.4f}  ({spark_acc * 100:.1f} %)")

    # Run sequential baseline for comparison (no timer needed here)
    baseline_preds = knn_predict(X_train, y_train, X_test, k=3)
    baseline_acc   = compute_accuracy(y_test, baseline_preds)
    print(f"  Baseline accuracy    : {baseline_acc:.4f}  ({baseline_acc * 100:.1f} %)")

    # Agreement between Spark and baseline (should be 100 % for exact algorithm)
    agreement = compute_accuracy(baseline_preds, spark_preds)
    print(f"  Spark vs baseline agreement : {agreement:.4f}  ({agreement * 100:.1f} %)")

    # ------------------------------------------------------------------
    # Experiment 2: tiny correctness check — compare label-by-label
    # ------------------------------------------------------------------
    print("\n[Experiment 2] Tiny correctness check | 20-train / 5-test | k=3")

    X_small, y_small = load_synthetic_dataset(
        n_samples=25,
        n_features=10,
        n_classes=3,
        random_state=7          # distinct seed keeps this set independent
    )

    # First 20 rows: training; last 5 rows: test
    X_tr_small = X_small[:20]
    y_tr_small = y_small[:20]
    X_te_small = X_small[20:]
    y_te_small = y_small[20:]

    # Spark prediction on tiny dataset
    spark_small = spark_knn_rdd(
        X_tr_small, y_tr_small, X_te_small,
        k=3,
        n_partitions=4          # more partitions than needed — tests empty-partition handling
    )

    # Baseline (sequential) prediction on the same tiny dataset
    baseline_small = knn_predict(X_tr_small, y_tr_small, X_te_small, k=3)

    print(f"  True labels      : {y_te_small.tolist()}")
    print(f"  Spark labels     : {spark_small.tolist()}")
    print(f"  Baseline labels  : {baseline_small.tolist()}")

    # Label-by-label match: lists must be identical for the algorithm to be exact
    match = np.all(spark_small == baseline_small)
    print(f"  Exact match with baseline : {'YES — implementation is correct' if match else 'NO — discrepancy detected'}")

    tiny_acc = compute_accuracy(y_te_small, spark_small)
    print(f"  Spark accuracy (tiny, k=3) : {tiny_acc:.4f}  ({tiny_acc * 100:.1f} %)")

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
