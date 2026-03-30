"""
spark_knn_df.py
---------------
DataFrame-based Spark implementation of exact k-Nearest Neighbours (k-NN)
classification, using the cross join + window rank strategy.

Algorithm overview
------------------
1. Convert training and test sets to Spark DataFrames.
2. Cross join every test row with every training row (cartesian product).
3. Compute the Euclidean distance as a new column using Spark SQL expressions.
4. Apply a Window function partitioned by test_id, ordered by distance ascending,
   and assign a rank to each neighbor.
5. Filter to keep only the top-k ranked neighbors per test point.
6. Majority-vote the label column grouped by test_id.
7. Collect results and align to the original X_test row order.

Contrast with RDD version
--------------------------
This module uses the high-level Spark SQL / DataFrame API (declarative style)
rather than low-level RDD transformations. The query planner can optimize
execution (projection push-down, broadcast hints, etc.), and the code is more
readable. However, the cross join is fundamentally O(n_train × n_test) in data
volume — the same asymptotic cost as the RDD approach — so it remains expensive
for large datasets; both approaches should be used with care beyond a few
thousand samples.

Correctness
-----------
This is still *exact* k-NN: every training point is considered for every test
point, so predictions are identical to the sequential baseline (baseline_knn.py)
given the same data and the same k.
"""

import collections
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

from utils import (
    compute_accuracy,
    load_synthetic_dataset,
    train_test_split_data,
    timer,
)
from baseline_knn import knn_predict


# ---------------------------------------------------------------------------
# Majority-vote UDF
# ---------------------------------------------------------------------------

# A Python UDF is necessary here because Spark SQL has no built-in "mode"
# aggregate function (the native approx_count_distinct / mode were only added
# in later Spark versions and are not universally available).  The UDF receives
# the collected list of neighbor labels for one test point and returns the most
# frequent label, matching the Counter-based tie-breaking used in the baseline.
@F.udf(IntegerType())
def majority_vote_udf(labels):
    """Return the most common integer label from a list of neighbor labels."""
    # Counter.most_common(1) returns [(label, count)] for the top element
    return collections.Counter(labels).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Euclidean distance column expression
# ---------------------------------------------------------------------------

def euclidean_distance_expr(feature_cols):
    """
    Build a Spark SQL Column expression for Euclidean distance.

    The cross-joined DataFrame has columns prefixed with "tr_" (train) and
    "te_" (test).  For each original feature name f, this function constructs:

        sqrt( (tr_f - te_f)^2 + (tr_f1 - te_f1)^2 + ... )

    Parameters
    ----------
    feature_cols : list[str]
        Original feature column names (e.g. ["feature_0", "feature_1", ...]).
        After the cross join, train columns are "tr_<name>" and test columns
        are "te_<name>".

    Returns
    -------
    pyspark.sql.Column
        A Column expression that evaluates to the Euclidean distance (float).
    """
    # Build one squared-difference term per feature dimension,
    # then reduce the list to a sum, and finally take the square root.
    squared_diff_terms = [
        F.pow(F.col(f"tr_{f}") - F.col(f"te_{f}"), 2)
        for f in feature_cols
    ]

    # Accumulate sum of squared differences across all dimensions
    sum_sq = squared_diff_terms[0]
    for term in squared_diff_terms[1:]:
        sum_sq = sum_sq + term  # add each subsequent squared difference

    # Euclidean distance = sqrt of the sum of squared differences
    return F.sqrt(sum_sq)


# ---------------------------------------------------------------------------
# Main k-NN function
# ---------------------------------------------------------------------------

def spark_knn_df(X_train, y_train, X_test, k, n_partitions=4):
    """
    Classify X_test using exact k-NN via Spark DataFrames.

    Parameters
    ----------
    X_train      : np.ndarray, shape (n_train, n_features)
    y_train      : np.ndarray, shape (n_train,)
    X_test       : np.ndarray, shape (n_test, n_features)
    k            : int   -- number of nearest neighbours
    n_partitions : int   -- number of partitions for the training DataFrame

    Returns
    -------
    predictions : np.ndarray, shape (n_test,)
        Predicted integer labels aligned to the row order of X_test.
    """
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]

    # Derive feature column names consistently across train and test DataFrames
    feature_cols = [f"feature_{i}" for i in range(n_features)]

    # ------------------------------------------------------------------
    # 1. Create SparkSession
    # ------------------------------------------------------------------
    # local[*] uses all available CPU cores; app name identifies the job in UI
    spark = SparkSession.builder \
        .appName("spark_knn_df") \
        .master("local[*]") \
        .getOrCreate()

    # Suppress verbose INFO logs so only warnings and errors appear
    spark.sparkContext.setLogLevel("ERROR")

    # ------------------------------------------------------------------
    # 2. Build training DataFrame  (feature_0 .. feature_n, label, train_id)
    # ------------------------------------------------------------------
    # Combine feature matrix, label vector, and a sequential ID into rows
    train_rows = [
        tuple(float(X_train[i, j]) for j in range(n_features))
        + (int(y_train[i]), i)           # append label and row index
        for i in range(n_train)
    ]
    train_schema = feature_cols + ["label", "train_id"]
    train_df = spark.createDataFrame(train_rows, schema=train_schema)

    # Repartition the training set to spread load across executors
    train_df = train_df.repartition(n_partitions)

    # ------------------------------------------------------------------
    # 3. Build test DataFrame  (feature_0 .. feature_n, test_id)
    # ------------------------------------------------------------------
    test_rows = [
        tuple(float(X_test[i, j]) for j in range(n_features)) + (i,)
        for i in range(n_test)
    ]
    test_schema = feature_cols + ["test_id"]
    test_df = spark.createDataFrame(test_rows, schema=test_schema)

    # ------------------------------------------------------------------
    # 4. Rename columns to avoid ambiguity after the cross join
    #    Train columns get prefix "tr_"; test columns get prefix "te_".
    # ------------------------------------------------------------------
    for col_name in feature_cols + ["label", "train_id"]:
        train_df = train_df.withColumnRenamed(col_name, f"tr_{col_name}")

    for col_name in feature_cols + ["test_id"]:
        test_df = test_df.withColumnRenamed(col_name, f"te_{col_name}")

    # ------------------------------------------------------------------
    # 5. Cross join: every test row paired with every training row
    #    Produces n_train * n_test rows — the cartesian product.
    # ------------------------------------------------------------------
    crossed_df = test_df.crossJoin(train_df)

    # ------------------------------------------------------------------
    # 6. Compute Euclidean distance column for each (test, train) pair
    # ------------------------------------------------------------------
    dist_expr = euclidean_distance_expr(feature_cols)
    crossed_df = crossed_df.withColumn("distance", dist_expr)

    # ------------------------------------------------------------------
    # 7. Window function: rank training points per test point by distance
    #    Partition by test_id so each test point has its own ranking.
    #    Order by distance ascending so the nearest neighbor gets rank 1.
    # ------------------------------------------------------------------
    window_spec = (
        Window
        .partitionBy("te_test_id")
        .orderBy(F.col("distance").asc())
    )
    ranked_df = crossed_df.withColumn("rank", F.rank().over(window_spec))

    # ------------------------------------------------------------------
    # 8. Filter: keep only the top-k neighbors per test point
    # ------------------------------------------------------------------
    topk_df = ranked_df.filter(F.col("rank") <= k)

    # ------------------------------------------------------------------
    # 9. Majority vote: for each test point collect neighbor labels and vote
    #    collect_list gathers all tr_label values in the group into a Python
    #    list, which is then passed to the majority_vote_udf.
    # ------------------------------------------------------------------
    preds_df = (
        topk_df
        .groupBy("te_test_id")
        .agg(
            majority_vote_udf(F.collect_list("tr_label")).alias("prediction")
        )
        .orderBy("te_test_id")  # ensure ascending test_id order before collect
    )

    # ------------------------------------------------------------------
    # 10. Collect results and convert to a numpy array
    #     Rows arrive in te_test_id order (enforced by orderBy above).
    # ------------------------------------------------------------------
    collected = preds_df.collect()  # brings data from all executors to driver
    predictions = np.array([row["prediction"] for row in collected])

    # ------------------------------------------------------------------
    # 11. Stop the SparkSession to release resources
    # ------------------------------------------------------------------
    spark.stop()

    return predictions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Spark DataFrame k-NN — Cross Join + Window Rank")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Generate a reproducible synthetic dataset
    # 200 train / 50 test, 10 features, 3 classes — same split as baseline
    # ------------------------------------------------------------------
    X, y = load_synthetic_dataset(
        n_samples=250,
        n_features=10,
        n_classes=3,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\n  Training samples : {X_train.shape[0]}")
    print(f"  Test samples     : {X_test.shape[0]}")
    print(f"  Features         : {X_train.shape[1]}")
    print(f"  k                : 3")
    print(f"  Partitions       : 4")

    # ------------------------------------------------------------------
    # Run Spark DataFrame k-NN and measure wall-clock time
    # ------------------------------------------------------------------
    print("\n[Spark DataFrame k-NN]")
    with timer("  Spark DF runtime"):
        spark_preds = spark_knn_df(X_train, y_train, X_test, k=3, n_partitions=4)

    spark_accuracy = compute_accuracy(y_test, spark_preds)
    print(f"  Accuracy (Spark DF, k=3) : {spark_accuracy:.4f}  ({spark_accuracy * 100:.1f} %)")

    # ------------------------------------------------------------------
    # Compare against the sequential baseline for correctness verification
    # ------------------------------------------------------------------
    print("\n[Baseline Sequential k-NN — for comparison]")
    with timer("  Baseline runtime"):
        baseline_preds = knn_predict(X_train, y_train, X_test, k=3)

    baseline_accuracy = compute_accuracy(y_test, baseline_preds)
    print(f"  Accuracy (Baseline, k=3) : {baseline_accuracy:.4f}  ({baseline_accuracy * 100:.1f} %)")

    # Check whether the two implementations agree on every test point
    n_agree = int(np.sum(spark_preds == baseline_preds))
    n_total = len(y_test)
    print(f"\n  Agreement: {n_agree}/{n_total} predictions match baseline")
    if n_agree == n_total:
        print("  CORRECTNESS CHECK PASSED: Spark DF output == Baseline output")
    else:
        print("  WARNING: Spark DF output differs from Baseline on some points")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
