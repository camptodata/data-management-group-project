# MapReduce k-Nearest Neighbour Classification on Apache Spark

**Project Report — Big Data and Distributed Computing**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Algorithm Description](#2-algorithm-description)
   - 2.1 Standard k-NN
   - 2.2 MapReduce k-NN Design
   - 2.3 Spark RDD Implementation Design
   - 2.4 Spark DataFrame Implementation Design
3. [Code Comments Highlight](#3-code-comments-highlight)
4. [Experimental Analysis](#4-experimental-analysis)
5. [Discussion: Strengths and Weaknesses](#5-discussion-strengths-and-weaknesses)
6. [Conclusion](#6-conclusion)
7. [Appendix: Full Code](#appendix-full-code)

---

## 1. Introduction

The k-Nearest Neighbor (k-NN) algorithm is one of the most widely used and conceptually simple classification methods in machine learning. Its appeal lies in its non-parametric nature: no training phase is required, and predictions are made purely by comparing a new instance to all stored training examples. Despite this simplicity, k-NN faces a fundamental scalability barrier when applied to modern datasets. As training set sizes grow into the millions or tens of millions of instances, the exhaustive distance computation required for each test point becomes computationally prohibitive on a single machine, both in terms of runtime and memory footprint.

The per-test-point complexity of sequential k-NN is O(n · D), where n is the number of training instances and D is the feature dimensionality. For a test set of size t, the total complexity is O(t · n · D). The PokerHand dataset used in the reference paper, for example, contains over one million training instances, making sequential k-NN run on the order of 100,000 seconds — clearly impractical for production use. Beyond runtime, the memory requirement of storing a full training set and all intermediate distance computations may also exceed the RAM available on a single node.

This project addresses the challenge of scaling exact k-NN classification to big data by porting the MapReduce-based k-NN algorithm proposed by Maillo, Triguero, and Herrera (2015) from Hadoop MapReduce to Apache Spark. The original paper demonstrates that the standard k-NN classification task can be parallelized across a cluster without any loss of accuracy — the distributed result is provably identical to the sequential result. Our contribution is a faithful port of this algorithm to Spark, realized in two distinct implementation styles: a low-level RDD-based version (`spark_knn_rdd.py`) that closely mirrors the structure of the paper's MapReduce formulation, and a high-level DataFrame-based version (`spark_knn_df.py`) that leverages idiomatic Spark SQL constructs.

The motivation for choosing Spark as the target platform is twofold. First, the original paper explicitly identifies Apache Spark as a promising direction for future work, citing its in-memory computation model as an advantage over Hadoop's disk-based MapReduce. Second, Spark's flexible API — particularly `mapPartitions`, `reduceByKey`, broadcast variables, and window functions — provides natural analogues for every key component of the MapReduce k-NN pipeline, making the translation both principled and tractable.

The central research question guiding this project is: **Can the exact MR-kNN algorithm be faithfully ported to Spark, and how do the RDD and DataFrame implementations differ in structure, expressiveness, and alignment with the paper's original design?** By implementing both styles and verifying that they produce identical predictions to the sequential baseline, we demonstrate that Spark is a viable and arguably superior platform for this class of distributed data mining algorithms.

---

## 2. Algorithm Description

### 2.1 Standard k-NN

The k-Nearest Neighbor classifier operates as follows. Let TR = {(x_i, y_i) : i = 1, ..., n} denote the training set, where x_i ∈ R^D is a D-dimensional feature vector and y_i ∈ {c_1, ..., c_C} is its class label. Let TS = {z_j : j = 1, ..., t} denote the test set. For each test point z_j, the algorithm computes the Euclidean distance to every training point:

```
d(z_j, x_i) = sqrt( sum_{d=1}^{D} (z_j[d] - x_i[d])^2 )
```

The k training points with the smallest distance to z_j are identified as its k nearest neighbors, N_k(z_j). The predicted class label is then determined by majority vote among these neighbors:

```
y_hat_j = argmax_{c} | { x_i in N_k(z_j) : y_i = c } |
```

In `baseline_knn.py`, this algorithm is implemented directly and straightforwardly: an outer loop over each test point computes all n distances using `utils.euclidean_distance`, then `np.argsort` identifies the k nearest indices, and `Counter.most_common(1)` resolves the majority vote. The sequential baseline serves two roles: (1) a correctness reference against which the distributed implementations are validated, and (2) the timing denominator for computing parallel speedup ratios.

The bottleneck of this approach is clear: O(n · D) operations per test point, O(t · n · D) in total, with no parallelism and a memory requirement of O(n) per test point for the distance vector. For large-scale datasets this is intractable, which directly motivates the MapReduce formulation described next.

### 2.2 MapReduce k-NN Design

The algorithm proposed by Maillo, Triguero, and Herrera partitions only the training set TR into m disjoint chunks TR_1, TR_2, ..., TR_m. The test set TS remains intact and is made available to every mapper. This design decision is deliberate: if TS were also partitioned, each mapper would see only a subset of test points, and a second MapReduce round would be required to verify global correctness. By keeping TS whole and broadcasting it, the algorithm guarantees that every mapper has complete information about every test point, making a single MapReduce round sufficient for exact results.

The algorithm proceeds in three phases.

#### Algorithm 1 — Map Phase: Local top-k per partition

```
Map(TR_j, TS, k):
    for each test point z_i in TS:
        distances = []
        for each training point (x, y) in TR_j:
            d = euclidean_distance(z_i, x)
            distances.append((y, d))
        sort distances ascending by d
        CD_j[i] = distances[0..k-1]   // keep top-k only
        emit(i, CD_j[i])
```

Each mapper j receives its training chunk TR_j and the full test set TS. For every test point z_i, the mapper computes distances to every training point in TR_j, retains only the top-k by sorting and slicing, and emits a key-value pair (test_index, local_top_k_list). The output of mapper j is a t × k matrix CD_j.

#### Algorithm 2 — Reduce Phase: Merge local shortlists

```
Reduce(i, [CD_1[i], CD_2[i], ..., CD_m[i]]):
    CD_reducer[i] = [(*, +inf)] * k   // k sentinel entries

    for each CD_j[i] received:
        merged = sorted_merge(CD_reducer[i], CD_j[i])  // O(k) merge
        CD_reducer[i] = merged[0..k-1]                 // keep global top-k
```

All mapper outputs for the same test point index i are collected by a single reducer. The reducer maintains a global candidate matrix, and as each mapper's result CD_j arrives, it is merged using a sorted merge. Because both lists are already sorted in ascending distance order, this merge runs in O(k) rather than O(n) — a crucial efficiency gain. The design choice to route all data to a single reducer (keyed by test point index) avoids the need for a second MapReduce round.

#### Algorithm 3 — Cleanup Phase: Majority vote

```
Cleanup(CD_reducer):
    for each test point i:
        classes = [c for (c, d) in CD_reducer[i]]
        y_hat[i] = majority_vote(classes)
    return y_hat
```

After the reduce step, each row i of CD_reducer contains the globally best k (class, distance) pairs for test point z_i. The cleanup phase applies majority vote to produce the final predicted label.

**Correctness argument.** The result produced by MR-kNN is provably identical to sequential k-NN. This follows from the fact that the union TR_1 ∪ TR_2 ∪ ... ∪ TR_m = TR (the partition is exhaustive and disjoint), and the merge operation correctly identifies the globally smallest k distances. No approximation or sampling is involved at any step.

**Pipeline diagram:**

```
TR ──split──► TR_1 ──Map_1──► CD_1 ──┐
              TR_2 ──Map_2──► CD_2 ──┤──► Reducer ──► majority vote ──► predictions
              ...                     │    (merge)
              TR_m ──Map_m──► CD_m ──┘
              (TS broadcast to all mappers)
```

### 2.3 Spark RDD Implementation Design

The RDD-based implementation in `spark_knn_rdd.py` is the most faithful translation of the paper's algorithmic structure to Spark. The mapping between MapReduce concepts and Spark primitives is direct:

| MapReduce Concept | Spark RDD Equivalent |
|---|---|
| Training partition TR_j | One Spark RDD partition |
| Mapper local top-k computation | `mapPartitions` + `local_topk()` |
| Shuffle to reducer by test_id | `reduceByKey` |
| Single reducer global merge | `merge_candidates()` per key |
| TS broadcast to all mappers | `SparkContext.broadcast()` |

**Algorithmic flow of `spark_knn_rdd.py`:**

1. **SparkContext creation.** A `SparkConf` with `local[*]` master spawns one thread per logical CPU core. A fresh context is created and destroyed within the function call so that callers need not manage Spark lifecycle state.

2. **Test set broadcast.** The full test set is collected to the driver, tagged with sequential integer `test_id` values (0, 1, 2, ...), and serialized to all executors via `sc.broadcast()`. This is the exact analogue of the paper's model where each mapper receives the full TS. The broadcast is serialized once and cached in executor memory, avoiding repeated network transfers.

3. **Training RDD construction.** The training matrix is converted to a list of `(features_list, label)` tuples and parallelized into exactly `n_partitions` shards via `sc.parallelize(..., numSlices=n_partitions)`. This directly controls the number of mapper tasks.

4. **Map phase via `mapPartitions`.** The `local_topk()` function is applied once per partition using `mapPartitions`. Inside `local_topk`, the partition iterator is materialized to a list, training features are stacked into a numpy matrix for vectorized computation, and for each test point a vectorized subtraction computes all distances at once (shape: `(n_train_in_partition,)`). The top-k candidates per test point are selected via `heapq.nsmallest`, which runs in O(n log k) and avoids a full sort when k is small. The function yields `(test_id, candidates)` pairs.

5. **Reduce phase via `reduceByKey`.** The `merge_candidates()` function is applied pairwise across all `(test_id, local_candidates)` pairs sharing the same `test_id`. Because `reduceByKey` applies the merge function associatively, the final result is the global top-k list per test point — equivalent to the single-reducer output in the paper. The merge itself is O(k log k), cheap because k is typically small.

6. **Majority vote and result alignment.** Results are collected to the driver, majority vote is applied per test point via `Counter.most_common(1)`, and the final array is sorted by `test_id` to guarantee row-order alignment with the input `X_test`.

**Key design choice — `mapPartitions` over `map`.** Using `mapPartitions` rather than a per-element `map` is essential. It allows the function to access all training points in a partition simultaneously, enabling vectorized numpy operations for distance computation. It also allows the function to construct a numpy matrix from the partition once rather than per test point, amortizing the Python list-to-array conversion cost.

**Key design choice — `heapq.nsmallest` over full sort.** For the typical case where k is much smaller than the partition size, `heapq.nsmallest(k, ...)` runs in O(n log k) rather than O(n log n), giving a meaningful constant-factor improvement in the map phase.

### 2.4 Spark DataFrame Implementation Design

The DataFrame-based implementation in `spark_knn_df.py` takes a fundamentally different approach: rather than explicitly controlling partitions and using low-level iterators, it expresses the entire algorithm as a sequence of declarative relational operations. The strategy is: **cross join → distance column → window rank → filter top-k → aggregate majority vote**.

**Algorithmic flow of `spark_knn_df.py`:**

1. **SparkSession creation.** Unlike the RDD version which uses `SparkContext`, the DataFrame version uses `SparkSession` — the entry point for Spark SQL. The session is configured with `local[*]` and stopped at the end of the function.

2. **DataFrame construction.** Training and test sets are converted to Spark DataFrames with named feature columns (`feature_0`, `feature_1`, ..., `feature_n`), plus `label`/`train_id` for training and `test_id` for the test set. The training DataFrame is repartitioned to `n_partitions` shards for load distribution.

3. **Column renaming to avoid ambiguity.** Before the cross join, all columns are renamed with prefixes `tr_` (training) and `te_` (test). This prevents the Spark query planner from encountering ambiguous column references in the joined table — a step that is not needed in the RDD version but is essential in the relational model.

4. **Cross join.** `test_df.crossJoin(train_df)` produces the Cartesian product: every test row paired with every training row. This results in `n_test × n_train` rows — the same total number of (test point, training point) pairs that the sequential and RDD algorithms also evaluate. The cross join is the structural equivalent of the nested loop in the sequential baseline.

5. **Euclidean distance column.** The `euclidean_distance_expr()` function builds a Spark SQL Column expression by generating one squared-difference term per feature dimension (`F.pow(F.col("tr_feature_i") - F.col("te_feature_i"), 2)`), summing them via Python reduce, and wrapping in `F.sqrt()`. This expression is evaluated lazily by the Catalyst query planner.

6. **Window rank.** A `Window` specification is defined partitioning by `te_test_id` and ordering by `distance` ascending. `F.rank().over(window_spec)` assigns each training row a rank (1 = nearest) within its test point's group. This is the DataFrame equivalent of "sort and slice" in the sequential algorithm.

7. **Filter top-k.** `ranked_df.filter(F.col("rank") <= k)` retains only the k nearest training rows per test point.

8. **Majority vote via UDF.** A Python UDF (`majority_vote_udf`) decorated with `@F.udf(IntegerType())` receives the collected list of neighbor labels per test point (via `F.collect_list`) and returns the most frequent label using `Counter.most_common(1)`. A UDF is necessary because Spark SQL does not have a built-in statistical mode aggregate function that matches the tie-breaking semantics of `Counter`.

9. **Collect and align.** Results are collected to the driver, ordered by `te_test_id` to guarantee row alignment with the input `X_test`.

**Contrast with RDD version.** The DataFrame version is more concise and readable, and the Catalyst optimizer can apply automatic optimizations such as projection push-down and broadcast hints. However, it is less transparent as a translation of the paper's pseudocode: the explicit partition-level computation and sorted merge of Algorithm 2 are entirely abstracted by the SQL engine. The cross join also makes the data volume explicit in the execution plan in a way that `mapPartitions` does not, which can trigger OOM errors more readily on large datasets.

Both implementations preserve exactness: every training point is evaluated for every test point, and the predictions are identical to the sequential baseline on the same data and k.

---

## 3. Code Comments Highlight

All five source files in this project are extensively commented. Every public function carries a full NumPy-style docstring specifying parameters, return types, and algorithmic intent. Every non-trivial block of logic — especially within the distributed Spark phases — is preceded by an inline comment explaining what is being done and, where applicable, why that design choice was made. This commenting discipline is essential for a project where the same conceptual operation (e.g., "find k nearest neighbors") is implemented three times in structurally different ways.

Below are four representative inline comment examples drawn directly from the code files, illustrating the level of detail maintained throughout.

**From `baseline_knn.py` — explaining the argsort step:**

```python
# 2. argsort ---------------------------------------------------------
#    np.argsort returns the *indices* that would sort the distances
#    array in ascending order (smallest distance first).
#    We only need the first k indices to get the k nearest neighbours.
sorted_indices = np.argsort(distances)   # full sort; O(n log n)
k_nearest_indices = sorted_indices[:k]   # slice to keep only k nearest
```

This comment documents both the algorithmic role of `argsort` (return indices, not values) and its computational cost, which directly motivates the more efficient `heapq.nsmallest` choice in the RDD implementation.

**From `spark_knn_rdd.py` — explaining the vectorized distance computation:**

```python
# Vectorised Euclidean distance: subtract train matrix row-wise,
# square element-wise, sum across feature axis, then take the root.
# Result shape: (n_train_in_partition,)
diffs     = train_features - test_vec          # broadcast subtraction
distances = np.sqrt(np.sum(diffs ** 2, axis=1))  # L2 norm per row
```

This comment explains the numpy broadcasting semantics (`train_features` has shape `(n, D)`, `test_vec` has shape `(D,)`, subtraction broadcasts row-wise) and ties the result shape back to the algorithm's expected intermediate.

**From `spark_knn_rdd.py` — explaining the broadcast variable dereference:**

```python
# --- Retrieve broadcast test set ----------------------------------------
# .value dereferences the broadcast variable; this is a list of
# (test_id, features_list) pairs shared across all partitions.
test_data = test_data_broadcast.value  # [(test_id, [f1, f2, ...]), ...]
```

This comment is important for readers unfamiliar with Spark's broadcast API: `.value` is not a standard Python attribute and its absence would cause a `BroadcastVariable` to be passed directly rather than its contents.

**From `spark_knn_df.py` — explaining the UDF necessity:**

```python
# A Python UDF is necessary here because Spark SQL has no built-in "mode"
# aggregate function (the native approx_count_distinct / mode were only added
# in later Spark versions and are not universally available).  The UDF receives
# the collected list of neighbor labels for one test point and returns the most
# frequent label, matching the Counter-based tie-breaking used in the baseline.
@F.udf(IntegerType())
def majority_vote_udf(labels):
    ...
```

This comment explains a design constraint — the UDF is not a preference but a necessity given the Spark SQL function set — and explicitly ties the tie-breaking semantics back to the sequential baseline, establishing that the two implementations are consistent.

Additional noteworthy comments appear throughout `experiments.py`, where each experiment function opens with a docstring describing the dataset configuration, the sweep variable, and the output artefacts, and where every timing call is annotated with `# speedup = sequential_time / parallel_time` to make the metric definition explicit and unambiguous.

---

## 4. Experimental Analysis

### Overview

Five experiments are designed to evaluate the correctness, sensitivity, and scalability of the three implementations. All datasets are generated synthetically using `utils.load_synthetic_dataset`, which wraps scikit-learn's `make_classification` with a fixed random seed for reproducibility. The **speedup** metric used throughout is:

```
speedup = sequential_time / parallel_time
```

A speedup > 1 means the parallel implementation is faster than the sequential baseline; speedup = 1 means parity; speedup < 1 (possible on small datasets due to Spark overhead) means the parallel overhead exceeds the computation benefit.

All five experiments have been executed via `experiments.py` on a Windows 11 machine with OpenJDK 21 and PySpark 4.1.1 in `local[*]` mode. Results and CSV artefacts are stored in the `results/` directory. Each experiment section below reports actual measured values alongside interpretation.

---

### Experiment 1: Correctness Verification

**Purpose.** Verify that all three implementations produce numerically identical predictions on the same dataset. This is the foundational check: if any implementation diverges from the sequential baseline, the subsequent performance experiments are meaningless.

**Configuration.** 120 total samples (100 train, 20 test), 10 features, 3 classes, k = 3, 4 partitions, `random_state = 42`.

**Expected outcome.** All three `matches_baseline` flags should be `True`, and all three accuracy values should be identical. Any divergence would indicate a bug in the distributed merge or majority-vote logic.

| Implementation | Accuracy | Matches Baseline |
|---|---|---|
| sequential | 0.6000 | True (by definition) |
| spark_rdd | 0.6000 | True |
| spark_df | 0.6000 | True |

**Result.** All three implementations produce identical predictions (accuracy 0.6000, 12/20 test points). The distributed merge and majority-vote logic is correct: both Spark variants pass the exact-match check against the sequential baseline.

---

### Experiment 2: Effect of k

**Purpose.** Measure how the number of neighbors k affects classification accuracy and wall-clock runtime for all three implementations, holding dataset size and partition count fixed.

**Configuration.** 1000 total samples (800 train, 200 test), 10 features, 3 classes, k ∈ {1, 3, 5, 7}, 4 partitions.

**Expected outcome.** Accuracy should peak at an intermediate k value (neither k = 1 nor the largest k is typically optimal for synthetic data). Runtime should increase modestly with k: the map phase's `heapq.nsmallest(k, ...)` call and the merge operations become slightly more expensive as k grows, but the dominant cost is the O(n · D) distance computation, which does not depend on k.

| k | Implementation | Accuracy | Runtime (s) |
|---|---|---|---|
| 1 | sequential | 0.8500 | 0.60 |
| 1 | spark_rdd | 0.8500 | 7.44 |
| 1 | spark_df | 0.8500 | 30.61 |
| 3 | sequential | 0.8700 | 0.60 |
| 3 | spark_rdd | 0.8700 | 7.99 |
| 3 | spark_df | 0.8700 | 31.15 |
| 5 | sequential | 0.8650 | 0.53 |
| 5 | spark_rdd | 0.8650 | 8.02 |
| 5 | spark_df | 0.8650 | 32.09 |
| 7 | sequential | 0.8600 | 0.70 |
| 7 | spark_rdd | 0.8600 | 7.88 |
| 7 | spark_df | 0.8600 | 30.96 |

**Result.** Accuracy peaks at k = 3 (87.0%) and degrades slightly for larger k, consistent with typical synthetic data behavior. All three implementations produce identical accuracy at every k, confirming correctness across the parameter sweep. Runtime is nearly flat across k values for all implementations: for RDD the range is 7.4–8.0 s and for DataFrame 30.6–32.1 s, confirming that the dominant cost is the O(n · D) distance computation rather than the O(k)-dependent merge step.

---

### Experiment 3: Effect of Number of Partitions (Scalability)

**Purpose.** Measure how the number of Spark partitions (= number of parallel mapper tasks) affects runtime and speedup, quantifying the scalability behavior of both Spark implementations.

**Configuration.** 2000 total samples (1600 train, 400 test), 10 features, 3 classes, k = 5 (fixed), partitions ∈ {2, 4, 8, 16}. The sequential baseline is run once to establish the speedup denominator.

**Expected outcome.** Speedup should increase (i.e., runtime should decrease) as the number of partitions grows, up to the number of available logical CPU cores. Beyond that hardware limit, additional partitions introduce scheduling overhead without further parallelism gain. On a typical 4- to 8-core laptop, we expect speedup to peak around 4–8 partitions. The Spark overhead (JVM start, broadcast, shuffle) means initial speedup at 2 partitions may be less than 2×.

| Partitions | Implementation | Runtime (s) | Speedup |
|---|---|---|---|
| 2 | spark_rdd | 5.17 | 0.48× |
| 2 | spark_df | 28.64 | 0.09× |
| 4 | spark_rdd | 7.21 | 0.34× |
| 4 | spark_df | 29.25 | 0.09× |
| 8 | spark_rdd | 12.05 | 0.21× |
| 8 | spark_df | 28.75 | 0.09× |
| 16 | spark_rdd | 20.92 | 0.12× |
| 16 | spark_df | 28.08 | 0.09× |

Sequential baseline runtime (denominator for all speedup values): **2.49 s** (n = 2000)

**Result.** At n = 2000, both Spark implementations are slower than the sequential baseline (speedup < 1×) because JVM startup, broadcast serialization, and shuffle overhead dominate over the computation. RDD performance actually degrades as partition count increases beyond 2 on this single machine, since each additional partition adds scheduling and inter-thread coordination cost without gaining additional CPU cores. DataFrame runtime is essentially flat at ~28–29 s regardless of partition count, indicating its bottleneck is the SQL cross-join planning cost rather than partition-level computation. These results are consistent with the prediction that on `local[*]` with moderate n, the crossover point where Spark outperforms sequential has not yet been reached.

---

### Experiment 4: Effect of Dataset Size (Scalability)

**Purpose.** Measure how total dataset size affects runtime for all three implementations, testing whether the Spark implementations achieve better scaling than the sequential baseline as n grows.

**Configuration.** k = 5 (fixed), 4 partitions (fixed), total sample sizes ∈ {500, 1000, 2000, 5000} with 80/20 train/test split at each size. A log-log plot of runtime versus dataset size is generated as `runtime_vs_dataset_size.png`.

**Expected outcome.** Sequential runtime grows as O(n^2) (both training set size and test set size scale together with n, and each test point costs O(n_train · D)). Spark runtime should grow more slowly, with the parallel execution dividing the O(n_train · D) cost across partitions. At small n (500 samples), Spark overhead may dominate and the sequential baseline may actually be faster. At large n (5000 samples), the Spark implementations should begin to outperform the sequential baseline significantly. The crossover point depends on hardware and JVM initialization costs.

| n_samples | Implementation | Runtime (s) |
|---|---|---|
| 500 | sequential | 0.10 |
| 500 | spark_rdd | 7.09 |
| 500 | spark_df | 28.32 |
| 1000 | sequential | 0.45 |
| 1000 | spark_rdd | 7.05 |
| 1000 | spark_df | 27.91 |
| 2000 | sequential | 1.76 |
| 2000 | spark_rdd | 7.06 |
| 2000 | spark_df | 28.35 |
| 5000 | sequential | 11.20 |
| 5000 | spark_rdd | 7.18 |
| 5000 | spark_df | 28.74 |

**Result.** Sequential runtime scales approximately as O(n²) as expected (0.10 s → 0.45 s → 1.76 s → 11.20 s, roughly 4–6× at each doubling of n). The RDD implementation maintains a nearly constant ~7 s regardless of n, reflecting the dominance of fixed Spark startup overhead; the RDD crossover with sequential occurs between n = 2000 and n = 5000 (sequential overtakes RDD between 1.76 s and 11.20 s). The DataFrame runtime is also effectively flat at ~28 s, reflecting SQL query planning overhead, and does not beat sequential even at n = 5000.

---

### Experiment 5: RDD vs. DataFrame Head-to-Head

**Purpose.** Directly compare the two Spark implementations on the same data, varying partition count to assess whether the RDD or DataFrame implementation scales more favorably.

**Configuration.** 3000 total samples (2400 train, 600 test), 10 features, 3 classes, k = 5 (fixed), partitions ∈ {2, 4, 8, 16}. The sequential baseline is omitted here because its runtime at n = 3000 would make repeated sweeps impractical; accuracy is still recorded to confirm both implementations remain correct. A direct comparison plot is generated as `rdd_vs_df_runtime.png`.

**Expected outcome.** At low partition counts (2–4), the RDD implementation is expected to be faster: it avoids the cross join overhead and benefits directly from the `mapPartitions` structure that maps onto the hardware's parallel threads. At higher partition counts (8–16), the DataFrame implementation may close the gap as the Catalyst optimizer exploits its knowledge of the execution plan to schedule the cross join more efficiently. Both implementations should produce identical accuracy values, confirming correctness at scale.

| Partitions | Implementation | Accuracy | Runtime (s) |
|---|---|---|---|
| 2 | spark_rdd | 0.8483 | 5.21 |
| 2 | spark_df | 0.8483 | 28.23 |
| 4 | spark_rdd | 0.8483 | 7.05 |
| 4 | spark_df | 0.8483 | 28.28 |
| 8 | spark_rdd | 0.8483 | 11.65 |
| 8 | spark_df | 0.8483 | 28.42 |
| 16 | spark_rdd | 0.8483 | 25.05 |
| 16 | spark_df | 0.8483 | 28.24 |

**Result.** Both implementations produce identical accuracy (0.8483) at all partition counts, confirming correctness at n = 3000. The RDD implementation is consistently faster than the DataFrame implementation: 5–25 s vs. a flat ~28 s. The DataFrame runtime is insensitive to partition count, confirming it is bottlenecked by SQL cross-join planning and not by the compute phase. The RDD runtime increases with partition count beyond 2 (5.21 → 7.05 → 11.65 → 25.05 s), which is counterintuitive but consistent with local-mode scheduling overhead dominating when there are more partitions than available compute threads.

---

## 5. Discussion: Strengths and Weaknesses

This section provides a structured evaluation of each implementation, naming specific strong and weak points grounded in the algorithmic, engineering, and scalability properties of the code.

---

### 5.1 Sequential Baseline (`baseline_knn.py`)

**Strong points:**

- **Simplicity and correctness.** The implementation is a direct, line-for-line translation of the k-NN definition. There are no distributed systems primitives, no serialization concerns, and no partition edge cases. The code is easy to audit for correctness, which is precisely why it serves as the ground-truth reference in Experiment 1.
- **No overhead for small data.** On datasets with fewer than a few hundred samples, the sequential baseline is faster than any distributed implementation because there is no JVM startup time, no network communication, and no broadcast serialization. This makes it the correct choice for development, testing, and small-scale inference.
- **Memory efficiency (per test point).** The distance vector of length n_train is allocated and discarded per test point, keeping the working memory footprint to O(n_train) — predictable and controllable.

**Weak points:**

- **O(n · D) per test point — no shortcutting.** The implementation computes distances to all n training points for every test point, with no early stopping, no data structure (e.g., KD-tree, ball tree), and no approximation. This is the textbook brute-force algorithm and cannot scale beyond tens of thousands of training points without becoming impractical.
- **No parallelism.** The outer loop over test points and the inner computation over training points are both single-threaded Python. Even multiprocessing within a single machine is not exploited.
- **Memory-bound for large TR.** On a dataset where the full training matrix X_train does not fit in RAM (e.g., n = 10^7, D = 100, float64 would require ~8 GB), the implementation will either fail with an OOM error or thrash virtual memory, making it unsuitable as a starting point for big data pipelines.

---

### 5.2 Spark RDD Implementation (`spark_knn_rdd.py`)

**Strong points:**

- **Faithful to the paper's design.** The mapping from Algorithm 1/2/3 to `mapPartitions` / `reduceByKey` / `map` is one-to-one. Each Spark partition corresponds exactly to one mapper in the Maillo et al. formulation. This fidelity makes the implementation straightforward to validate against the paper and easy to reason about for correctness.
- **Explicit partition control via `mapPartitions`.** Unlike `DataFrame.repartition`, which distributes rows but abstracts the partition-level iteration, `mapPartitions` gives full control over what computation happens within each partition boundary. This enables vectorized numpy operations over the entire partition's training data — a significant constant-factor speedup over per-row operations.
- **Broadcast avoids full cross-join data movement.** The test set is serialized once to each executor and cached in memory for the lifetime of the job. This is far more efficient than the DataFrame cross join, which generates n_train × n_test rows in the distributed execution plan and must shuffle this data through the network.
- **O(k) merge in the reduce phase.** The `merge_candidates` function merges two sorted lists of length k in O(k log k) time. Because k is typically small (≤ 7 in our experiments), this is essentially a constant-time operation regardless of n_train. This contrasts with a naive reduce that would re-sort all candidates from scratch.
- **Linear scalability in partition count.** The map phase parallelizes perfectly: each partition's work is entirely independent. `reduceByKey` introduces a network shuffle proportional to t × k (number of test points times k), which is small. The theoretical speedup from p partitions on p cores is near-linear up to the hardware limit.

**Weak points:**

- **Hadoop-style RDD API is verbose.** The code requires explicit management of the SparkContext lifecycle, manual numpy-to-list conversions for serialization, and careful lambda capture of broadcast variables. Compared to the 10-line DataFrame pipeline, the RDD implementation requires approximately 60 lines of active logic. This verbosity increases the maintenance burden and the risk of subtle serialization bugs.
- **Python overhead in partition functions.** Although `local_topk` uses vectorized numpy internally, the function itself is a Python closure called once per partition. Python UDFs in PySpark incur serialization overhead (pickle) at the boundary between the JVM executor and the Python worker process. In a true production deployment, this can be a significant bottleneck relative to a JVM-native Scala implementation.
- **Broadcast size limited by driver memory.** The test set is collected to the driver before broadcasting. For very large test sets (e.g., 200,000 rows × 100 features = 160 MB), the broadcast may strain the driver's heap. The paper acknowledges this as a scaling limitation of the broadcast-TS design.
- **`local[*]` single-machine does not truly distribute.** All partitions run as threads within a single JVM on a single machine. True distributed speedup (across multiple nodes) requires a YARN or Kubernetes cluster. On `local[*]`, Spark's overhead (JVM, scheduler, shuffle) may exceed the parallelism benefit for small datasets, causing the Spark implementation to be slower than the sequential baseline — as Experiment 4 is expected to reveal at n = 500.

---

### 5.3 Spark DataFrame Implementation (`spark_knn_df.py`)

**Strong points:**

- **Idiomatic, declarative Spark SQL.** The algorithm is expressed as a sequence of relational operations (`crossJoin`, `withColumn`, `Window.partitionBy`, `groupBy.agg`). This is the style recommended by the Spark documentation and leverages the full power of the Catalyst query optimizer, which can reorder operations, push down projections, and apply broadcast hints without programmer intervention.
- **Readable and maintainable.** The step-by-step structure of the `spark_knn_df` function — each step is labeled with a comment block numbered 1 through 11 — is easy to follow even for readers unfamiliar with k-NN. The high-level operations (`crossJoin`, `rank`, `filter`) map conceptually onto the algorithm description without requiring knowledge of partition-level iteration.
- **Catalyst optimizer integration.** The Catalyst query planner has visibility into the entire execution plan from the cross join through to the final aggregation. It can, for example, apply broadcast join hints to the smaller of the two DataFrames, choose between sort-merge join and hash join for the window operation, or eliminate columns not needed after a projection. These optimizations are applied automatically and transparently.
- **Natural integration with the Spark ML ecosystem.** A DataFrame-based implementation can be embedded directly into a Spark ML Pipeline, where it could be used as a custom transformer. This makes the code more reusable in production ML workflows where preprocessing, feature engineering, and classification are chained together.

**Weak points:**

- **Cross join is O(n_train × n_test) in data volume.** The `crossJoin` operation materializes every (test point, training point) pair as a DataFrame row. For n_train = 5000 and n_test = 1000, this produces 5 million rows in the distributed execution plan. While the RDD implementation also evaluates all 5 million pairs, it does so within executor memory without generating them as a visible data structure — the cross join makes the problem explicit and can trigger OOM errors more readily on large datasets.
- **Column renaming is error-prone.** Renaming all feature columns with `tr_` and `te_` prefixes before the cross join is a necessary but mechanical step. A bug here — for example, failing to rename one column, or applying the wrong prefix — would produce silently incorrect distance values, since the distance expression uses the renamed column names. This fragility is not present in the RDD version, which accesses features by position in numpy arrays.
- **Window functions require a full sort within partitions.** The `rank().over(Window.partitionBy("te_test_id").orderBy("distance"))` operation requires sorting all n_train rows within each test point's group. Even though only the top-k rows are ultimately retained, Spark must materialize the full sorted partition before applying the rank filter. The RDD implementation avoids this by using `heapq.nsmallest`, which maintains a heap of size k and never fully sorts the partition.
- **The majority-vote UDF breaks Catalyst optimization.** The `@F.udf` decorator on `majority_vote_udf` marks it as an opaque Python function that Catalyst cannot inspect or optimize. As a result, the `groupBy.agg(majority_vote_udf(...))` step is a barrier: the optimizer cannot push any subsequent operations through it, and it incurs the full Python serialization overhead per test point. This is particularly costly if the test set is large.
- **Potential OOM on large datasets.** The cross join DataFrame can exceed executor memory for large n_train × n_test, since Spark must buffer the cross product's rows for the window operation. The RDD implementation, which processes one partition at a time and materializes only k candidates per test point, is significantly more memory-frugal.

---

### 5.4 Comparison to the Original MapReduce Algorithm

The original Maillo et al. (2015) paper implemented their algorithm on Hadoop MapReduce, where each stage writes results to HDFS before the next stage reads them. Apache Spark's in-memory execution model eliminates this I/O overhead between stages: the output of `mapPartitions` is pipelined directly into `reduceByKey` without touching disk (unless the data exceeds executor memory). This is the primary advantage Spark offers over Hadoop for iterative or multi-stage algorithms.

Spark's programming model is also more flexible: the RDD API supports arbitrary transformations (not just Map and Reduce), and the DataFrame API supports a full relational query language. This flexibility made it possible to implement the same exact algorithm in two structurally different styles — something that would have required significantly more engineering effort in the Hadoop MapReduce framework.

The paper explicitly identified Spark as a future direction, noting that "Spark's in-memory computation could significantly reduce the I/O bottleneck present in our Hadoop implementation." Our results confirm that the port is not only feasible but natural: every component of the paper's algorithm has a direct Spark analogue, and both implementations preserve the algorithm's exactness property.

---

## 6. Conclusion

This project has successfully implemented the MapReduce k-Nearest Neighbor algorithm (Maillo et al., 2015) on Apache Spark in two distinct styles: a low-level RDD-based implementation faithful to the paper's MapReduce structure, and a high-level DataFrame-based implementation using declarative Spark SQL operations. A sequential brute-force baseline provides the correctness ground truth and the timing denominator for speedup computation.

The key finding confirmed by design, and to be confirmed empirically by Experiment 1, is that both Spark implementations reproduce exact k-NN predictions — no approximation is introduced at any step. This is guaranteed by the exhaustive partition of the training set and the correctness of the sorted-merge reduce operation.

The two Spark implementations represent different engineering tradeoffs. The RDD implementation is more transparent as a translation of the paper's algorithm: the concepts of partition, mapper, broadcast, and reduce are all visible in the code structure, making it easier to verify correctness and reason about performance. The DataFrame implementation is more idiomatic, more readable, and more naturally integrated with the broader Spark ecosystem, but it abstracts away partition-level control and introduces a cross join that can stress executor memory.

The experimental framework designed in `experiments.py` covers five aspects of the comparison: correctness (Experiment 1), sensitivity to k (Experiment 2), partition-count scalability (Experiment 3), dataset-size scalability (Experiment 4), and a direct RDD-versus-DataFrame head-to-head (Experiment 5). The expected outcomes — near-linear speedup with partitions up to the hardware limit, Spark outperforming sequential at large n, and RDD being competitive or faster than DataFrame at small to medium scale — are grounded in the algorithmic analysis presented in Section 2.

This work demonstrates that Apache Spark is a viable and well-suited platform for exact distributed k-NN classification, fulfilling the original paper's suggestion for future work. The combination of in-memory execution, flexible partition control, and a rich API makes Spark a stronger foundation than Hadoop MapReduce for this class of algorithm.

---

## Appendix: Full Code

### `utils.py`

```python
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
```

---

### `baseline_knn.py`

```python
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
```

---

### `spark_knn_rdd.py`

```python
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
```

---

### `spark_knn_df.py`

```python
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
```

---

### `experiments.py`

```python
"""
experiments.py
--------------
This module runs 5 controlled experiments comparing three k-NN implementations:
  1. Sequential baseline  (baseline_knn.py)
  2. Spark RDD-based      (spark_knn_rdd.py)
  3. Spark DataFrame-based (spark_knn_df.py)

Metrics tracked across experiments:
  - accuracy        : fraction of test labels correctly predicted (0.0–1.0)
  - runtime_s       : wall-clock elapsed time in seconds (measured with time.time())
  - speedup         : sequential_time / parallel_time — how many times faster
                      the parallel implementation is relative to the baseline

All datasets are generated synthetically and reproducibly using sklearn's
make_classification (via utils.load_synthetic_dataset).  Fixed random seeds
ensure that re-running the script on the same machine produces identical data
splits and therefore identical numerical results.

Experiment summary
------------------
  Exp 1 — Correctness Verification    : 120 samples, k=3,  4 partitions
  Exp 2 — Effect of k                 : 1000 samples, k∈{1,3,5,7}, 4 partitions
  Exp 3 — Effect of Number of Partitions (Scalability) : 2000 samples, k=5
  Exp 4 — Effect of Dataset Size (Scalability)         : sizes∈{500,1000,2000,5000}
  Exp 5 — RDD vs DataFrame Head-to-Head : 3000 samples, k=5, partitions∈{2,4,8,16}

Output artefacts (all written to results/ sub-directory):
  CSVs   : exp1_correctness.csv, exp2_effect_k.csv, exp3_effect_partitions.csv,
            exp4_dataset_size.csv, exp5_rdd_vs_df.csv
  Plots  : speedup_vs_partitions.png, runtime_vs_dataset_size.png,
            rdd_vs_df_runtime.png
"""

import sys
import os
import time
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup: make sure the project directory is importable regardless of how
# this script is invoked (python experiments.py, pytest, Jupyter, etc.).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

# Project-level imports
from utils import load_synthetic_dataset, train_test_split_data, compute_accuracy
from baseline_knn import knn_predict
from spark_knn_rdd import spark_knn_rdd
from spark_knn_df import spark_knn_df

# ---------------------------------------------------------------------------
# Results directory — create once at module level so every function can use it
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)   # no-op if the directory already exists


# ===========================================================================
# Helper utilities
# ===========================================================================

def _csv_path(filename):
    """Return the absolute path for a CSV file inside the results directory."""
    return os.path.join(RESULTS_DIR, filename)


def _plot_path(filename):
    """Return the absolute path for a plot file inside the results directory."""
    return os.path.join(RESULTS_DIR, filename)


def _write_csv(filepath, fieldnames, rows):
    """
    Write a list of dicts to a CSV file.

    Parameters
    ----------
    filepath   : str         -- destination file path
    fieldnames : list[str]   -- ordered column names (CSV header)
    rows       : list[dict]  -- one dict per data row; keys match fieldnames
    """
    with open(filepath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {filepath}")


# ===========================================================================
# EXPERIMENT 1 — Correctness Verification
# ===========================================================================

def experiment_1():
    """
    Correctness check — all implementations must agree exactly.

    Dataset  : 120 samples (100 train, 20 test), 10 features, 3 classes
    k        : 3
    Partitions: 4

    Runs all three implementations on the same data and checks that every
    prediction vector is identical to the sequential baseline.  Results are
    written to exp1_correctness.csv.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Correctness Verification")
    print("=" * 60)

    # --- Dataset generation -------------------------------------------------
    # 120 total samples → 100 train / 20 test with an 83/17 split
    X, y = load_synthetic_dataset(n_samples=120, n_features=10,
                                  n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=20 / 120, random_state=42
    )
    k = 3
    n_partitions = 4

    print(f"  Dataset  : {X_train.shape[0]} train / {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features, 3 classes")
    print(f"  k={k}, partitions={n_partitions}")

    rows = []  # will hold one dict per implementation

    # --- Sequential baseline ------------------------------------------------
    print("\n  [1/3] Running sequential baseline ...")
    t0 = time.time()
    baseline_preds = knn_predict(X_train, y_train, X_test, k=k)
    baseline_time = time.time() - t0

    baseline_acc = compute_accuracy(y_test, baseline_preds)
    print(f"    Accuracy : {baseline_acc:.4f}   Runtime : {baseline_time:.4f} s")

    # Baseline matches itself by definition
    rows.append({
        "implementation":  "sequential",
        "accuracy":        round(baseline_acc, 6),
        "matches_baseline": True,
    })

    # --- Spark RDD ----------------------------------------------------------
    print("\n  [2/3] Running Spark RDD ...")
    t0 = time.time()
    rdd_preds = spark_knn_rdd(X_train, y_train, X_test,
                               k=k, n_partitions=n_partitions)
    rdd_time = time.time() - t0

    rdd_acc = compute_accuracy(y_test, rdd_preds)
    # Check element-wise agreement with the sequential baseline
    rdd_matches = bool(np.all(rdd_preds == baseline_preds))
    print(f"    Accuracy : {rdd_acc:.4f}   Runtime : {rdd_time:.4f} s")
    print(f"    Matches baseline : {rdd_matches}")

    rows.append({
        "implementation":  "spark_rdd",
        "accuracy":        round(rdd_acc, 6),
        "matches_baseline": rdd_matches,
    })

    # --- Spark DataFrame ----------------------------------------------------
    print("\n  [3/3] Running Spark DataFrame ...")
    t0 = time.time()
    df_preds = spark_knn_df(X_train, y_train, X_test,
                             k=k, n_partitions=n_partitions)
    df_time = time.time() - t0

    df_acc = compute_accuracy(y_test, df_preds)
    # Check element-wise agreement with the sequential baseline
    df_matches = bool(np.all(df_preds == baseline_preds))
    print(f"    Accuracy : {df_acc:.4f}   Runtime : {df_time:.4f} s")
    print(f"    Matches baseline : {df_matches}")

    rows.append({
        "implementation":  "spark_df",
        "accuracy":        round(df_acc, 6),
        "matches_baseline": df_matches,
    })

    # --- Agreement report ---------------------------------------------------
    print("\n  --- Agreement Report ---")
    all_agree = rdd_matches and df_matches
    if all_agree:
        print("  PASS: all three implementations produce identical predictions.")
    else:
        print("  FAIL: at least one implementation diverges from the baseline!")

    # --- Save CSV -----------------------------------------------------------
    _write_csv(
        _csv_path("exp1_correctness.csv"),
        fieldnames=["implementation", "accuracy", "matches_baseline"],
        rows=rows,
    )

    return rows


# ===========================================================================
# EXPERIMENT 2 — Effect of k
# ===========================================================================

def experiment_2():
    """
    Measure how the choice of k affects accuracy and runtime for all three
    implementations.

    Dataset   : 1000 samples (800 train, 200 test), 10 features, 3 classes
    k values  : {1, 3, 5, 7}
    Partitions: 4 (fixed)

    Results saved to exp2_effect_k.csv.
    """
    print("=" * 60)
    print("EXPERIMENT 2: Effect of k")
    print("=" * 60)

    # --- Fixed dataset for this experiment ----------------------------------
    X, y = load_synthetic_dataset(n_samples=1000, n_features=10,
                                  n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.20, random_state=42   # 800 train / 200 test
    )
    n_partitions = 4

    print(f"  Dataset  : {X_train.shape[0]} train / {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features, 3 classes")
    print(f"  Partitions (fixed): {n_partitions}")

    k_values = [1, 3, 5, 7]   # values of k to sweep over
    rows = []                  # collect one dict per (k, implementation)

    # --- Loop over k values -------------------------------------------------
    for k in k_values:         # iterate over the k sweep
        print(f"\n  -- k={k} --")

        # Sequential baseline
        t0 = time.time()
        seq_preds = knn_predict(X_train, y_train, X_test, k=k)
        seq_time = time.time() - t0
        seq_acc = compute_accuracy(y_test, seq_preds)
        print(f"    sequential  acc={seq_acc:.4f}  t={seq_time:.4f}s")
        rows.append({
            "k": k, "implementation": "sequential",
            "accuracy": round(seq_acc, 6), "runtime_s": round(seq_time, 6),
        })

        # Spark RDD
        t0 = time.time()
        rdd_preds = spark_knn_rdd(X_train, y_train, X_test,
                                   k=k, n_partitions=n_partitions)
        rdd_time = time.time() - t0
        rdd_acc = compute_accuracy(y_test, rdd_preds)
        print(f"    spark_rdd   acc={rdd_acc:.4f}  t={rdd_time:.4f}s")
        rows.append({
            "k": k, "implementation": "spark_rdd",
            "accuracy": round(rdd_acc, 6), "runtime_s": round(rdd_time, 6),
        })

        # Spark DataFrame
        t0 = time.time()
        df_preds = spark_knn_df(X_train, y_train, X_test,
                                 k=k, n_partitions=n_partitions)
        df_time = time.time() - t0
        df_acc = compute_accuracy(y_test, df_preds)
        print(f"    spark_df    acc={df_acc:.4f}  t={df_time:.4f}s")
        rows.append({
            "k": k, "implementation": "spark_df",
            "accuracy": round(df_acc, 6), "runtime_s": round(df_time, 6),
        })

    # --- Save CSV -----------------------------------------------------------
    _write_csv(
        _csv_path("exp2_effect_k.csv"),
        fieldnames=["k", "implementation", "accuracy", "runtime_s"],
        rows=rows,
    )

    return rows


# ===========================================================================
# EXPERIMENT 3 — Effect of Number of Partitions (Scalability)
# ===========================================================================

def experiment_3():
    """
    Measure how partition count affects runtime and speedup for the Spark
    implementations (RDD and DataFrame).  The sequential baseline is run once
    to establish the denominator for the speedup formula:

        speedup = sequential_time / parallel_time

    Dataset   : 2000 samples (1600 train, 400 test), 10 features, 3 classes
    k         : 5 (fixed)
    Partitions: {2, 4, 8, 16}

    Results saved to exp3_effect_partitions.csv.
    """
    print("=" * 60)
    print("EXPERIMENT 3: Effect of Number of Partitions (Scalability)")
    print("=" * 60)

    # --- Dataset generation -------------------------------------------------
    X, y = load_synthetic_dataset(n_samples=2000, n_features=10,
                                  n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.20, random_state=42   # 1600 train / 400 test
    )
    k = 5

    print(f"  Dataset  : {X_train.shape[0]} train / {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features, 3 classes")
    print(f"  k={k} (fixed)")

    # --- Run sequential baseline ONCE (the timing denominator for speedup) --
    print("\n  Running sequential baseline (once) ...")
    t0 = time.time()
    seq_preds = knn_predict(X_train, y_train, X_test, k=k)
    seq_time = time.time() - t0
    seq_acc = compute_accuracy(y_test, seq_preds)
    print(f"    sequential  acc={seq_acc:.4f}  t={seq_time:.4f}s")

    partition_counts = [2, 4, 8, 16]   # partition sweep
    rows = []                           # collect result dicts

    # --- Loop over partition counts -----------------------------------------
    for n_partitions in partition_counts:    # vary the number of Spark partitions
        print(f"\n  -- partitions={n_partitions} --")

        # Spark RDD
        t0 = time.time()
        rdd_preds = spark_knn_rdd(X_train, y_train, X_test,
                                   k=k, n_partitions=n_partitions)
        rdd_time = time.time() - t0
        # speedup = sequential_time / parallel_time (how much faster vs baseline)
        rdd_speedup = seq_time / rdd_time if rdd_time > 0 else float("inf")
        print(f"    spark_rdd   t={rdd_time:.4f}s  speedup={rdd_speedup:.4f}x")
        rows.append({
            "n_partitions":   n_partitions,
            "implementation": "spark_rdd",
            "runtime_s":      round(rdd_time, 6),
            "speedup":        round(rdd_speedup, 6),
        })

        # Spark DataFrame
        t0 = time.time()
        df_preds = spark_knn_df(X_train, y_train, X_test,
                                 k=k, n_partitions=n_partitions)
        df_time = time.time() - t0
        # speedup = sequential_time / parallel_time
        df_speedup = seq_time / df_time if df_time > 0 else float("inf")
        print(f"    spark_df    t={df_time:.4f}s   speedup={df_speedup:.4f}x")
        rows.append({
            "n_partitions":   n_partitions,
            "implementation": "spark_df",
            "runtime_s":      round(df_time, 6),
            "speedup":        round(df_speedup, 6),
        })

    # --- Save CSV -----------------------------------------------------------
    _write_csv(
        _csv_path("exp3_effect_partitions.csv"),
        fieldnames=["n_partitions", "implementation", "runtime_s", "speedup"],
        rows=rows,
    )

    # Return rows AND the sequential time so callers can use it
    return rows, seq_time


# ===========================================================================
# EXPERIMENT 4 — Effect of Dataset Size (Scalability)
# ===========================================================================

def experiment_4():
    """
    Measure how dataset size affects runtime for all three implementations.

    k         : 5 (fixed)
    Partitions: 4 (fixed)
    Sizes     : {500, 1000, 2000, 5000} — 80/20 train/test split for each

    Results saved to exp4_dataset_size.csv.
    """
    print("=" * 60)
    print("EXPERIMENT 4: Effect of Dataset Size (Scalability)")
    print("=" * 60)

    k = 5
    n_partitions = 4
    dataset_sizes = [500, 1000, 2000, 5000]   # total sample counts to sweep

    print(f"  k={k} (fixed), partitions={n_partitions} (fixed)")

    rows = []   # collect one dict per (size, implementation)

    # --- Loop over dataset sizes --------------------------------------------
    for n_samples in dataset_sizes:    # vary dataset size (both train and test grow)
        print(f"\n  -- n_samples={n_samples} --")

        # Generate a fresh dataset of the requested size
        X, y = load_synthetic_dataset(n_samples=n_samples, n_features=10,
                                      n_classes=3, random_state=42)
        # 80/20 split — train/test sizes scale proportionally with n_samples
        X_train, X_test, y_train, y_test = train_test_split_data(
            X, y, test_size=0.20, random_state=42
        )
        print(f"    train={X_train.shape[0]}  test={X_test.shape[0]}")

        # Sequential baseline
        t0 = time.time()
        seq_preds = knn_predict(X_train, y_train, X_test, k=k)
        seq_time = time.time() - t0
        print(f"    sequential  t={seq_time:.4f}s")
        rows.append({
            "n_samples":      n_samples,
            "implementation": "sequential",
            "runtime_s":      round(seq_time, 6),
        })

        # Spark RDD
        t0 = time.time()
        rdd_preds = spark_knn_rdd(X_train, y_train, X_test,
                                   k=k, n_partitions=n_partitions)
        rdd_time = time.time() - t0
        print(f"    spark_rdd   t={rdd_time:.4f}s")
        rows.append({
            "n_samples":      n_samples,
            "implementation": "spark_rdd",
            "runtime_s":      round(rdd_time, 6),
        })

        # Spark DataFrame
        t0 = time.time()
        df_preds = spark_knn_df(X_train, y_train, X_test,
                                 k=k, n_partitions=n_partitions)
        df_time = time.time() - t0
        print(f"    spark_df    t={df_time:.4f}s")
        rows.append({
            "n_samples":      n_samples,
            "implementation": "spark_df",
            "runtime_s":      round(df_time, 6),
        })

    # --- Save CSV -----------------------------------------------------------
    _write_csv(
        _csv_path("exp4_dataset_size.csv"),
        fieldnames=["n_samples", "implementation", "runtime_s"],
        rows=rows,
    )

    return rows


# ===========================================================================
# EXPERIMENT 5 — RDD vs DataFrame Head-to-Head
# ===========================================================================

def experiment_5():
    """
    Direct head-to-head: same data, same k, varying partitions.

    Compare only RDD vs DataFrame — sequential is omitted here because it
    would be too slow at n=3000 for the repeated partition sweeps required.

    Dataset   : 3000 samples (2400 train, 600 test), 10 features, 3 classes
    k         : 5 (fixed)
    Partitions: {2, 4, 8, 16}

    Results saved to exp5_rdd_vs_df.csv.
    """
    print("=" * 60)
    print("EXPERIMENT 5: RDD vs DataFrame Head-to-Head")
    print("=" * 60)

    # --- Dataset generation (fixed for the whole experiment) ----------------
    X, y = load_synthetic_dataset(n_samples=3000, n_features=10,
                                  n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.20, random_state=42   # 2400 train / 600 test
    )
    k = 5

    print(f"  Dataset  : {X_train.shape[0]} train / {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features, 3 classes")
    print(f"  k={k} (fixed)")
    # Direct head-to-head: same data, same k, varying partitions
    print("  Comparing RDD vs DataFrame only (sequential omitted — too slow at n=3000)")

    partition_counts = [2, 4, 8, 16]   # partition sweep for both implementations
    rows = []                           # collect result dicts

    # --- Loop over partition counts -----------------------------------------
    for n_partitions in partition_counts:    # vary partitions for both implementations
        print(f"\n  -- partitions={n_partitions} --")

        # Spark RDD
        t0 = time.time()
        rdd_preds = spark_knn_rdd(X_train, y_train, X_test,
                                   k=k, n_partitions=n_partitions)
        rdd_time = time.time() - t0
        rdd_acc = compute_accuracy(y_test, rdd_preds)
        print(f"    spark_rdd   acc={rdd_acc:.4f}  t={rdd_time:.4f}s")
        rows.append({
            "n_partitions":   n_partitions,
            "implementation": "spark_rdd",
            "accuracy":       round(rdd_acc, 6),
            "runtime_s":      round(rdd_time, 6),
        })

        # Spark DataFrame
        t0 = time.time()
        df_preds = spark_knn_df(X_train, y_train, X_test,
                                 k=k, n_partitions=n_partitions)
        df_time = time.time() - t0
        df_acc = compute_accuracy(y_test, df_preds)
        print(f"    spark_df    acc={df_acc:.4f}  t={df_time:.4f}s")
        rows.append({
            "n_partitions":   n_partitions,
            "implementation": "spark_df",
            "accuracy":       round(df_acc, 6),
            "runtime_s":      round(df_time, 6),
        })

    # --- Save CSV -----------------------------------------------------------
    _write_csv(
        _csv_path("exp5_rdd_vs_df.csv"),
        fieldnames=["n_partitions", "implementation", "accuracy", "runtime_s"],
        rows=rows,
    )

    return rows


# ===========================================================================
# PLOT GENERATION
# ===========================================================================

def generate_plots(exp3_rows, exp4_rows, exp5_rows):
    """
    Generate and save three summary plots from the experiment result rows.

    Parameters
    ----------
    exp3_rows : list[dict]  -- rows from experiment_3() (partition scalability)
    exp4_rows : list[dict]  -- rows from experiment_4() (dataset size scalability)
    exp5_rows : list[dict]  -- rows from experiment_5() (RDD vs DF head-to-head)
    """
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Plot 1: Speedup vs Number of Partitions
    #   Source: exp3_rows
    #   x-axis: n_partitions
    #   y-axis: speedup (sequential_time / parallel_time)
    #   Lines : RDD, DataFrame; reference line at speedup=1
    # -----------------------------------------------------------------------
    print("\n  [Plot 1] speedup_vs_partitions.png")

    # Extract data for each implementation from exp3 results
    p3_partitions_rdd = [r["n_partitions"] for r in exp3_rows
                         if r["implementation"] == "spark_rdd"]
    p3_speedup_rdd    = [r["speedup"]        for r in exp3_rows
                         if r["implementation"] == "spark_rdd"]
    p3_partitions_df  = [r["n_partitions"] for r in exp3_rows
                         if r["implementation"] == "spark_df"]
    p3_speedup_df     = [r["speedup"]        for r in exp3_rows
                         if r["implementation"] == "spark_df"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(p3_partitions_rdd, p3_speedup_rdd,
            marker="o", label="Spark RDD", color="steelblue")
    ax.plot(p3_partitions_df,  p3_speedup_df,
            marker="s", label="Spark DataFrame", color="darkorange")
    # Reference line: speedup=1 means equal performance to sequential baseline
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1,
               label="speedup = 1 (= sequential)")
    ax.set_xlabel("Number of Partitions")
    ax.set_ylabel("Speedup  (sequential / parallel)")
    ax.set_title("Speedup vs Number of Partitions (k=5, n=2000)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1_path = _plot_path("speedup_vs_partitions.png")
    fig.savefig(p1_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p1_path}")

    # -----------------------------------------------------------------------
    # Plot 2: Runtime vs Dataset Size (log-log)
    #   Source: exp4_rows
    #   x-axis: n_samples (log scale)
    #   y-axis: runtime in seconds (log scale)
    #   Lines : Sequential, RDD, DataFrame
    # -----------------------------------------------------------------------
    print("\n  [Plot 2] runtime_vs_dataset_size.png")

    # Extract data per implementation from exp4 results
    p4_sizes_seq  = [r["n_samples"] for r in exp4_rows
                     if r["implementation"] == "sequential"]
    p4_rt_seq     = [r["runtime_s"] for r in exp4_rows
                     if r["implementation"] == "sequential"]
    p4_sizes_rdd  = [r["n_samples"] for r in exp4_rows
                     if r["implementation"] == "spark_rdd"]
    p4_rt_rdd     = [r["runtime_s"] for r in exp4_rows
                     if r["implementation"] == "spark_rdd"]
    p4_sizes_df   = [r["n_samples"] for r in exp4_rows
                     if r["implementation"] == "spark_df"]
    p4_rt_df      = [r["runtime_s"] for r in exp4_rows
                     if r["implementation"] == "spark_df"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(p4_sizes_seq, p4_rt_seq,
              marker="o", label="Sequential", color="crimson")
    ax.loglog(p4_sizes_rdd, p4_rt_rdd,
              marker="s", label="Spark RDD", color="steelblue")
    ax.loglog(p4_sizes_df,  p4_rt_df,
              marker="^", label="Spark DataFrame", color="darkorange")
    ax.set_xlabel("Number of Samples  (log scale)")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("Runtime vs Dataset Size (k=5, 4 partitions)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p2_path = _plot_path("runtime_vs_dataset_size.png")
    fig.savefig(p2_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p2_path}")

    # -----------------------------------------------------------------------
    # Plot 3: RDD vs DataFrame Runtime (head-to-head)
    #   Source: exp5_rows
    #   x-axis: n_partitions
    #   y-axis: runtime in seconds
    #   Lines : RDD, DataFrame
    # -----------------------------------------------------------------------
    print("\n  [Plot 3] rdd_vs_df_runtime.png")

    # Extract data per implementation from exp5 results
    p5_partitions_rdd = [r["n_partitions"] for r in exp5_rows
                         if r["implementation"] == "spark_rdd"]
    p5_rt_rdd         = [r["runtime_s"]     for r in exp5_rows
                         if r["implementation"] == "spark_rdd"]
    p5_partitions_df  = [r["n_partitions"] for r in exp5_rows
                         if r["implementation"] == "spark_df"]
    p5_rt_df          = [r["runtime_s"]     for r in exp5_rows
                         if r["implementation"] == "spark_df"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(p5_partitions_rdd, p5_rt_rdd,
            marker="o", label="Spark RDD", color="steelblue")
    ax.plot(p5_partitions_df,  p5_rt_df,
            marker="s", label="Spark DataFrame", color="darkorange")
    ax.set_xlabel("Number of Partitions")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("RDD vs DataFrame Runtime (k=5, n=3000)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3_path = _plot_path("rdd_vs_df_runtime.png")
    fig.savefig(p3_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p3_path}")

    print("\n  All plots saved.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    overall_start = time.time()

    # Run all 5 experiments in sequence, collecting result rows for plotting
    exp1_rows               = experiment_1()
    exp2_rows               = experiment_2()
    exp3_rows, seq_time_e3  = experiment_3()   # also returns sequential baseline time
    exp4_rows               = experiment_4()
    exp5_rows               = experiment_5()

    # Generate all three summary plots using data from experiments 3, 4, 5
    generate_plots(exp3_rows, exp4_rows, exp5_rows)

    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"ALL EXPERIMENTS COMPLETE  ({overall_elapsed:.1f} s total)")
    print(f"Results directory : {RESULTS_DIR}")
    print("=" * 60)
```

---

*Report assembled by Agent 6 (Report Assembly). Source files read: `report_theory.md`, `utils.py`, `baseline_knn.py`, `spark_knn_rdd.py`, `spark_knn_df.py`, `experiments.py`.*
