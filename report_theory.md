# Theory: MapReduce k-Nearest Neighbor on Apache Spark

---

## 1. Introduction and Solution Description

The k-Nearest Neighbor (k-NN) algorithm is one of the most widely used and conceptually simple classification methods in machine learning. Its appeal lies in its non-parametric nature: no training phase is required, and predictions are made purely by comparing a new instance to all stored training examples. Despite this simplicity, k-NN faces a fundamental scalability barrier when applied to modern datasets. As training set sizes grow into the millions or tens of millions of instances, the exhaustive distance computation required for each test point becomes computationally prohibitive on a single machine, both in terms of runtime and memory footprint.

This project addresses the challenge of scaling exact k-NN classification to big data by porting the MapReduce-based k-NN algorithm proposed by Maillo, Triguero, and Herrera (2015) from Hadoop MapReduce to Apache Spark. The original paper demonstrates that the standard k-NN classification task can be parallelized across a cluster without any loss of accuracy — the distributed result is provably identical to the sequential result. Our contribution is a faithful port of this algorithm to Spark, realized in two distinct implementation styles: a low-level RDD-based version that closely mirrors the structure of the paper's MapReduce formulation, and a high-level DataFrame-based version that leverages idiomatic Spark SQL constructs.

The motivation for choosing Spark as the target platform is twofold. First, the original paper explicitly identifies Apache Spark as a promising direction for future work, citing its in-memory computation model as an advantage over Hadoop's disk-based MapReduce. Second, Spark's flexible API — particularly `mapPartitions`, `reduceByKey`, broadcast variables, and window functions — provides natural analogues for every key component of the MapReduce k-NN pipeline, making the translation both principled and tractable.

The central research question guiding this project is: **Can the exact MR-kNN algorithm be faithfully ported to Spark, and how do the RDD and DataFrame implementations differ in structure, expressiveness, and alignment with the paper's original design?** By implementing both styles and verifying that they produce identical predictions, we demonstrate that Spark is a viable and arguably superior platform for this class of distributed data mining algorithms.

---

## 2. Standard k-NN Algorithm

The k-Nearest Neighbor classifier operates as follows. Let TR = {(x_i, y_i) : i = 1, ..., n} denote the training set, where x_i ∈ R^D is a D-dimensional feature vector and y_i ∈ {c_1, ..., c_C} is its class label. Let TS = {z_j : j = 1, ..., t} denote the test set. For each test point z_j, the algorithm computes the Euclidean distance to every training point:

```
d(z_j, x_i) = sqrt( sum_{d=1}^{D} (z_j[d] - x_i[d])^2 )
```

The k training points with smallest distance to z_j are identified as its k nearest neighbors, N_k(z_j). The predicted class label is then determined by majority vote among these neighbors:

```
y_hat_j = argmax_{c} | { x_i in N_k(z_j) : y_i = c } |
```

The per-test-point complexity is O(n · D): for each of the t test points, distances to all n training points must be computed. The total complexity is therefore O(t · n · D). For large datasets — where n may exceed one million and D may be in the tens or hundreds — this is computationally intractable on a single machine. The PokerHand dataset used in the reference paper, for example, contains over one million instances, making sequential k-NN take on the order of 100,000 seconds. Beyond runtime, memory is also a constraint: storing the full training set and all pairwise distances may exceed available RAM. These twin bottlenecks of time and space are the motivating problem that distributed k-NN must address.

---

## 3. The MapReduce Paradigm

MapReduce is a distributed programming model originally developed at Google and popularized through the Apache Hadoop framework. The model decomposes a computation into two user-defined functions: **Map** and **Reduce**. The Map function takes an input key-value pair and emits zero or more intermediate key-value pairs. The framework then groups all intermediate values by key (the **shuffle** phase) and passes each group to a Reduce function, which aggregates them into a final output.

This paradigm is well-suited to distributed data mining for several reasons. First, it provides natural fault tolerance: if a worker node fails, the framework can re-execute only the affected map or reduce tasks on another node. Second, the Map phase exhibits no inter-node communication — each mapper operates independently on its local data partition, which means the most computationally expensive part of k-NN (distance computation) can be parallelized without synchronization overhead. Third, the shuffle phase handles data redistribution automatically, so the algorithm designer only needs to specify what key to aggregate on, not how to move data across the network.

The suitability of MapReduce for k-NN stems directly from the observation that, for a fixed test point, the global k nearest neighbors can be found by first computing local top-k candidates on each partition, and then merging those candidates. This insight is the core of the Maillo et al. algorithm.

---

## 4. The MR-kNN Algorithm

The algorithm proposed by Maillo, Triguero, and Herrera partitions only the training set TR into m disjoint chunks TR_1, TR_2, ..., TR_m. The test set TS remains intact and is made available to every mapper. The algorithm proceeds in three phases.

### 4.1 Map Phase

Each mapper j receives its training chunk TR_j and the full test set TS. For every test point z in TS, the mapper computes the Euclidean distance to every training point in TR_j, retains only the top-k nearest neighbors (the k smallest distances), and emits a key-value pair where the key is the test point index and the value is a sorted list of (class, distance) pairs of length k.

**Algorithm 1 — Map(TR_j, TS, k):**
```
for each test point z_i in TS:
    distances = []
    for each training point (x, y) in TR_j:
        d = euclidean_distance(z_i, x)
        distances.append((y, d))
    sort distances ascending by d
    CD_j[i] = distances[0..k-1]   // keep top-k only
    emit(i, CD_j[i])
```

The output of mapper j is a t × k matrix CD_j, where each row i contains the k best (class, distance) candidates found within TR_j for test point z_i.

### 4.2 Reduce Phase

All mapper outputs for the same test point index i are collected by a single reducer. The reducer maintains a global candidate matrix CD_reducer, initialized with k entries of distance +infinity for each test point. As each mapper's CD_j arrives, it is merged into CD_reducer using a sorted merge. Because both lists are already sorted in ascending order of distance, this merge runs in O(k) time rather than O(n).

**Algorithm 2 — Reduce(i, [CD_1[i], CD_2[i], ..., CD_m[i]]):**
```
CD_reducer[i] = [(*, +inf)] * k   // k sentinel entries

for each CD_j[i] received:
    merged = sorted_merge(CD_reducer[i], CD_j[i])  // O(k) merge
    CD_reducer[i] = merged[0..k-1]                 // keep global top-k
```

The design choice to route all data to a single reducer (keyed by test point index) avoids the need for a second MapReduce round to aggregate partial results from multiple reducers. This is feasible because the output size is small: t · k (class, distance) pairs, where t and k are both typically much smaller than n.

### 4.3 Cleanup Phase

After the reduce step, each row i of CD_reducer contains the globally best k (class, distance) pairs for test point z_i. The cleanup phase applies majority vote to produce the final predicted label.

**Algorithm 3 — Cleanup(CD_reducer):**
```
for each test point i:
    classes = [c for (c, d) in CD_reducer[i]]
    y_hat[i] = majority_vote(classes)

return y_hat
```

### 4.4 Exactness of the Algorithm

The result produced by MR-kNN is provably identical to the result of sequential k-NN. This follows from the fact that the union TR_1 ∪ TR_2 ∪ ... ∪ TR_m = TR (the partition is exhaustive and disjoint), and the merge operation correctly identifies the globally smallest k distances by sorting and retaining only the best candidates across all partitions. No approximation or sampling is involved at any step. The paper verifies this empirically: accuracy is identical regardless of the number of mappers used.

### 4.5 Pipeline Diagram

```
TR ──split──► TR_1 ──Map_1──► CD_1 ──┐
              TR_2 ──Map_2──► CD_2 ──┤──► Reducer ──► majority vote ──► predictions
              ...                     │    (merge)
              TR_m ──Map_m──► CD_m ──┘
              (TS broadcast to all mappers)
```

The training set is partitioned horizontally. The test set TS flows to every mapper via broadcast. Each mapper independently computes local top-k lists. All local lists converge at the single reducer, which performs O(k) merges to build the global top-k for each test point. The cleanup phase converts the global top-k lists into predicted class labels.

---

## 5. Spark Translation

Apache Spark provides a rich set of distributed computing primitives that map naturally onto the MapReduce k-NN algorithm. The translation from Hadoop MapReduce to Spark can be described component by component.

| MapReduce Concept | Spark Equivalent |
|---|---|
| Training data partition TR_j | Spark RDD/DataFrame partition |
| Mapper local top-k computation | `mapPartitions` transformation |
| Shuffle to reducer by test_id | `reduceByKey` or `groupBy` aggregation |
| Single reducer global merge | Global aggregation keyed by test point index |
| TS broadcast to all mappers | Spark broadcast variable |

### 5.1 RDD Implementation

The RDD-based implementation is the most faithful translation of the paper's algorithmic structure. The training set is loaded as a Spark RDD and naturally partitioned across the cluster. The test set is collected to the driver and broadcast to all executors using `SparkContext.broadcast()`, exactly analogous to the paper's model where each mapper receives the full TS.

The map phase is implemented using `mapPartitions`: for each partition (corresponding to TR_j), the closure computes Euclidean distances from every training point in the partition to every test point in the broadcast TS, retains the top-k by sorting and slicing, and emits (test_index, local_top_k_list) pairs. This is a direct code-level translation of Algorithm 1.

The reduce phase is implemented using `reduceByKey`: all (test_index, local_top_k_list) pairs sharing the same test index are merged pairwise using the sorted merge of Algorithm 2. Because `reduceByKey` applies the merge function associatively, the final result for each key is the global top-k list, equivalent to the single-reducer output in the paper.

The cleanup phase (majority vote) is a final `map` over the aggregated RDD, converting each (test_index, global_top_k_list) pair into a (test_index, predicted_class) pair.

### 5.2 DataFrame Implementation

The DataFrame-based implementation takes a more idiomatic Spark SQL approach. Rather than explicitly controlling partitioning and using `mapPartitions`, it leverages a cross join between the test set and the training set to compute all pairwise distances as a distributed table operation. A window function (`rank() OVER (PARTITION BY test_id ORDER BY distance ASC)`) then identifies the top-k neighbors for each test point. A final `groupBy` and aggregation applies majority vote.

This approach produces the same result as the RDD version and the sequential k-NN, but the structure is quite different: the algorithm is expressed as a sequence of relational operations rather than as an explicit parallel loop. The DataFrame version benefits from Spark's Catalyst query optimizer, which can reorder and push down operations automatically. However, it is less transparent as a translation of the paper's pseudocode, since the explicit partition-level computation and sorted merge are abstracted away by the SQL engine.

Both implementations preserve exactness: no approximation is introduced, and the predicted labels match the sequential k-NN output on identical data.

---

## 6. Important Design Choices

Several design decisions in both the original paper and our Spark implementations deserve explicit attention.

**Broadcast of the test set.** In the original Hadoop implementation, each mapper reads the test set TS from HDFS line by line as part of its input. In Spark, the natural analogue is a broadcast variable: the driver collects TS and serializes it to every executor once, where it is cached in memory for the duration of the job. This avoids repeated network transfers and is essential for performance when TS is large. For very large test sets, this broadcast becomes a bottleneck, but for the dataset sizes considered in the paper (the test fold of PokerHand is approximately 205,000 instances), broadcasting TS is practical.

**Single reducer vs. distributed aggregation.** The paper's choice to use a single reducer keyed by test point index is deliberate: it minimizes network traffic by ensuring that each intermediate result is sent exactly once to one location. In Spark, `reduceByKey` distributes this aggregation across the cluster's partitions, but the logical result is equivalent. The key insight is that the merge operation (Algorithm 2) is associative and commutative with respect to maintaining the global top-k, so the order in which local results are merged does not affect the final output.

**Exactness vs. approximation.** Many distributed k-NN approaches sacrifice exactness for speed, using locality-sensitive hashing or random projections to avoid exhaustive distance computation. The Maillo et al. algorithm and both of our Spark implementations are exact: every training point is considered for every test point, and the global top-k is computed without approximation. This comes at a cost — the O(t · n · D) distance computations cannot be reduced — but the parallelization achieves linear speedup in the number of mappers, making exact k-NN tractable at scale.

**Partitioning the training set only.** The algorithm partitions TR but not TS. This is a deliberate choice: if TS were also partitioned, each mapper would only see a subset of test points, and a second MapReduce round would be required to verify that no closer neighbor exists in a different training partition for any given test point. By keeping TS intact and broadcasting it, the algorithm guarantees that every mapper has complete information about every test point, making a single MapReduce round sufficient for exact results. The cost is that the broadcast size grows with |TS|, but this is generally acceptable since the test set is typically smaller than the training set.

**Number of mappers as a tuning parameter.** The paper demonstrates that increasing the number of mappers (equivalently, decreasing the size of each training partition) yields near-linear speedup up to 256 mappers on their 16-node cluster, reaching approximately 156x speedup at k=7. In Spark, the number of partitions plays the same role and can be controlled via `repartition()` or by configuring the default parallelism. Unlike MapReduce, Spark can also exploit pipelining and in-memory caching to reduce the overhead of multiple iterations, making it potentially more efficient for repeated experiments with varying k values.

Together, these design choices make MR-kNN a principled and practical algorithm for distributed classification, and Apache Spark a natural and well-suited platform for its implementation.
