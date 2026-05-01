# kNN at Scale — Spark RDD vs DataFrame Benchmark

HEC Paris MSc group project (Big Data Management course). We benchmarked k-Nearest-Neighbours classification across three implementations on synthetically generated datasets (up to 5,000 samples, 10 features, 3 classes via scikit-learn `make_classification`) to study scaling behaviour and the cost of going from raw RDDs to typed DataFrames. The algorithm is a port of the MapReduce-kNN design by Maillo, Triguero, and Herrera (2015) from Hadoop to Apache Spark.

## Implementations
- **`baseline_knn.py`** — single-node scikit-learn baseline (brute-force, O(n_train * n_features) per test point)
- **`spark_knn_rdd.py`** — Spark RDD implementation (mapPartitions + reduceByKey, faithful to the paper's MapReduce structure, vectorised numpy within each partition)
- **`spark_knn_df.py`** — Spark DataFrame / SQL implementation (cross join, window rank, Catalyst-optimized)
- **`experiments.py`** — orchestrates five timed experiments across implementations and dataset sizes; writes CSVs and plots to `results/`
- **`utils.py`** — shared helpers (distance, accuracy, train/test split, synthetic dataset loader)

## How to run

Requirements: Python 3.x, PySpark 4.x, OpenJDK 21+, scikit-learn, numpy, matplotlib.

```bash
pip install pyspark scikit-learn numpy matplotlib
python experiments.py
```

Results (CSVs and plots) are written to the `results/` subdirectory.

## Findings
- **Both Spark implementations match the sequential baseline exactly** on every test (Experiment 1: all three produce identical predictions at 60.0% accuracy on the 120-sample check dataset).
- **The RDD implementation consistently outperforms the DataFrame implementation** at all partition counts on local mode: ~5–7 s (RDD) vs. ~28 s (DataFrame) at n=3,000 — the DataFrame bottleneck is SQL cross-join planning, not compute.
- **The RDD crossover with sequential occurs between n=2,000 and n=5,000**: sequential takes 1.76 s at n=2,000 and 11.20 s at n=5,000, while RDD stays flat at ~7 s, demonstrating the parallelism benefit kicking in as data grows.
- **Increasing partitions beyond 2 does not improve RDD speed in local mode**: RDD runtime rises from 5.2 s (2 partitions) to 25 s (16 partitions) at n=3,000, as scheduling overhead dominates when partitions exceed available CPU threads.

Full writeup in `report.md`. Theory notes in `report_theory.md`.
