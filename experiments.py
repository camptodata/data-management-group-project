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
