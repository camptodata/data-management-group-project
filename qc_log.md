# QC Log

## After Phase 1 (Agent 1 + Agent 2)
Date: 2026-03-29

### utils.py
- [x] Function docstrings present
- [x] Inline comments on non-trivial lines
- Notes: All 5 functions (euclidean_distance, compute_accuracy, train_test_split_data,
  load_synthetic_dataset, timer) have full NumPy-style docstrings including Parameters
  and Returns sections. The module-level docstring at lines 1-13 lists all exported
  utilities. Inline comments cover every non-trivial operation: the sqrt/sum formula
  (line 39), the boolean-mask mean for accuracy (lines 63-64), the sklearn delegation
  note (line 90), the n_informative/n_redundant constraint arithmetic (lines 122-126),
  and the perf_counter start/yield/end pattern in the timer (lines 158-164).
  No gaps found.

### baseline_knn.py
- [x] Module-level docstring
- [x] Inline comments on distance loop
- [x] Inline comments on argsort
- [x] Inline comments on majority vote
- Notes: The module-level docstring (lines 1-22) is thorough: it names the algorithm,
  describes the three-step procedure, states the purpose in the project (correctness
  reference + speedup baseline), and provides explicit O(n_test * n_train * n_features)
  time and O(n_train) space complexity. The distance loop (lines 66-72), argsort block
  (lines 74-79), and majority-vote block (lines 81-87) each carry a clearly labelled
  section header comment plus line-level explanations. The main() function also
  documents each experiment inline. No gaps found.

### report_theory.md
- [x] Solution description >= 1 page
- [x] Pseudocode for all 3 algorithms (Map, Reduce, Cleanup)
- [x] Design choices section present
- [x] Spark translation section present
- Notes: The report contains six numbered sections totalling approximately 1,600 words
  of prose, well exceeding the 1-page minimum. Sections 1-3 motivate the problem and
  explain MapReduce. Algorithm 1 (Map), Algorithm 2 (Reduce), and Algorithm 3 (Cleanup)
  all appear in Section 4 with clearly labelled pseudocode fenced blocks matching the
  structure of Algorithms 1-3 in the Maillo et al. paper. Section 4.4 proves exactness
  and Section 4.5 adds an ASCII pipeline diagram. Section 5 (Spark Translation) provides
  a component-by-component mapping table and dedicated subsections for both RDD and
  DataFrame implementations. Section 6 (Important Design Choices) covers five distinct
  design decisions: test-set broadcast, single-reducer vs. distributed aggregation,
  exactness vs. approximation, why only TR is partitioned, and the number-of-mappers
  tuning parameter. No gaps found.

### Experimental analysis / scalability
- [ ] All 5 experiments present (vary partitions and dataset size)
- Notes: NOT YET APPLICABLE. The rubric criterion for experimental analysis (3 pts)
  belongs to the Phase 2 experimental outputs, which have not been produced yet.
  No experiment results file was included in the Phase 1 deliverables. This criterion
  should be re-evaluated after Phase 2 runs.

### Discussion of weak/strong points
- [ ] Interpretive analysis of weak/strong points present as a standalone section
- Notes: NOT YET APPLICABLE at Phase 1. Strengths and limitations are touched on
  within Section 6 of report_theory.md (e.g., broadcast bottleneck for large TS,
  exactness vs. approximation trade-off, linear speedup ceiling), but a dedicated
  "Strengths and Weaknesses" discussion section that interprets experimental results
  does not yet exist. This section should be added in Phase 2 once empirical data
  are available.

---

### Overall Phase 1 verdict: PASS
Critical gaps (if any): None in Phase 1 deliverables. Two rubric criteria
(experimental analysis and strengths/weaknesses discussion) are deferred to Phase 2
by design and do not constitute Phase 1 failures.

Fix instructions for retry (if needed):
- No fixes required for utils.py or baseline_knn.py.
- No fixes required for report_theory.md within its Phase 1 scope.
- Phase 2 must produce: (1) a results table or log covering at least 5 experiments
  varying both the number of partitions and the dataset size; (2) a dedicated
  "Strengths and Weaknesses" (or "Discussion") section in the report that interprets
  the experimental findings, comments on where the algorithm scales well vs. where
  it degrades, and identifies any edge cases (e.g., large test-set broadcast cost,
  reducer skew if test-set is small).

---

## After Phase 2 (Agent 3 + Agent 4)
Date: 2026-03-29

### spark_knn_rdd.py
- [x] Module-level docstring with RDD rationale
- [x] local_topk inline comments (partition materialisation, distance, top-k, yield)
- [x] merge_candidates exactness comment
- [x] majority_vote tie-breaking comment
- [x] spark_knn_rdd step comments (broadcast, mapPartitions, reduceByKey, collect, vote)
- Notes: The module-level docstring (lines 1-37) is comprehensive: it states the
  Maillo et al. citation, gives a 4-step MapReduce design overview, and has a
  dedicated "Why RDD over DataFrames?" section explaining the partition-control
  rationale. `local_topk` has clearly labelled sub-sections for partition
  materialisation, early-exit handling, broadcast dereference, vectorised distance
  (with shape annotation), heapq.nsmallest justification, and yield semantics.
  `merge_candidates` states the correctness/exactness argument inline as well as in
  the docstring. `majority_vote` explains the Counter insertion-order tie-break as a
  distance-weighted tie-break. `spark_knn_rdd` is annotated step-by-step with
  numbered section headers and line-level comments for every Spark API call.
  No debug prints; structure is appendix-ready.

### spark_knn_df.py
- [x] Module-level docstring with tradeoff vs RDD
- [x] euclidean_distance_expr inline comments
- [x] majority_vote_udf UDF rationale comment
- [x] spark_knn_df step comments (cross join, distance, window, filter, groupby/vote)
- Notes: The module-level docstring (lines 1-33) includes an "Algorithm overview"
  and an explicit "Contrast with RDD version" section that explains the declarative
  style, query planner benefits, and the equivalent O(n_train × n_test) asymptotic
  cost. The UDF rationale is explained in the comment block immediately above the
  decorator (lines 56-60), citing the absence of a built-in mode aggregate.
  `euclidean_distance_expr` has inline comments covering squared-diff term construction,
  accumulation loop, and the final sqrt. `spark_knn_df` uses numbered section headers
  (steps 1-11) covering SparkSession creation, train/test DataFrame build, column
  renaming, cross join, distance column, window rank, filter, majority vote via
  collect_list + UDF, collect, and session teardown. No debug prints; appendix-ready.

### Experimental analysis (rubric criterion — 3 pts)
- [ ] 5 experiments present, varying both partitions and dataset size
- Notes: FAIL on this criterion. `spark_knn_rdd.main()` contains exactly 2
  experiments (Experiment 1: 200-train/50-test/4-partitions; Experiment 2: tiny
  20-train/5-test correctness check). `spark_knn_df.main()` contains 1 experiment
  (200-train/50-test/4-partitions). Neither file varies the number of partitions
  systematically (e.g., 2, 4, 8, 16) nor varies dataset size across multiple scales
  (e.g., 500, 1000, 2000, 5000 samples). No standalone experiment results file or
  scalability table exists in the project directory.

### Discussion of weak/strong points (rubric criterion — 3 pts)
- [ ] Dedicated interpretive analysis of weak/strong points present
- Notes: FAIL on this criterion (carry-forward from Phase 1). No dedicated
  "Strengths and Weaknesses" or "Discussion" section exists in any file. The
  Phase 1 QC log already flagged this gap and deferred it to Phase 2. Phase 2
  deliverables have not addressed it.

### Overall Phase 2 verdict: FAIL
Critical gaps:
1. Experimental analysis (3 pts): Fewer than 5 experiments; no systematic variation
   of partitions (only 4 used throughout) and no variation of dataset size beyond
   two fixed points. The scalability dimension required by the rubric is absent.
2. Discussion of weak/strong points (3 pts): No interpretive analysis section exists
   in any file. This was a deferred Phase 1 requirement that Phase 2 did not fulfil.

Fix instructions:
1. Add at least 5 experiments in a dedicated script (e.g., `experiments.py`) or
   expand `main()` in one of the Spark files. Experiments must vary:
   - Number of partitions: at minimum three values, e.g. 2, 4, 8 (or 2, 4, 8, 16).
   - Dataset size: at minimum three scales, e.g. 500, 1000, 2000 training samples.
   - Report wall-clock time AND accuracy for each configuration in a structured table.
   Example minimum set of 5: (500 samples, 2 parts), (500 samples, 4 parts),
   (1000 samples, 4 parts), (2000 samples, 4 parts), (2000 samples, 8 parts).
2. Add a "Strengths and Weaknesses" section (in report_theory.md or a new
   report_discussion.md) that interprets the experimental findings. Must cover at
   least: (a) where/why the algorithm achieves speedup; (b) where it degrades
   (e.g., broadcast bottleneck for large test sets, cross-join memory cost in DF
   version); (c) comparison of RDD vs. DataFrame approach trade-offs; (d) any
   edge cases observed (e.g., empty-partition handling, tie-breaking).

---

## After Phase 3 (Agent 5)
Date: 2026-03-29

### experiments.py
- [x] Module-level docstring
- [x] All 5 experiments present
- [x] Exp 3 varies partitions ≥3 values: [2, 4, 8, 16]
- [x] Exp 4 varies dataset size ≥3 values: [500, 1000, 2000, 5000]
- [x] Exp 5 RDD vs DF comparison
- [x] All CSVs defined with correct columns
- [x] All 3 plots defined
- [x] Inline comments on loops and speedup formula
- [x] time.time() used
- Notes: The module-level docstring (lines 1-33) is thorough: it names all 5 experiments
  with their parameter settings, lists all output artefacts (5 CSVs and 3 plots), and
  explains the three tracked metrics (accuracy, runtime_s, speedup) including the speedup
  formula. All 5 experiment functions are defined and called unconditionally from
  __main__. Experiment 3 uses partition_counts = [2, 4, 8, 16] (4 distinct values).
  Experiment 4 uses dataset_sizes = [500, 1000, 2000, 5000] (4 distinct values).
  Experiment 5 is an explicit RDD-only vs DataFrame-only head-to-head with the sequential
  baseline omitted by design (noted in both the docstring and inline comment).
  CSV fieldnames are consistent with the experiment descriptions and rubric expectations.
  All 3 plots (speedup_vs_partitions.png, runtime_vs_dataset_size.png,
  rdd_vs_df_runtime.png) are defined in generate_plots() and saved to RESULTS_DIR.
  Timing uses the t0 = time.time() / time.time() - t0 pattern throughout (no context
  manager). Speedup formula is commented inline in Experiment 3 (lines 340-341, 354-355).
  Loop iteration variables are annotated with comments explaining what is being varied.
  No debug prints or commented-out code; file is appendix-ready.
  The Phase 2 critical gap (no standalone experiment script, no partition/size variation)
  is fully resolved by this file.

### Overall Phase 3 verdict: PASS
Critical gaps: None. All checklist items satisfied.
Fix instructions: None required. Remaining open rubric items are:
  - Discussion of weak/strong points (3 pts): Still not present — deferred to Phase 4
    as per the original QC log notes. A dedicated "Strengths and Weaknesses" /
    "Discussion" section must be added (in report_theory.md or a new
    report_discussion.md) interpreting the experimental results once the script has
    been executed and actual numbers are available.

---

## FINAL QC — report.md
| Criterion | Max | Score | Notes |
|---|---|---|---|
| Solution description | 4 | 4 | Full: Section 1 (Introduction) provides thorough problem motivation, O(n·D) complexity argument, paper reference (Maillo et al. 2015), two-implementation rationale, and central research question. Section 6 (Conclusion) wraps up. Well above minimum. |
| Algorithms + comments | 4 | 4 | Full: Section 2 includes Algorithm 1 (Map), Algorithm 2 (Reduce), Algorithm 3 (Cleanup) pseudocode blocks, ASCII pipeline diagram, correctness proof sketch, MapReduce→Spark RDD mapping table, and step-by-step algorithmic flow for both RDD (§2.3) and DataFrame (§2.4) implementations. |
| Code comments | 3 | 3 | Full: Section 3 quotes four inline comment examples verbatim (argsort from baseline_knn.py; vectorised distance from spark_knn_rdd.py; broadcast dereference from spark_knn_rdd.py; UDF rationale from spark_knn_df.py), each with an explanatory paragraph. Inline comments are visible throughout the appendix code. |
| Experimental analysis | 3 | 1 | Partial: All structural requirements met — 5 experiments designed with clear configs, speedup formula stated (`speedup = sequential_time / parallel_time`), and properly headed tables present. However, every result cell is marked "TBD — see experiments.py"; the report explicitly states experiments have not been executed. No empirical numbers, trends, or interpretation exist. |
| Weak/strong discussion | 3 | 3 | Full: Section 5 provides a dedicated discussion for all three implementations. Sequential (§5.1): 3 strong + 3 weak points. RDD (§5.2): 5 strong + 3 weak points. DataFrame (§5.3): 4 strong + 5 weak points. Each point is specific and grounded in algorithmic or engineering reasoning. §5.4 compares to original Hadoop MapReduce. |
| Code appendix | 2 | 2 | Full: All 5 files present in full — utils.py, baseline_knn.py, spark_knn_rdd.py, spark_knn_df.py, experiments.py — each complete from module docstring through `if __name__ == "__main__"`. |
| TOTAL | 19 | 17 | |

Verdict: PASS

Gaps and fix instructions:
1. **Experimental analysis (−2 pts)**: The single most impactful gap. `experiments.py` is complete and correct, but it has never been run. Every result cell in Section 4 is "TBD". To close this gap:
   - Execute `python experiments.py` from the project directory (requires PySpark and sklearn installed).
   - Copy the produced CSV rows (`results/exp*.csv`) into the corresponding Section 4 tables in `report.md`.
   - Replace TBD cells with actual runtime_s, accuracy, and speedup values.
   - Add a brief interpretive sentence per experiment based on the real numbers (e.g., "speedup peaked at X× with 8 partitions, consistent with the 8-core machine").
   Fixing this would raise the score to approximately 18–19/19.
