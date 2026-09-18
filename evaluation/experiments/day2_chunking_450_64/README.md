# Historical Experiment: Chunk Size Comparison (450/64 vs 800/150)

## Overview
- **Experiment Date:** Day 2
- **Objective:** Evaluate whether a smaller chunk size (450 characters, 64 chunk overlap) improves retrieval granularity over the baseline (800 characters, 150 chunk overlap).
- **Corpus Presets Tested:**
  - `baseline`: 800 char chunk / 150 char overlap (4,239 chunks)
  - `experiment_450_64`: 450 char chunk / 64 char overlap (8,300+ chunks)
- **Artifact:** `retrieval_comparison_results.json`

## Findings & Conclusions
- The 450/64 chunk configuration showed small improvements in pure dense MRR on some queries, but increased fragmentation across multi-concept queries (Tier 3).
- Overall hybrid performance remained strongest on the baseline 800/150 chunk preset.
- Decision: Baseline 800/150 was retained as the default production configuration; 450/64 remains available as an experimental preset.
