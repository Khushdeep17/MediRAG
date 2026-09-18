# Historical Experiment: RRF Candidate Pool Depth (20 vs 50 vs 100)

## Overview
- **Experiment Date:** Day 3
- **Objective:** Evaluate whether deepening the dense and sparse candidate pools retrieved prior to Reciprocal Rank Fusion (RRF) allows high-value sparse candidates to surface into the final top-10 hybrid ranking.
- **Corpus Preset:** Baseline 800/150 (4,239 chunks)
- **Script:** `benchmark_candidate_depth.py`
- **Artifact:** `retrieval_candidate_depth_results.json`
- **Depths Tested:**
  - `depth_20`: `top_k * 2 = 20` candidates per retriever
  - `depth_50`: 50 candidates per retriever
  - `depth_100`: 100 candidates per retriever

## Benchmark Results (50 Queries)
- **Hybrid MRR:**
  - Depth 20: `0.883`
  - Depth 50: `0.871` (-0.012)
  - Depth 100: `0.865` (-0.018)
- **Diagnostic Findings:**
  - At all tested depths (20, 50, 100), 0 sparse-only candidates broke into the final top-10 hybrid results.
  - Deeper candidate pools introduced lower-ranking dense noise into RRF, leading to slight rank dilution on 2 Tier-3 queries.

## Conclusion
- Default candidate depth remains `20` (`candidate_depth=None` dynamically defaults to `top_k * 2`).
- `candidate_depth` parameter in `retrieval/fusion.py` was made explicitly configurable for future tuning.
