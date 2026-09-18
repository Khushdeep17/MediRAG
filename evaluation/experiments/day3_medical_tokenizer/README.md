# Historical Experiment: Medical-Aware BM25 Tokenizer

## Overview
- **Experiment Date:** Day 3
- **Objective:** Evaluate whether a domain-specific medical tokenizer with medical synonym normalization, clinical acronym handling, and hyphenated compound preservation improves BM25 sparse retrieval and hybrid RRF retrieval.
- **Corpus Preset:** Baseline 800/150 (4,239 chunks)
- **Script:** `benchmark_medical_tokenizer.py`
- **Artifact:** `retrieval_medical_tokenizer_results.json`

## Benchmark Results (50 Queries)
- **Sparse BM25:**
  - MRR: `0.695` → `0.708` (+0.013)
  - NDCG@10: `0.754` → `0.763` (+0.009)
  - Recall@5: `0.840` → `0.840`
  - Recall@10: `0.940` → `0.940`
  - Query movement: 3 improved, 1 regressed, 46 tied
- **Hybrid Fusion (Dense + Sparse, α=0.7, RRF k=60):**
  - Identical across 50/50 queries (Dense dominance at α=0.7 masked sparse changes).

## Conclusion
- Positive isolated effect for sparse retrieval.
- Zero regression on hybrid retrieval.
- Retained baseline tokenizer as default; medical tokenizer is available as an opt-in experimental feature via `retrieval/sparse.py`.
