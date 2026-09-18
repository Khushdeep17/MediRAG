# MediRAG Evaluation Runners

This directory houses the standardized command-line entry points for executing retrieval and generation benchmarks.

---

## Runners Overview

- `run_retrieval_benchmark.py`: Executes retrieval evaluation across dense, sparse, and hybrid configurations against versioned query sets (`datasets/retrieval_queries_v1.json`).
- `run_generation_benchmark.py`: Executes generation evaluation (retrieval + synthesis) against versioned query sets (`datasets/generation_queries_v1.json`).

---

## Design Principles
1. **Separation of Concerns**: Runners orchestrate data loading, batch execution, and result serialization without embedding hardcoded queries or metric implementations.
2. **Deterministic Inputs**: All evaluation queries are ingested from `evaluation/datasets/`.
3. **Structured Outputs**: Results are saved to versioned JSON paths under `evaluation/results/`.
