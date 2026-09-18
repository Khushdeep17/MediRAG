# MediRAG Evaluation Notebooks

This directory contains interactive analysis and visualization dashboards.

---

## Notebooks

- `retrieval_dashboard.ipynb`: Visualizes retrieval benchmark metrics (Recall@5, Recall@10, MRR, NDCG@10, tier comparisons, and alpha sweeps).
- `generation_dashboard.ipynb`: Visualizes generation evaluation metrics (groundedness, citation accuracy, LLM-as-judge distributions, and manual grading correlations).

---

## Technical Debt Note

> [!WARNING]
> **Duplicated Benchmark Logic**
> Currently, both notebooks contain embedded routines that can trigger benchmark runs directly. In the modernized architecture:
> 1. Benchmark execution is strictly delegated to CLI runners in `evaluation/runners/`.
> 2. Notebooks should serve purely as **visualization and post-hoc analysis tools**, ingesting JSON artifacts produced by runners.
> 
> Refactoring these notebooks to strictly consume runner artifacts will be addressed in future phases.
