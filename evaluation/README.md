# MediRAG Evaluation Subsystem

This directory contains the benchmarking, dataset management, and evaluation infrastructure for MediRAG.

---

## Evaluation Architecture & Subsystem Layout

```text
evaluation/
├── README.md                           # This document
│
├── datasets/                           # Versioned ground-truth query sets
│   ├── README.md                       # Dataset specifications and annotations
│   ├── retrieval_queries_v1.json       # 50 frozen retrieval benchmark queries
│   └── generation_queries_v1.json      # 20 frozen generation benchmark queries
│
├── runners/                            # Standardized CLI benchmark entry points
│   ├── README.md                       # Runner usage and invocation parameters
│   ├── run_retrieval_benchmark.py      # Thin runner for dense/sparse/hybrid retrieval
│   └── run_generation_benchmark.py     # Thin runner for generation and evidence logging
│
├── results/                            # Benchmark execution outputs & artifacts
│   ├── README.md                       # Results organization and provenances
│   ├── current/                        # Active baseline run artifacts (transitional)
│   ├── historical/                     # Completed architectural experiment outputs
│   └── legacy/                         # Early static CSV exports (audit trail)
│
├── experiments/                        # Archived historical architectural studies
│   ├── day2_chunking_450_64/           # Chunk size comparison (450/64 vs 800/150)
│   ├── day3_medical_tokenizer/         # Medical-aware BM25 tokenizer benchmark
│   └── day3_candidate_depth/           # RRF candidate pool depth sweep (20/50/100)
│
├── notebooks/                          # Interactive analysis & visualization dashboards
│   ├── README.md                       # Notebook documentation & technical debt note
│   ├── retrieval_dashboard.ipynb       # Retrieval metric plots & alpha sweeps
│   └── generation_dashboard.ipynb      # Generation quality & judge score distributions
│
├── metrics/                            # Planned: Modular metric implementations
│   └── README.md                       # Target metrics (retrieval, NLI, citations)
│
├── judges/                             # Planned: Standardized LLM-as-judge engines
│   └── README.md                       # Judge architecture, rubrics, & schemas
│
├── manual/                             # Human clinician review & grading guidelines
│   └── README.md                       # Clinical accuracy rubric (0/1/2) & protocols
│
├── reports/                            # Consolidated evaluation reports & scorecards
│   └── README.md                       # Formatted benchmark reports
│
├── auto_metrics.py                     # Current automated metric calculator
├── generation_eval.py                  # Core generation evaluation engine
├── llm_judge.py                        # Current LLM-as-judge module
├── manual_grades.py                    # Current manual grading & report assembler
└── retrieval_metrics.py                # Core retrieval evaluation engine
```

---

## Evaluation Purpose & Scope

The MediRAG evaluation subsystem assesses two distinct phases of the clinical question-answering pipeline:

1. **Retrieval Performance (Modernized Passage-Level & Historical Baselines)**:
   - Primary ground truth: `evaluation/datasets/retrieval_queries_v2.json` (50 queries, passage-level graded chunk relevance 0/1/2, annotated exact evidence spans).
   - **Graded Chunk NDCG@5 & NDCG@10**: Evaluates passage ranking quality using graded relevance ($0 = \text{non-relevant}$, $1 = \text{partially relevant}$, $2 = \text{directly relevant}$) with standard exponential gain $2^{\text{rel}} - 1$ and logarithmic position discount $\log_2(\text{rank} + 1)$ normalized by IDCG@K.
   - **Grade-2 Chunk MRR**: Evaluates the reciprocal rank ($1/\text{rank}$) of the first retrieved chunk containing direct, core clinical relevance (Grade 2). Returns $0.0$ if no Grade-2 chunk is retrieved.
   - **Evidence Recall@K (K=5, K=10)**: Measures the proportion of annotated evidence-bearing chunks retrieved in top-K: $|S_K \cap E| / |E|$, accompanied by **Evidence Hit@K**.
   - **Legacy Chapter Recall@5, Recall@10, and Chapter MRR**: Maintained for direct historical comparison with v1 benchmarks.
   - Evaluated across three difficulty tiers: Direct (Tier 1, n=17), Indirect Clinical Framing (Tier 2, n=18), and Complex Multi-Concept (Tier 3, n=15).

2. **Generation Quality & Grounding**:
   - Evaluates whether generated clinical summaries are faithful to retrieved evidence and properly cited.
   - Evaluates adherence to 1-indexed citation syntax `[N]`, citation attribution fidelity, and answer relevance.

---

## Evaluation Datasets

All benchmark queries are externalized and versioned under `evaluation/datasets/`:
- `retrieval_queries_v2.json`: Primary ground truth. 50 clinical evaluation queries with chapter sets, passage-level graded chunk relevance (0/1/2), and exact evidence spans.
- `retrieval_queries_v1.json`: Frozen legacy retrieval benchmark dataset (50 queries, chapter-level labels only).
- `generation_queries_v2.json`: Primary generation ground truth. 20 clinical queries with reference answers, expected chapters, clinical answer aspects, and safety-critical notes.
- `generation_queries_v1.json`: Frozen legacy generation benchmark dataset (20 queries, chapter-level labels only).

---

## Results Classification

To prevent confusion between historical runs and current baselines:
- **`results/current/`**: Houses the active baseline outputs (`generation_outputs.json`, `auto_metrics_results.json`, `generation_eval_report.json`, `llm_judge_results.json`, `manual_grades.json`). These reflect the transitional baseline before modern metric implementation.
- **`results/historical/`**: Houses frozen artifacts from Day 2 and Day 3 retrieval optimization experiments.
- **`results/legacy/`**: Houses static CSV exports preserved strictly for historical reference.
- **Future Final Results**: A single, fresh, end-to-end benchmark will be executed and versioned in `results/final/` after the complete modernization roadmap is finished.

---

## Methodology Status: Transitional

> [!IMPORTANT]
> The current generation metrics (`auto_metrics.py`) include legacy heuristics such as `structural_grounded_rate`, `citation_accuracy` (chapter presence), and `hallucination_rate` (`1 - structural_grounded_rate`).
> 
> As established during the Phase 3 audit, these heuristics are **transitional proxies**, not clinical ground truth. In upcoming phases:
> - Groundedness will be upgraded to **atomic claim extraction + NLI entailment**.
> - Citation evaluation will measure **claim-to-evidence support**.
> - LLM-as-judge will use **Pydantic structured output** and calibrated clinical rubrics.

---

## Reproducibility Standards

Future final benchmark runs will enforce full provenance logging:
- Exact Git commit SHA.
- Dataset version (`v1.0`).
- Corpus chunk preset (`baseline` 800/150).
- Embedding model (`BAAI/bge-large-en-v1.5`) and sparse tokenizer configuration.
- Generation model name, temperature, and prompt template version.
- Judge model name and structured rubric schema.
