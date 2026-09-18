# MediRAG Evaluation Results

This directory organizes all evaluation outputs by operational lifecycle and provenance.

---

## Directory Organization

```text
results/
├── current/       # Active system baseline artifacts (transitional, pre-modernization)
├── historical/    # Completed architectural experiments (Day 2/Day 3)
└── legacy/        # Stale/legacy CSV exports from earlier manual analysis
```

---

## 1. `results/current/` (Active Baseline Run)
These artifacts represent the current baseline pipeline before the Phase 5–8 modernization:
- `generation_outputs.json`: Raw model generations and retrieved context for 20 queries (lossless 2,200c evidence contract).
- `auto_metrics_results.json`: Automated lexical and structural proxy metrics.
- `llm_judge_results.json`: Baseline LLM-as-judge ratings (`llama-3.3-70b-versatile`).
- `manual_grades.json`: Baseline manual clinical accuracy grades (0/1/2).
- `generation_eval_report.json`: Consolidated evaluation report.

> [!NOTE]
> These results are transitional baseline outputs. A fresh, fully versioned final evaluation will be executed in Phase 9 after all modernization steps are complete.

---

## 2. `results/historical/` (Architectural Experiments)
Preserved outputs from specific retrieval optimization experiments:
- `retrieval_comparison_results.json`: Day 2 chunk size comparison (450/64 vs 800/150).
- `retrieval_medical_tokenizer_results.json`: Day 3 medical BM25 tokenizer benchmark.
- `retrieval_candidate_depth_results.json`: Day 3 RRF candidate depth sweep (20 vs 50 vs 100).

---

## 3. `results/legacy/` (Static CSV Exports)
Static exports from early project milestones:
- `medirag_eval_full.csv`: Early per-query retrieval log.
- `medirag_eval_summary.csv`: Early high-level retrieval summary.
- `medirag_generation_eval.csv`: Early generation evaluation export.

These CSVs are preserved for historical audit trails but are not consumed by active runners.
