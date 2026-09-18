# MediRAG Evaluation Metrics Subsystem

This directory is designated for modular metric calculators in the modernized evaluation stack.

---

## Target Modern Metric Modules (Planned)

1. **Retrieval Metrics (`metrics/retrieval.py`)**:
   - Recall@K, MRR, NDCG@K, Hit Rate.
   - Chapter-level vs. passage-level relevance scoring.
   - Query movement and tier diagnostic breakdown.

2. **Claim-Level Grounding (`metrics/groundedness.py`)**:
   - Atomic claim extraction from generated medical text.
   - NLI verification against cited and retrieved context chunks (Supported / Unsupported / Contradicted).

3. **Citation Quality (`metrics/citations.py`)**:
   - Citation validity (syntax and 1-indexed bounds).
   - Citation entailment (does the cited chunk actually support the specific claim).
   - Citation completeness (are medical claims missing required citations).

4. **Lexical / Utility Baselines (`metrics/lexical.py`)**:
   - Context utilization and overlap ratios.
