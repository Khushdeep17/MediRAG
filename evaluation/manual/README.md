# MediRAG Manual Clinical Evaluation Subsystem

This directory provides tools and guidelines for human clinician grading and expert review.

---

## Evaluation Guidelines

- **Clinical Accuracy Scale**:
  - `0`: Medically inaccurate, contradictory to clinical guidelines, or hazardous.
  - `1`: Partially accurate; omits essential clinical caveats or key therapy lines.
  - `2`: Fully accurate, clinically sound, and appropriately contextualized.
- **Citation Spot-Checks**:
  - Verification that flagged claims match the corresponding cited textbook passage.

---

## Tooling
- CLI inspection utilities and grading interfaces.
- Inter-annotator agreement tools (Cohen's Kappa / Fleiss' Kappa) when multi-expert annotations are integrated.
