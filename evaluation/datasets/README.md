# MediRAG Evaluation Datasets

This directory contains the frozen, versioned query sets and ground-truth specifications used for evaluating the MediRAG retrieval and generation pipelines.

---

## Datasets Overview

| Dataset | File | Version | Query Count | Difficulty Tiers | Relevance Granularity | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Retrieval Benchmark (v1)** | `retrieval_queries_v1.json` | `1.0` | 50 | Tier 1 (17), Tier 2 (18), Tier 3 (15) | Chapter-level (single int) | **Active Frozen Baseline** |
| **Generation Benchmark (v1)** | `generation_queries_v1.json` | `1.0` | 20 | Tier 1 (7), Tier 2 (7), Tier 3 (6) | Chapter-level (single int) | **Active Frozen Baseline** |
| **Retrieval Benchmark (v2)** | `retrieval_queries_v2.json` | `2.0` | 50 | Tier 1 (17), Tier 2 (18), Tier 3 (15) | Passage-level graded (0/1/2) + Exact evidence spans | **Validated & Frozen (v2.0)** |
| **Generation Benchmark (v2)** | `generation_queries_v2.json` | `2.0` | 20 | Tier 1 (7), Tier 2 (7), Tier 3 (6) | Clinical answer aspects + Safety notes | **Validated & Frozen (v2.0)** |

---

## Evolution Stages: Specification vs Annotation vs Frozen Dataset

To maintain rigorous experimental integrity, MediRAG distinguishes three dataset phases:
1. **Design Specification (Phase 5)**: Established schema contracts, graded relevance definitions (0/1/2), evidence span representation rules, and clinical answer-aspect checklists with Query 1 demonstration.
2. **Human-Authored Annotation (Phase 5B)**: Exhaustively completed human-authored project annotations for all 50 retrieval queries and all 20 generation queries directly grounded in the Merck Manual (800/150) corpus.
3. **Validated & Frozen Dataset (Phase 5B Completion)**: Cryptographically hashed (SHA-256), programmatically validated with 0 errors via `validate_dataset.py`, verified for zero benchmark query leakage in executable code, and frozen.

> **Terminology Note**: All v2 annotations represent *human-authored project annotations* grounded verbatim in the Merck Manual text. They are not claimed to be formal multi-center clinician-validated consensus guidelines.

---

## Difficulty Tiers

Both datasets categorize clinical queries into three difficulty levels:

1. **Tier 1 — Direct Queries** (17 retrieval / 7 generation):
   - Explicit condition name in question (e.g., *"What are the causes and treatment of migraine?"*).
   - High lexical overlap with standard textbook chapter and section headings.
2. **Tier 2 — Indirect / Clinical Framing** (18 retrieval / 7 generation):
   - Clinical vignette or symptom-based presentation without explicit condition name (e.g., *"A patient presents with recurring severe headaches with nausea and light sensitivity. What is the diagnosis and management?"*).
   - Requires semantic clinical inference across diagnostic criteria.
3. **Tier 3 — Hard / Multi-Concept / Ambiguous** (15 retrieval / 6 generation):
   - Cross-system mechanisms, drug interactions, multi-organ pathophysiology, or subtle differential diagnoses (e.g., *"What mechanisms explain why certain medications worsen thyroid function?"*).
   - Requires multi-chunk synthesis and cross-chapter evidence gathering.

---

## Retrieval Dataset v2 Architecture

### 1. Graded Relevance Scale (0 / 1 / 2)
Relevance is graded strictly against actual chunk content:
- **Grade 0 (Irrelevant / Non-informative)**: Chunk is from an adjacent or unrelated section and does not answer the clinical prompt (negative control).
- **Grade 1 (Related / Contextual)**: Chunk provides valuable background, physiology, general disease classification, or diagnostic mimics without definitive therapeutic/pathological answers.
- **Grade 2 (Directly Relevant / Definitive)**: Chunk provides direct, factual, and actionable answers to the query's core clinical questions.

### 2. Exact Evidence Anchor Policy
To avoid brittle character offsets across retokenizations or whitespace normalizations:
- **`chunk_id + text` is authoritative**.
- Every evidence quote MUST be an exact literal substring of the corresponding chunk in `data/processed/merck_chunks_800_150.json`.
- Paraphrasing or fabricating evidence quotes is strictly prohibited.
- `approx_char_start` and `approx_char_end` are optional convenience offsets for UI rendering, never the source of truth.

### 3. Cross-Chapter Scope Policy
Real-world clinical queries often span multiple organ systems. Unlike v1 (which artificially forced every query into a single integer chapter), v2 represents multi-chapter scope when supported by evidence:
- 14 of 50 retrieval queries have explicit cross-chapter coverage in `expected_chapters` (e.g., Q36 Amiodarone thyroid toxicity spans Ch. 93 [Thyroid] and Ch. 213 [Arrhythmias]; Q40 ACE inhibitors in renal disease spans Ch. 239 [CKD], Ch. 99 [Diabetes], and Ch. 208 [Hypertension]).
- All Grade-2 chunks must belong to a chapter listed in `expected_chapters`.

---

## Generation Dataset v2 Architecture

### 1. Answer Aspects Checklist (No Fabricated Reference Answers)
Rather than writing subjective, single-prose reference answers (which penalize legitimate stylistic variations and create artificial bias in evaluation), v2 utilizes structured **Key Clinical Answer Aspects**:
- Each query has 3–6 clinically meaningful, factual aspects describing concrete concepts a complete answer must cover.
- Each aspect is categorized by criticality:
  - `essential`: Mandatory core clinical facts required for minimum competence.
  - `recommended`: Important clinical context, diagnostic nuance, or secondary management.
  - `safety_critical`: Critical contraindications, high-risk drug interactions, or red flag conditions where omission or misstatement could lead to patient harm.

### 2. Safety Notes Provenance
- All safety notes are directly grounded in the Merck Manual corpus text.
- No external unverified clinical guidelines or fabricated claims are included.

### 3. Reference Answer Policy
- `reference_answer: null` is strictly maintained across all 20 generation queries.
- No synthetic or LLM-generated prose answers are stored as gold ground truth.

---

## Benchmark Leakage Policy and Cleanup

To ensure complete evaluation integrity:
- **Zero executable leakage**: Benchmark query strings must never appear as test queries, default parameters, placeholders, or interactive examples in production or development code (`retrieval/dense.py`, `retrieval/fusion.py`, `generate.py`, `app.py`).
- Identified historical leakage was removed in Phase 5B and replaced with non-benchmark medical examples:
  - *"What are the manifestations of acute pancreatitis?"*
  - *"How is celiac disease diagnosed?"*
  - *"What are the clinical signs of acute appendicitis?"*
- Verification scans confirmed **0 occurrences of benchmark queries in executable application code**.

---

## Dataset Freeze Manifest

| Parameter | Retrieval v2 | Generation v2 |
| :--- | :--- | :--- |
| **Filename** | `retrieval_queries_v2.json` | `generation_queries_v2.json` |
| **Version** | `2.0` | `2.0` |
| **Query Count** | 50 (T1: 17, T2: 18, T3: 15) | 20 (T1: 7, T2: 7, T3: 6) |
| **Corpus File** | `merck_chunks_800_150.json` (4,239 chunks) | `merck_chunks_800_150.json` (4,239 chunks) |
| **Annotation Status** | Complete (Human-authored project annotation) | Complete (Human-authored project annotation) |
| **Validation Status** | Passed (0 errors) | Passed (0 errors) |
| **Completion Date** | 2026-09-17 | 2026-09-17 |
| **SHA-256 Hash** | `6368095e15cbd90aec3e161e5a8876f965c16d24c6a395610b2a1176a360f193` | `91618a57749f498abb025e3a6d383b640bc0227d0ae0ce142df32e699dd20f7b` |

---

## Automated Dataset Validation

To verify dataset integrity at any time, run:

```powershell
python evaluation/datasets/validate_dataset.py
```

The validator verifies:
1. Top-level schema compliance and exact query counts (50 and 20).
2. Query ID, query text, and tier match v1 frozen baseline exactly.
3. Every cited chunk ID exists in `data/processed/merck_chunks_800_150.json`.
4. Relevance scores are strictly within `{0, 1, 2}`.
5. All Grade 2 chunks have their chapter included in `expected_chapters`.
6. Every evidence span is an exact verbatim substring of its corpus chunk.
7. Clinical answer aspects have valid non-empty descriptions and allowed criticalities (`essential`, `recommended`, `safety_critical`).
