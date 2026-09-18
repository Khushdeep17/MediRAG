import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# --- Fix import path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from retrieval.dense import dense_search
from retrieval.sparse import sparse_search
from retrieval.fusion import hybrid_search

# =====================================================
# CONFIG
# =====================================================

RETRIEVAL_K = 30
DEBUG       = True   # Set False for clean summary-only output

# =====================================================
# EVAL QUERIES — 50 total, 3 difficulty tiers
#
# TIER 1 — Direct (17 queries): topic name in question, easy match
# TIER 2 — Indirect (18 queries): clinical framing, no topic name
# TIER 3 — Hard (15 queries): multi-concept, ambiguous, edge-case
#
# Chapter labels verified against actual book index.
# =====================================================

_FALLBACK_EVAL_QUERIES = [

    # ── TIER 1: Direct Queries ──────────────────────────────────────────
    {"query": "What are the causes and treatment of migraine?",                    "relevant_chapter": 178, "tier": 1},
    {"query": "What are the symptoms of Parkinson disease?",                       "relevant_chapter": 183, "tier": 1},
    {"query": "How is epilepsy diagnosed and managed?",                            "relevant_chapter": 176, "tier": 1},
    {"query": "What causes multiple sclerosis?",                                   "relevant_chapter": 184, "tier": 1},
    {"query": "How is asthma treated?",                                            "relevant_chapter": 191, "tier": 1},
    {"query": "What are the causes and management of COPD?",                       "relevant_chapter": 192, "tier": 1},
    {"query": "What causes iron deficiency anemia?",                               "relevant_chapter": 105, "tier": 1},
    {"query": "How is diabetes mellitus managed?",                                 "relevant_chapter":  99, "tier": 1},
    {"query": "What are the causes and symptoms of hypothyroidism?",               "relevant_chapter":  93, "tier": 1},
    {"query": "What causes Cushing syndrome?",                                     "relevant_chapter":  94, "tier": 1},
    {"query": "What are the risk factors and treatment of hypertension?",          "relevant_chapter": 208, "tier": 1},
    {"query": "How is heart failure classified and managed?",                      "relevant_chapter": 211, "tier": 1},
    {"query": "What causes peptic ulcer disease and how is it treated?",           "relevant_chapter":  13, "tier": 1},
    {"query": "What are the symptoms and management of Crohn disease?",            "relevant_chapter":  19, "tier": 1},
    {"query": "How is pneumonia diagnosed and treated?",                           "relevant_chapter": 196, "tier": 1},
    {"query": "What is the treatment for sickle cell disease?",                    "relevant_chapter": 106, "tier": 1},
    {"query": "How is leukemia classified and treated?",                           "relevant_chapter": 117, "tier": 1},

    # ── TIER 2: Indirect / Clinical Framing ─────────────────────────────
    {"query": "A patient presents with recurring severe headaches with nausea and light sensitivity. What is the diagnosis and management?",
                                                                                   "relevant_chapter": 178, "tier": 2},
    {"query": "What neurological condition causes resting tremor, rigidity, and bradykinesia?",
                                                                                   "relevant_chapter": 183, "tier": 2},
    {"query": "How do you manage uncontrolled seizures in an adult patient?",      "relevant_chapter": 176, "tier": 2},
    {"query": "What is the pathophysiology behind demyelination in the CNS?",      "relevant_chapter": 184, "tier": 2},
    {"query": "Patient with focal neurological deficits of sudden onset — what is the workup?",
                                                                                   "relevant_chapter": 173, "tier": 2},
    {"query": "How is reversible airflow obstruction treated in adults?",          "relevant_chapter": 191, "tier": 2},
    {"query": "What long-term complications arise from progressive airflow limitation in smokers?",
                                                                                   "relevant_chapter": 192, "tier": 2},
    {"query": "How is acid-fast bacilli infection of the lungs managed?",          "relevant_chapter": 141, "tier": 2},
    {"query": "A patient has low hemoglobin and low ferritin — what are the causes and treatment?",
                                                                                   "relevant_chapter": 105, "tier": 2},
    {"query": "How is hyperglycemia managed in a patient with insulin resistance?", "relevant_chapter": 99, "tier": 2},
    {"query": "What clinical signs suggest an underactive thyroid and how is it corrected?",
                                                                                   "relevant_chapter":  93, "tier": 2},
    {"query": "How is cortisol excess from an adrenal or pituitary source differentiated and treated?",
                                                                                   "relevant_chapter":  94, "tier": 2},
    {"query": "What lifestyle and pharmacological interventions reduce blood pressure?",
                                                                                   "relevant_chapter": 208, "tier": 2},
    {"query": "How is reduced cardiac ejection fraction diagnosed and compensated?",
                                                                                   "relevant_chapter": 211, "tier": 2},
    {"query": "What are the first and second line therapies for H. pylori related gastric disease?",
                                                                                   "relevant_chapter":  13, "tier": 2},
    {"query": "How is transmural intestinal inflammation distinguished from ulcerative disease?",
                                                                                   "relevant_chapter":  19, "tier": 2},
    {"query": "How is community-acquired lower respiratory tract infection evaluated and treated?",
                                                                                   "relevant_chapter": 196, "tier": 2},
    {"query": "What causes vaso-occlusive crises in hemoglobin disorders?",        "relevant_chapter": 106, "tier": 2},

    # ── TIER 3: Hard / Multi-concept / Ambiguous ────────────────────────
    {"query": "What mechanisms explain why certain medications worsen thyroid function?",
                                                                                   "relevant_chapter":  93, "tier": 3},
    {"query": "How does electrolyte imbalance affect cardiac rhythm?",             "relevant_chapter":  97, "tier": 3},
    {"query": "What are the neurological consequences of untreated vitamin B12 deficiency?",
                                                                                   "relevant_chapter":   4, "tier": 3},
    {"query": "How does portal hypertension develop in patients with liver fibrosis?",
                                                                                   "relevant_chapter":  27, "tier": 3},
    {"query": "What is the role of ACE inhibitors in slowing renal disease progression?",
                                                                                   "relevant_chapter": 239, "tier": 3},
    {"query": "How does obstructive sleep apnea relate to cardiovascular morbidity?",
                                                                                   "relevant_chapter": 193, "tier": 3},
    {"query": "What clotting cascade abnormalities lead to bleeding in liver disease?",
                                                                                   "relevant_chapter": 111, "tier": 3},
    {"query": "How is sepsis distinguished from systemic inflammatory response and how is it managed?",
                                                                                   "relevant_chapter": 227, "tier": 3},
    {"query": "What are the pulmonary manifestations of autoimmune connective tissue disorders?",
                                                                                   "relevant_chapter":  33, "tier": 3},
    {"query": "How does chronic kidney disease affect erythropoiesis and bone metabolism?",
                                                                                   "relevant_chapter": 239, "tier": 3},
    {"query": "What distinguishes angina pectoris from acute myocardial ischemia on presentation?",
                                                                                   "relevant_chapter": 210, "tier": 3},
    {"query": "How is adrenal insufficiency differentiated from primary and secondary causes?",
                                                                                   "relevant_chapter":  94, "tier": 3},
    {"query": "What are the metabolic consequences of long-term corticosteroid therapy?",
                                                                                   "relevant_chapter":  94, "tier": 3},
    {"query": "How is venous thromboembolism risk stratified and prophylactically managed?",
                                                                                   "relevant_chapter": 110, "tier": 3},
    {"query": "What is the mechanism behind refeeding syndrome in malnourished patients?",
                                                                                   "relevant_chapter":   3, "tier": 3},
]

DEFAULT_DATASET_V2 = PROJECT_ROOT / "evaluation" / "datasets" / "retrieval_queries_v2.json"
DEFAULT_DATASET_V1 = PROJECT_ROOT / "evaluation" / "datasets" / "retrieval_queries_v1.json"
DEFAULT_DATASET    = DEFAULT_DATASET_V2


def load_retrieval_dataset(dataset_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load full retrieval evaluation dataset dictionary including metadata."""
    target_path = dataset_path or DEFAULT_DATASET
    if not target_path.exists() and target_path == DEFAULT_DATASET_V2:
        target_path = DEFAULT_DATASET_V1
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load dataset from {target_path}: {e}. Falling back to embedded list.")
    return {
        "dataset_name": "MediRAG Fallback Queries",
        "dataset_version": "1.0-fallback",
        "queries": _FALLBACK_EVAL_QUERIES,
    }


def load_retrieval_queries(dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load retrieval evaluation queries list from versioned JSON dataset."""
    data = load_retrieval_dataset(dataset_path)
    return data.get("queries", [])


EVAL_QUERIES = load_retrieval_queries()


# =====================================================
# METRICS — MODERN PASSAGE-LEVEL & LEGACY CHAPTER-LEVEL
# =====================================================

def chapter_recall_at_k(results: list, expected_chapters: Any, k: int) -> int:
    """Legacy Chapter Recall@K: 1 if any expected chapter appears in top-K."""
    if isinstance(expected_chapters, int):
        expected_set = {expected_chapters}
    elif isinstance(expected_chapters, (list, set, tuple)):
        expected_set = {int(c) for c in expected_chapters if c is not None}
    else:
        expected_set = set()
    top_k_chapters = {r.get("chapter_number") for r in results[:k] if "chapter_number" in r}
    return int(bool(top_k_chapters & expected_set))


def chapter_mrr(results: list, expected_chapters: Any) -> float:
    """Legacy Chapter MRR: reciprocal rank of first expected chapter."""
    if isinstance(expected_chapters, int):
        expected_set = {expected_chapters}
    elif isinstance(expected_chapters, (list, set, tuple)):
        expected_set = {int(c) for c in expected_chapters if c is not None}
    else:
        expected_set = set()
    seen = set()
    for rank, r in enumerate(results, start=1):
        chap = r.get("chapter_number")
        if chap in seen:
            continue
        seen.add(chap)
        if chap in expected_set:
            return 1.0 / rank
    return 0.0


def chapter_ndcg_at_k(results: list, expected_chapters: Any, k: int) -> float:
    """Legacy Chapter NDCG@K: binary discounted cumulative gain for chapter match."""
    if isinstance(expected_chapters, int):
        expected_set = {expected_chapters}
    elif isinstance(expected_chapters, (list, set, tuple)):
        expected_set = {int(c) for c in expected_chapters if c is not None}
    else:
        expected_set = set()
    seen = set()
    pos = 0
    for r in results[:k]:
        chap = r.get("chapter_number")
        if chap in seen:
            continue
        seen.add(chap)
        pos += 1
        if chap in expected_set:
            return float(1.0 / np.log2(pos + 1))
    return 0.0


def graded_chunk_ndcg_at_k(
    results: list,
    relevant_chunks: List[Dict[str, Any]],
    k: int,
) -> float:
    """
    Graded Chunk NDCG@K using relevance scores {0, 1, 2}.
    Gain formulation: 2^rel - 1 (exponential gain).
    Discount: log2(rank + 1) where rank is 1-indexed.
    Normalized by IDCG@K of ideal descending ranking of positive ground-truth chunks.
    """
    if not relevant_chunks:
        return 0.0

    rel_map = {c["chunk_id"]: int(c.get("relevance", 0)) for c in relevant_chunks if "chunk_id" in c}

    dcg = 0.0
    for i, r in enumerate(results[:k], start=1):
        chunk_id = r.get("chunk_id")
        rel = rel_map.get(chunk_id, 0)
        if rel > 0:
            dcg += (2.0**rel - 1.0) / np.log2(i + 1)

    ideal_rels = sorted(
        [int(c.get("relevance", 0)) for c in relevant_chunks if int(c.get("relevance", 0)) > 0],
        reverse=True,
    )
    if not ideal_rels:
        return 0.0

    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:k], start=1):
        idcg += (2.0**rel - 1.0) / np.log2(i + 1)

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def grade2_chunk_mrr_and_rank(
    results: list,
    relevant_chunks: List[Dict[str, Any]],
) -> Tuple[float, Optional[int]]:
    """
    Calculate MRR using the first Grade-2 (directly relevant) retrieved chunk.
    Returns (mrr, first_rank). If no Grade-2 chunk is retrieved, returns (0.0, None).
    """
    if not relevant_chunks:
        return 0.0, None

    grade2_chunk_ids = {
        c["chunk_id"] for c in relevant_chunks
        if int(c.get("relevance", 0)) == 2 and "chunk_id" in c
    }
    if not grade2_chunk_ids:
        return 0.0, None

    for rank, r in enumerate(results, start=1):
        if r.get("chunk_id") in grade2_chunk_ids:
            return float(1.0 / rank), rank

    return 0.0, None


def grade2_chunk_mrr(results: list, relevant_chunks: List[Dict[str, Any]]) -> float:
    """MRR score using the first Grade-2 chunk."""
    mrr, _ = grade2_chunk_mrr_and_rank(results, relevant_chunks)
    return mrr


def evidence_recall_at_k(
    results: list,
    evidence_spans: List[Dict[str, Any]],
    relevant_chunks: Optional[List[Dict[str, Any]]] = None,
    k: int = 10,
) -> float:
    """
    Evidence Recall@K: proportion of annotated evidence-bearing chunks retrieved in top-K.
    """
    ev_chunk_ids = {s["chunk_id"] for s in evidence_spans if s.get("chunk_id")}
    if not ev_chunk_ids and relevant_chunks:
        ev_chunk_ids = {
            c["chunk_id"] for c in relevant_chunks
            if int(c.get("relevance", 0)) == 2 and c.get("chunk_id")
        }
    if not ev_chunk_ids:
        return 0.0

    top_k_chunks = {r.get("chunk_id") for r in results[:k] if r.get("chunk_id")}
    hits = len(top_k_chunks & ev_chunk_ids)
    return float(hits / len(ev_chunk_ids))


def evidence_hit_at_k(
    results: list,
    evidence_spans: List[Dict[str, Any]],
    relevant_chunks: Optional[List[Dict[str, Any]]] = None,
    k: int = 10,
) -> int:
    """
    Evidence Hit@K: 1 if at least one annotated evidence-bearing chunk is in top-K, else 0.
    """
    ev_chunk_ids = {s["chunk_id"] for s in evidence_spans if s.get("chunk_id")}
    if not ev_chunk_ids and relevant_chunks:
        ev_chunk_ids = {
            c["chunk_id"] for c in relevant_chunks
            if int(c.get("relevance", 0)) == 2 and c.get("chunk_id")
        }
    if not ev_chunk_ids:
        return 0

    top_k_chunks = {r.get("chunk_id") for r in results[:k] if r.get("chunk_id")}
    return int(bool(top_k_chunks & ev_chunk_ids))


# Backward-compatibility aliases for existing callers
recall_at_k = chapter_recall_at_k
mrr_score   = chapter_mrr
ndcg_at_k   = chapter_ndcg_at_k


# =====================================================
# EVALUATION LOOP
# =====================================================

def evaluate_system(
    search_fn,
    name: str,
    queries: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = DEBUG,
    k: int = RETRIEVAL_K,
) -> Dict[str, Any]:
    eval_set = queries if queries is not None else EVAL_QUERIES

    # Metrics collectors
    ndcg5_list = []
    ndcg10_list = []
    g2_mrr_list = []
    ev_r5_list = []
    ev_r10_list = []
    ev_hit5_list = []
    ev_hit10_list = []

    ch_r5_list = []
    ch_r10_list = []
    ch_mrr_list = []
    ch_ndcg10_list = []

    per_query_details = []

    # Tier collectors (1: Direct, 2: Indirect, 3: Hard)
    tiers = [1, 2, 3]
    tier_records = {t: [] for t in tiers}

    if verbose:
        print(f"\n{'─' * 70}")
        print(f"  EVALUATING: {name} (n={len(eval_set)} queries)")
        print(f"{'─' * 70}")

    for idx, item in enumerate(eval_set):
        qid = item.get("id", idx + 1)
        query = item["query"]
        tier = item.get("tier", 1)

        # Ground truth structures
        expected_chapters = item.get("expected_chapters")
        if not expected_chapters:
            if "primary_chapter" in item:
                expected_chapters = [item["primary_chapter"]]
            elif "relevant_chapter" in item:
                expected_chapters = [item["relevant_chapter"]]
            else:
                expected_chapters = []

        relevant_chunks = item.get("relevant_chunks", [])
        evidence_spans = item.get("evidence_spans", [])

        # Execute search
        results = search_fn(query, return_results=True)

        # Legacy chapter metrics
        ch_r5 = chapter_recall_at_k(results, expected_chapters, 5)
        ch_r10 = chapter_recall_at_k(results, expected_chapters, 10)
        ch_mrr = chapter_mrr(results, expected_chapters)
        ch_ndcg10 = chapter_ndcg_at_k(results, expected_chapters, 10)

        # Modern passage-level metrics
        if relevant_chunks:
            chunk_ndcg5 = graded_chunk_ndcg_at_k(results, relevant_chunks, 5)
            chunk_ndcg10 = graded_chunk_ndcg_at_k(results, relevant_chunks, 10)
            g2_mrr, g2_rank = grade2_chunk_mrr_and_rank(results, relevant_chunks)
        else:
            chunk_ndcg5 = ch_ndcg10
            chunk_ndcg10 = ch_ndcg10
            g2_mrr = ch_mrr
            g2_rank = None

        if evidence_spans or relevant_chunks:
            ev_r5 = evidence_recall_at_k(results, evidence_spans, relevant_chunks, 5)
            ev_r10 = evidence_recall_at_k(results, evidence_spans, relevant_chunks, 10)
            ev_hit5 = evidence_hit_at_k(results, evidence_spans, relevant_chunks, 5)
            ev_hit10 = evidence_hit_at_k(results, evidence_spans, relevant_chunks, 10)
        else:
            ev_r5 = float(ch_r5)
            ev_r10 = float(ch_r10)
            ev_hit5 = ch_r5
            ev_hit10 = ch_r10

        # Chapter rank
        ch_rank = None
        for rank, r in enumerate(results, start=1):
            if r.get("chapter_number") in expected_chapters:
                ch_rank = rank
                break

        # Record lists
        ndcg5_list.append(chunk_ndcg5)
        ndcg10_list.append(chunk_ndcg10)
        g2_mrr_list.append(g2_mrr)
        ev_r5_list.append(ev_r5)
        ev_r10_list.append(ev_r10)
        ev_hit5_list.append(ev_hit5)
        ev_hit10_list.append(ev_hit10)

        ch_r5_list.append(ch_r5)
        ch_r10_list.append(ch_r10)
        ch_mrr_list.append(ch_mrr)
        ch_ndcg10_list.append(ch_ndcg10)

        q_record = {
            "id": qid,
            "query": query,
            "tier": tier,
            "expected": expected_chapters[0] if len(expected_chapters) == 1 else expected_chapters,
            "expected_chapters": expected_chapters,
            # Modern metrics
            "chunk_ndcg@5": round(chunk_ndcg5, 4),
            "chunk_ndcg@10": round(chunk_ndcg10, 4),
            "grade2_mrr": round(g2_mrr, 4),
            "grade2_rank": g2_rank,
            "evidence_recall@5": round(ev_r5, 4),
            "evidence_recall@10": round(ev_r10, 4),
            "evidence_hit@5": ev_hit5,
            "evidence_hit@10": ev_hit10,
            # Legacy metrics
            "chapter_recall@5": ch_r5,
            "chapter_recall@10": ch_r10,
            "chapter_mrr": round(ch_mrr, 4),
            "chapter_ndcg@10": round(ch_ndcg10, 4),
            "chapter_rank": ch_rank,
            # Legacy backward-compatibility keys
            "r5": ch_r5,
            "r10": ch_r10,
            "mrr": round(g2_mrr if relevant_chunks else ch_mrr, 4),
            "ndcg": round(chunk_ndcg10, 4),
            "rank": g2_rank if g2_rank is not None else ch_rank,
            "retrieved_top_5": [
                {"rank": r.get("rank", i), "chunk_id": r.get("chunk_id"), "chapter": r.get("chapter_number")}
                for i, r in enumerate(results[:5], 1)
            ],
        }
        per_query_details.append(q_record)
        if tier in tier_records:
            tier_records[tier].append(q_record)

        if verbose:
            hit_marker = "✅" if (ev_hit10 if evidence_spans else ch_r10) else "❌"
            ret_ids = [r.get("chunk_id", str(r.get("chapter_number"))) for r in results[:5]]
            print(f"\n{hit_marker} [Q{qid:02d}][T{tier}] {query[:70]}")
            print(f"   Expected Ch: {expected_chapters} | Top Chunks: {ret_ids}")
            print(f"   Chunk NDCG@10: {chunk_ndcg10:.3f} | G2-MRR: {g2_mrr:.3f} (Rank: {g2_rank}) | Ev-R@10: {ev_r10:.3f}")
            print(f"   Legacy Ch-R@5: {ch_r5} | Ch-R@10: {ch_r10} | Ch-MRR: {ch_mrr:.3f}")

    # Build tier summaries
    tier_summary = {}
    for t in tiers:
        t_records = tier_records[t]
        n_t = len(t_records)
        if n_t > 0:
            tier_summary[t] = {
                "count": n_t,
                "chunk_ndcg@5": round(float(np.mean([r["chunk_ndcg@5"] for r in t_records])), 4),
                "chunk_ndcg@10": round(float(np.mean([r["chunk_ndcg@10"] for r in t_records])), 4),
                "grade2_mrr": round(float(np.mean([r["grade2_mrr"] for r in t_records])), 4),
                "evidence_recall@5": round(float(np.mean([r["evidence_recall@5"] for r in t_records])), 4),
                "evidence_recall@10": round(float(np.mean([r["evidence_recall@10"] for r in t_records])), 4),
                "evidence_hit@5": round(float(np.mean([r["evidence_hit@5"] for r in t_records])), 4),
                "evidence_hit@10": round(float(np.mean([r["evidence_hit@10"] for r in t_records])), 4),
                "chapter_recall@5": round(float(np.mean([r["chapter_recall@5"] for r in t_records])), 4),
                "chapter_recall@10": round(float(np.mean([r["chapter_recall@10"] for r in t_records])), 4),
                "chapter_mrr": round(float(np.mean([r["chapter_mrr"] for r in t_records])), 4),
            }
        else:
            tier_summary[t] = {"count": 0}

    # Summary dictionary
    summary = {
        "name": name,
        "query_count": len(eval_set),
        # Modern Passage-Level Metrics
        "modern_metrics": {
            "chunk_ndcg@5": round(float(np.mean(ndcg5_list)), 4),
            "chunk_ndcg@10": round(float(np.mean(ndcg10_list)), 4),
            "grade2_mrr": round(float(np.mean(g2_mrr_list)), 4),
            "evidence_recall@5": round(float(np.mean(ev_r5_list)), 4),
            "evidence_recall@10": round(float(np.mean(ev_r10_list)), 4),
            "evidence_hit@5": round(float(np.mean(ev_hit5_list)), 4),
            "evidence_hit@10": round(float(np.mean(ev_hit10_list)), 4),
        },
        # Legacy Chapter-Level Metrics
        "legacy_metrics": {
            "chapter_recall@5": round(float(np.mean(ch_r5_list)), 4),
            "chapter_recall@10": round(float(np.mean(ch_r10_list)), 4),
            "chapter_mrr": round(float(np.mean(ch_mrr_list)), 4),
            "chapter_ndcg@10": round(float(np.mean(ch_ndcg10_list)), 4),
        },
        # Direct backward-compatible keys
        "recall@5": round(float(np.mean(ch_r5_list)), 3),
        "recall@10": round(float(np.mean(ch_r10_list)), 3),
        "mrr": round(float(np.mean(g2_mrr_list if any(item.get("relevant_chunks") for item in eval_set) else ch_mrr_list)), 3),
        "ndcg@10": round(float(np.mean(ndcg10_list if any(item.get("relevant_chunks") for item in eval_set) else ch_ndcg10_list)), 3),
        "tier_summary": tier_summary,
        "tier_mrr": {t: tier_summary[t].get("grade2_mrr", 0.0) for t in tiers},
        "tier_r5": {t: tier_summary[t].get("chapter_recall@5", 0.0) for t in tiers},
        "tier_r10": {t: tier_summary[t].get("chapter_recall@10", 0.0) for t in tiers},
        "tier_ndcg": {t: tier_summary[t].get("chunk_ndcg@10", 0.0) for t in tiers},
        "queries": per_query_details,
    }

    # Console output
    m = summary["modern_metrics"]
    leg = summary["legacy_metrics"]
    print(f"\n{'═' * 70}")
    print(f"  RESULTS: {name} (n={len(eval_set)} queries, K={k})")
    print(f"{'═' * 70}")
    print(f"  MODERN PASSAGE-LEVEL METRICS:")
    print(f"    Graded Chunk NDCG@5  : {m['chunk_ndcg@5']:.4f}")
    print(f"    Graded Chunk NDCG@10 : {m['chunk_ndcg@10']:.4f}")
    print(f"    Grade-2 Chunk MRR    : {m['grade2_mrr']:.4f}")
    print(f"    Evidence Recall@5    : {m['evidence_recall@5']:.4f}")
    print(f"    Evidence Recall@10   : {m['evidence_recall@10']:.4f}  (Hit@10: {m['evidence_hit@10']:.4f})")
    print(f"  LEGACY CHAPTER-LEVEL METRICS (HISTORICAL BASELINE):")
    print(f"    Chapter Recall@5     : {leg['chapter_recall@5']:.4f}  ({sum(ch_r5_list)}/{len(ch_r5_list)} hits)")
    print(f"    Chapter Recall@10    : {leg['chapter_recall@10']:.4f}  ({sum(ch_r10_list)}/{len(ch_r10_list)} hits)")
    print(f"    Chapter MRR          : {leg['chapter_mrr']:.4f}")
    print(f"    Chapter NDCG@10      : {leg['chapter_ndcg@10']:.4f}")
    print(f"  ── DIFFICULTY BREAKDOWN (Grade-2 MRR | Chunk NDCG@10 | Evidence R@10 | Ch-R@5) ──")
    for t in tiers:
        ts = tier_summary[t]
        label = {1: "Tier 1 Direct  ", 2: "Tier 2 Indirect", 3: "Tier 3 Hard    "}.get(t, f"Tier {t}")
        print(
            f"    {label} (n={ts['count']:2d}): "
            f"G2-MRR={ts['grade2_mrr']:.3f} | "
            f"NDCG@10={ts['chunk_ndcg@10']:.3f} | "
            f"Ev-R@10={ts['evidence_recall@10']:.3f} | "
            f"Ch-R@5={ts['chapter_recall@5']:.3f}"
        )
    print(f"{'═' * 70}\n")

    return summary


# =====================================================
# REGRESSION & COMPARISON ANALYSIS
# =====================================================

def analyze_regressions(baseline_run: dict, experiment_run: dict, system_label: str = "Dense"):
    """Identify queries where the experiment improves, regresses, or ties."""
    b_queries = baseline_run["queries"]
    e_queries = experiment_run["queries"]

    improved = []
    regressed = []
    tied = []

    for b, e in zip(b_queries, e_queries):
        diff = e["mrr"] - b["mrr"]
        record = {
            "query": b["query"],
            "tier": b["tier"],
            "expected": b["expected"],
            "base_rank": b["rank"],
            "exp_rank": e["rank"],
            "base_mrr": b["mrr"],
            "exp_mrr": e["mrr"],
            "diff_mrr": diff,
        }
        if diff > 1e-4:
            improved.append(record)
        elif diff < -1e-4:
            regressed.append(record)
        else:
            tied.append(record)

    print(f"\n{'═' * 75}")
    print(f"  REGRESSION ANALYSIS: {system_label} (Baseline 800/150 vs Experiment 450/64)")
    print(f"{'═' * 75}")
    print(f"  Total Queries : {len(b_queries)}")
    print(f"  Improved (▲)  : {len(improved)}")
    print(f"  Regressed (▼) : {len(regressed)}")
    print(f"  Tied (=)      : {len(tied)}")

    # Tier 3 breakdown
    t3_improved = [r for r in improved if r["tier"] == 3]
    t3_regressed = [r for r in regressed if r["tier"] == 3]
    t3_tied = [r for r in tied if r["tier"] == 3]
    print(f"  Tier 3 (Hard) : {len(t3_improved)} improved, {len(t3_regressed)} regressed, {len(t3_tied)} tied")
    print(f"{'─' * 75}")

    if improved:
        print("\n  ▲ IMPROVED QUERIES:")
        for r in improved:
            print(f"    [T{r['tier']}] {r['query'][:65]}...")
            print(f"        Rank: {r['base_rank']} → {r['exp_rank']} (MRR: {r['base_mrr']:.3f} → {r['exp_mrr']:.3f})")

    if regressed:
        print("\n  ▼ REGRESSED QUERIES:")
        for r in regressed:
            print(f"    [T{r['tier']}] {r['query'][:65]}...")
            print(f"        Rank: {r['base_rank']} → {r['exp_rank']} (MRR: {r['base_mrr']:.3f} → {r['exp_mrr']:.3f})")

    print(f"{'═' * 75}\n")
    return {"improved": improved, "regressed": regressed, "tied": tied}


def print_comparison_table(results: list):
    print(f"\n{'═' * 105}")
    print(f"  RETRIEVAL BENCHMARK COMPARISON TABLE  (Modern Passage-Level + Legacy Chapter Metrics)")
    print(f"{'═' * 105}")
    print(f"  {'System':<30} {'NDCG@5':>7} {'NDCG@10':>8} {'G2-MRR':>7} {'Ev-R@5':>7} {'Ev-R@10':>8} | {'Ch-R@5':>7} {'Ch-R@10':>8} {'Ch-MRR':>7}")
    print(f"  {'─' * 103}")
    for r in results:
        m = r.get("modern_metrics", {})
        leg = r.get("legacy_metrics", {})
        ndcg5 = m.get("chunk_ndcg@5", 0.0)
        ndcg10 = m.get("chunk_ndcg@10", r.get("ndcg@10", 0.0))
        g2_mrr = m.get("grade2_mrr", r.get("mrr", 0.0))
        ev_r5 = m.get("evidence_recall@5", 0.0)
        ev_r10 = m.get("evidence_recall@10", 0.0)
        ch_r5 = leg.get("chapter_recall@5", r.get("recall@5", 0.0))
        ch_r10 = leg.get("chapter_recall@10", r.get("recall@10", 0.0))
        ch_mrr = leg.get("chapter_mrr", 0.0)

        print(
            f"  {r['name']:<30} "
            f"{ndcg5:>7.3f} "
            f"{ndcg10:>8.3f} "
            f"{g2_mrr:>7.3f} "
            f"{ev_r5:>7.3f} "
            f"{ev_r10:>8.3f} | "
            f"{ch_r5:>7.3f} "
            f"{ch_r10:>8.3f} "
            f"{ch_mrr:>7.3f}"
        )
    print(f"{'═' * 105}\n")


# =====================================================
# MAIN / ENTRYPOINT
# =====================================================

def run_benchmarks(mode: str = "compare"):
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    all_results = []

    print("\n🚀 Running BASELINE retrieval benchmarks (800/150)...")
    base_dense = evaluate_system(
        lambda q, return_results=True: dense_search(q, return_results=True, preset="baseline"),
        "Baseline Dense (800/150)",
        verbose=False,
    )
    base_sparse = evaluate_system(
        lambda q, return_results=True: sparse_search(q, return_results=True, preset="baseline"),
        "Baseline Sparse (800/150)",
        verbose=False,
    )
    base_hybrid = evaluate_system(
        lambda q, return_results=True: hybrid_search(q, alpha=0.7, return_results=True, preset="baseline"),
        "Baseline Hybrid α=0.7 (800/150)",
        verbose=False,
    )

    all_results.extend([base_dense, base_sparse, base_hybrid])

    if mode in ("experiment", "compare"):
        print("\n🚀 Running EXPERIMENTAL retrieval benchmarks (450/64)...")
        exp_dense = evaluate_system(
            lambda q, return_results=True: dense_search(q, return_results=True, preset="experiment_450_64"),
            "Experiment Dense (450/64)",
            verbose=False,
        )
        exp_sparse = evaluate_system(
            lambda q, return_results=True: sparse_search(q, return_results=True, preset="experiment_450_64"),
            "Experiment Sparse (450/64)",
            verbose=False,
        )
        exp_hybrid = evaluate_system(
            lambda q, return_results=True: hybrid_search(q, alpha=0.7, return_results=True, preset="experiment_450_64"),
            "Experiment Hybrid α=0.7 (450/64)",
            verbose=False,
        )

        all_results.extend([exp_dense, exp_sparse, exp_hybrid])

        print_comparison_table(all_results)

        # Regression analysis for Dense
        dense_reg = analyze_regressions(base_dense, exp_dense, system_label="Dense Retrieval")

        # Regression analysis for Hybrid
        hybrid_reg = analyze_regressions(base_hybrid, exp_hybrid, system_label="Hybrid Retrieval (α=0.7)")

        # Save structured results to disk
        out_path = PROJECT_ROOT / "evaluation" / "results" / "historical" / "retrieval_comparison_results.json"
        try:
            import json
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            print(f"💾 Saved full comparative evaluation results to: {out_path}")
        except Exception as e:
            print(f"⚠️ Could not save evaluation JSON: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MediRAG retrieval benchmarks.")
    parser.add_argument("--mode", type=str, default="compare", choices=["baseline", "experiment", "compare"])
    args = parser.parse_args()
    run_benchmarks(mode=args.mode)