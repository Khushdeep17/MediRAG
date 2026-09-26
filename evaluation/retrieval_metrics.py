import sys
import os
import numpy as np

# --- Fix import path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

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

EVAL_QUERIES = [

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

# =====================================================
# METRICS
# =====================================================

def recall_at_k(results: list, relevant_chapter: int, k: int) -> int:
    top_k_chapters = {r["chapter_number"] for r in results[:k]}
    return int(relevant_chapter in top_k_chapters)


def mrr_score(results: list, relevant_chapter: int) -> float:
    seen = set()
    for rank, r in enumerate(results, 1):
        chap = r["chapter_number"]
        if chap in seen:
            continue
        seen.add(chap)
        if chap == relevant_chapter:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list, relevant_chapter: int, k: int) -> float:
    """Binary NDCG@K — rewards finding the relevant chapter higher in ranking."""
    seen = set()
    pos  = 0
    for r in results[:k]:
        chap = r["chapter_number"]
        if chap in seen:
            continue
        seen.add(chap)
        pos += 1
        if chap == relevant_chapter:
            return (1.0 / np.log2(pos + 1))   # IDCG = 1.0 (rank 1 = perfect)
    return 0.0

# =====================================================
# EVALUATION LOOP
# =====================================================

def evaluate_system(search_fn, name: str):

    recalls_5   = []
    recalls_10  = []
    mrr_scores  = []
    ndcg_scores = []

    tier_mrr = {1: [], 2: [], 3: []}

    if DEBUG:
        print(f"\n{'─' * 65}")
        print(f"  DEBUG: {name}")
        print(f"{'─' * 65}")

    for item in EVAL_QUERIES:
        query    = item["query"]
        expected = item["relevant_chapter"]
        tier     = item["tier"]

        results = search_fn(query, return_results=True)

        r5   = recall_at_k(results, expected, 5)
        r10  = recall_at_k(results, expected, 10)
        mrr  = mrr_score(results, expected)
        ndcg = ndcg_at_k(results, expected, 10)

        recalls_5.append(r5)
        recalls_10.append(r10)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        tier_mrr[tier].append(mrr)

        if DEBUG:
            returned_chapters = [r["chapter_number"] for r in results[:10]]
            hit_marker = "✅" if r10 else "❌"
            print(f"\n{hit_marker} [T{tier}] {query[:75]}")
            print(f"   Expected : Ch.{expected}")
            print(f"   Top-10   : {returned_chapters}")
            print(f"   R@5={r5}  R@10={r10}  MRR={mrr:.3f}  NDCG@10={ndcg:.3f}")

    # ── Per-tier MRR breakdown ───────────────────────────────────────────
    tier_summary = {
        t: round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_mrr.items()
    }

    print(f"\n{'═' * 65}")
    print(f"  RESULTS: {name}  (n={len(EVAL_QUERIES)} queries, K={RETRIEVAL_K})")
    print(f"{'═' * 65}")
    print(f"  Recall@5   : {np.mean(recalls_5):.3f}   ({sum(recalls_5)}/{len(recalls_5)} hits)")
    print(f"  Recall@10  : {np.mean(recalls_10):.3f}   ({sum(recalls_10)}/{len(recalls_10)} hits)")
    print(f"  MRR        : {np.mean(mrr_scores):.3f}")
    print(f"  NDCG@10    : {np.mean(ndcg_scores):.3f}")
    print(f"  ── MRR by difficulty ──────────────────────────────────")
    print(f"  Tier 1 Direct    (n={len(tier_mrr[1])}): {tier_summary[1]:.3f}")
    print(f"  Tier 2 Indirect  (n={len(tier_mrr[2])}): {tier_summary[2]:.3f}")
    print(f"  Tier 3 Hard      (n={len(tier_mrr[3])}): {tier_summary[3]:.3f}")
    print(f"{'═' * 65}")

    return {
        "name":      name,
        "recall@5":  round(float(np.mean(recalls_5)),  3),
        "recall@10": round(float(np.mean(recalls_10)), 3),
        "mrr":       round(float(np.mean(mrr_scores)),  3),
        "ndcg@10":   round(float(np.mean(ndcg_scores)), 3),
        "tier_mrr":  tier_summary,
    }

# =====================================================
# COMPARISON TABLE
# =====================================================

def print_comparison_table(results: list):
    print(f"\n{'═' * 72}")
    print(f"  FINAL COMPARISON TABLE  (n=50 queries across 3 difficulty tiers)")
    print(f"{'═' * 72}")
    print(f"  {'System':<22} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG@10':>9}  {'T1':>5} {'T2':>5} {'T3':>5}")
    print(f"  {'─' * 66}")
    for r in results:
        t = r["tier_mrr"]
        print(
            f"  {r['name']:<22} "
            f"{r['recall@5']:>6.3f} "
            f"{r['recall@10']:>6.3f} "
            f"{r['mrr']:>6.3f} "
            f"{r['ndcg@10']:>9.3f}  "
            f"{t[1]:>5.3f} {t[2]:>5.3f} {t[3]:>5.3f}"
        )
    print(f"{'═' * 72}")
    print(f"  T1=Direct  T2=Indirect  T3=Hard\n")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    all_results = []

    all_results.append(
        evaluate_system(
            lambda q, return_results=True: dense_search(q, return_results=True),
            "Dense (BGE)"
        )
    )

    all_results.append(
        evaluate_system(
            lambda q, return_results=True: sparse_search(q, return_results=True),
            "Sparse (BM25)"
        )
    )

    for alpha in [0.3, 0.5, 0.7]:
        all_results.append(
            evaluate_system(
                lambda q, a=alpha, return_results=True: hybrid_search(
                    q, alpha=a, return_results=True
                ),
                f"Hybrid α={alpha}"
            )
        )

    print_comparison_table(all_results)

    # Best config recommendation (by MRR as primary, NDCG as tiebreak)
    best = max(all_results, key=lambda x: (x["mrr"], x["ndcg@10"]))
    print(f"  🏆 Best config → {best['name']}")
    print(f"     Recall@10={best['recall@10']}  MRR={best['mrr']}  NDCG@10={best['ndcg@10']}")
    print(f"     Tier MRR breakdown: T1={best['tier_mrr'][1]}  T2={best['tier_mrr'][2]}  T3={best['tier_mrr'][3]}\n")