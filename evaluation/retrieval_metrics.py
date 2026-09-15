from pathlib import Path
import sys
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

def evaluate_system(search_fn, name: str, verbose: bool = DEBUG):

    recalls_5   = []
    recalls_10  = []
    mrr_scores  = []
    ndcg_scores = []
    per_query_details = []

    tier_mrr = {1: [], 2: [], 3: []}
    tier_r5  = {1: [], 2: [], 3: []}
    tier_r10 = {1: [], 2: [], 3: []}
    tier_ndcg = {1: [], 2: [], 3: []}

    if verbose:
        print(f"\n{'─' * 65}")
        print(f"  DEBUG: {name}")
        print(f"{'─' * 65}")

    for idx, item in enumerate(EVAL_QUERIES):
        query    = item["query"]
        expected = item["relevant_chapter"]
        tier     = item["tier"]

        results = search_fn(query, return_results=True)

        r5   = recall_at_k(results, expected, 5)
        r10  = recall_at_k(results, expected, 10)
        mrr  = mrr_score(results, expected)
        ndcg = ndcg_at_k(results, expected, 10)

        # Determine rank of expected chapter
        expected_rank = None
        for rank, r in enumerate(results, 1):
            if r["chapter_number"] == expected:
                expected_rank = rank
                break

        recalls_5.append(r5)
        recalls_10.append(r10)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        tier_mrr[tier].append(mrr)
        tier_r5[tier].append(r5)
        tier_r10[tier].append(r10)
        tier_ndcg[tier].append(ndcg)

        per_query_details.append({
            "idx": idx,
            "query": query,
            "expected": expected,
            "tier": tier,
            "r5": r5,
            "r10": r10,
            "mrr": mrr,
            "ndcg": ndcg,
            "rank": expected_rank,
        })

        if verbose:
            returned_chapters = [r["chapter_number"] for r in results[:10]]
            hit_marker = "✅" if r10 else "❌"
            print(f"\n{hit_marker} [T{tier}] {query[:75]}")
            print(f"   Expected : Ch.{expected}")
            print(f"   Top-10   : {returned_chapters}")
            print(f"   R@5={r5}  R@10={r10}  MRR={mrr:.3f}  NDCG={ndcg:.3f}")

    tier_summary = {
        t: round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_mrr.items()
    }
    tier_r5_summary = {
        t: round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_r5.items()
    }
    tier_r10_summary = {
        t: round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_r10.items()
    }
    tier_ndcg_summary = {
        t: round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_ndcg.items()
    }

    print(f"\n{'═' * 65}")
    print(f"  RESULTS: {name}  (n={len(EVAL_QUERIES)} queries, K={RETRIEVAL_K})")
    print(f"{'═' * 65}")
    print(f"  Recall@5   : {np.mean(recalls_5):.3f}   ({sum(recalls_5)}/{len(recalls_5)} hits)")
    print(f"  Recall@10  : {np.mean(recalls_10):.3f}   ({sum(recalls_10)}/{len(recalls_10)} hits)")
    print(f"  MRR        : {np.mean(mrr_scores):.3f}")
    print(f"  NDCG@10    : {np.mean(ndcg_scores):.3f}")
    print(f"  ── MRR by difficulty ──────────────────────────────────")
    print(f"  Tier 1 Direct    (n={len(tier_mrr[1])}): {tier_summary[1]:.3f}  (R@5={tier_r5_summary[1]:.3f})")
    print(f"  Tier 2 Indirect  (n={len(tier_mrr[2])}): {tier_summary[2]:.3f}  (R@5={tier_r5_summary[2]:.3f})")
    print(f"  Tier 3 Hard      (n={len(tier_mrr[3])}): {tier_summary[3]:.3f}  (R@5={tier_r5_summary[3]:.3f})")
    print(f"{'═' * 65}")

    return {
        "name":      name,
        "recall@5":  round(float(np.mean(recalls_5)),  3),
        "recall@10": round(float(np.mean(recalls_10)), 3),
        "mrr":       round(float(np.mean(mrr_scores)),  3),
        "ndcg@10":   round(float(np.mean(ndcg_scores)), 3),
        "tier_mrr":  tier_summary,
        "tier_r5":   tier_r5_summary,
        "tier_r10":  tier_r10_summary,
        "tier_ndcg": tier_ndcg_summary,
        "queries":   per_query_details,
    }


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
    print(f"\n{'═' * 78}")
    print(f"  RETRIEVAL COMPARISON TABLE  (n=50 queries across 3 difficulty tiers)")
    print(f"{'═' * 78}")
    print(f"  {'System':<28} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG@10':>9}  {'T1':>5} {'T2':>5} {'T3':>5}")
    print(f"  {'─' * 74}")
    for r in results:
        t = r["tier_mrr"]
        print(
            f"  {r['name']:<28} "
            f"{r['recall@5']:>6.3f} "
            f"{r['recall@10']:>6.3f} "
            f"{r['mrr']:>6.3f} "
            f"{r['ndcg@10']:>9.3f}  "
            f"{t[1]:>5.3f} {t[2]:>5.3f} {t[3]:>5.3f}"
        )
    print(f"{'═' * 78}")
    print(f"  T1=Direct  T2=Indirect  T3=Hard\n")


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
        out_path = PROJECT_ROOT / "evaluation" / "retrieval_comparison_results.json"
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