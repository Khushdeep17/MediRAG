import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import faiss

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.dense import dense_search
from retrieval.sparse import sparse_search
from retrieval.fusion import hybrid_search
from evaluation.retrieval_metrics import (
    EVAL_QUERIES,
    recall_at_k,
    mrr_score,
    ndcg_at_k,
)

# =====================================================
# INTEGRITY CHECKS
# =====================================================

def verify_integrity() -> Dict[str, Any]:
    print("\n🔍 Running pre-flight integrity checks...")

    chunks_path = PROJECT_ROOT / "data" / "processed" / "merck_chunks_800_150.json"
    emb_path = PROJECT_ROOT / "embeddings" / "embeddings.npy"
    ids_path = PROJECT_ROOT / "embeddings" / "ids.json"
    index_path = PROJECT_ROOT / "index" / "faiss.index"

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    emb = np.load(emb_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))

    chunk_count = len(chunks)
    emb_shape = list(emb.shape)
    ids_count = len(ids)
    faiss_ntotal = index.ntotal
    ids_aligned = (chunk_count == emb_shape[0] == ids_count == faiss_ntotal == 4239) and (
        set(c["chunk_id"] for c in chunks) == set(ids)
    )

    queries_count = len(EVAL_QUERIES)

    assert chunk_count == 4239, f"Baseline chunk count expected 4239, got {chunk_count}"
    assert emb_shape == [4239, 1024], f"Baseline embeddings shape expected [4239, 1024], got {emb_shape}"
    assert faiss_ntotal == 4239, f"Baseline FAISS ntotal expected 4239, got {faiss_ntotal}"
    assert ids_aligned, "Baseline IDs are not aligned"
    assert queries_count == 50, f"Query count expected 50, got {queries_count}"

    print("  [PASS] Baseline chunk count: 4,239")
    print(f"  [PASS] Baseline embeddings shape: {emb_shape}")
    print(f"  [PASS] Baseline FAISS ntotal: {faiss_ntotal}")
    print("  [PASS] Baseline IDs aligned: True")
    print(f"  [PASS] Evaluation queries count: {queries_count}")

    return {
        "baseline_chunk_count": chunk_count,
        "baseline_embeddings_shape": emb_shape,
        "baseline_faiss_ntotal": faiss_ntotal,
        "baseline_ids_count": ids_count,
        "baseline_ids_aligned": ids_aligned,
        "eval_queries_count": queries_count,
    }


# =====================================================
# SYSTEM EVALUATION
# =====================================================

def evaluate_retrieval_system(search_fn, name: str) -> Dict[str, Any]:
    print(f"\nEvaluating: {name} (n={len(EVAL_QUERIES)} queries)...")
    recalls_5 = []
    recalls_10 = []
    mrr_scores = []
    ndcg_scores = []
    per_query_details = []

    tier_mrr = {1: [], 2: [], 3: []}
    tier_r5 = {1: [], 2: [], 3: []}
    tier_r10 = {1: [], 2: [], 3: []}
    tier_ndcg = {1: [], 2: [], 3: []}

    for idx, item in enumerate(EVAL_QUERIES):
        query = item["query"]
        expected = item["relevant_chapter"]
        tier = item["tier"]

        results = search_fn(query, return_results=True)

        r5 = recall_at_k(results, expected, 5)
        r10 = recall_at_k(results, expected, 10)
        mrr = mrr_score(results, expected)
        ndcg = ndcg_at_k(results, expected, 10)

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
            "mrr": round(float(mrr), 4),
            "ndcg": round(float(ndcg), 4),
            "rank": expected_rank,
            "top_chunk_ids": [r["chunk_id"] for r in results[:10]],
            "top_chapters": [r["chapter_number"] for r in results[:10]],
        })

    tier_summary = {
        str(t): round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_mrr.items()
    }
    tier_r5_summary = {
        str(t): round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_r5.items()
    }
    tier_r10_summary = {
        str(t): round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_r10.items()
    }
    tier_ndcg_summary = {
        str(t): round(float(np.mean(scores)), 3) if scores else 0.0
        for t, scores in tier_ndcg.items()
    }

    res = {
        "name": name,
        "recall@5": round(float(np.mean(recalls_5)), 3),
        "recall@10": round(float(np.mean(recalls_10)), 3),
        "mrr": round(float(np.mean(mrr_scores)), 3),
        "ndcg@10": round(float(np.mean(ndcg_scores)), 3),
        "tier_mrr": tier_summary,
        "tier_r5": tier_r5_summary,
        "tier_r10": tier_r10_summary,
        "tier_ndcg": tier_ndcg_summary,
        "queries": per_query_details,
    }
    print(f"  --> R@5: {res['recall@5']:.3f} | R@10: {res['recall@10']:.3f} | MRR: {res['mrr']:.3f} | NDCG@10: {res['ndcg@10']:.3f} (T1: {tier_summary['1']}, T2: {tier_summary['2']}, T3: {tier_summary['3']})")
    return res


# =====================================================
# QUERY MOVEMENT ANALYSIS
# =====================================================

def analyze_query_movement(
    baseline_run: dict,
    experiment_run: dict,
    comparison_name: str,
    sparse_cache: Dict[str, List[Dict[str, Any]]] = None,
    dense_cache: Dict[str, List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    b_queries = baseline_run["queries"]
    e_queries = experiment_run["queries"]
    total = len(b_queries)

    improved = []
    regressed = []
    tied = []

    for b, e in zip(b_queries, e_queries):
        diff_mrr = round(e["mrr"] - b["mrr"], 4)
        record = {
            "idx": b["idx"],
            "query": b["query"],
            "tier": b["tier"],
            "expected": b["expected"],
            "base_rank": b["rank"],
            "exp_rank": e["rank"],
            "base_mrr": b["mrr"],
            "exp_mrr": e["mrr"],
            "diff_mrr": diff_mrr,
            "base_r5": b["r5"],
            "exp_r5": e["r5"],
            "base_ndcg": b["ndcg"],
            "exp_ndcg": e["ndcg"],
        }

        if diff_mrr > 1e-4:
            improved.append(record)
        elif diff_mrr < -1e-4:
            regressed.append(record)
        else:
            tied.append(record)

    def tier_stats(records):
        return {
            "tier_1": sum(1 for r in records if r["tier"] == 1),
            "tier_2": sum(1 for r in records if r["tier"] == 2),
            "tier_3": sum(1 for r in records if r["tier"] == 3),
        }

    return {
        "comparison": comparison_name,
        "total_queries": total,
        "improved_count": len(improved),
        "improved_pct": round((len(improved) / total) * 100, 1),
        "regressed_count": len(regressed),
        "regressed_pct": round((len(regressed) / total) * 100, 1),
        "tied_count": len(tied),
        "tied_pct": round((len(tied) / total) * 100, 1),
        "improved_tier_breakdown": tier_stats(improved),
        "regressed_tier_breakdown": tier_stats(regressed),
        "tied_tier_breakdown": tier_stats(tied),
        "improved_queries": improved,
        "regressed_queries": regressed,
    }


# =====================================================
# CANDIDATE-UNION DIAGNOSTICS
# =====================================================

def compute_candidate_diagnostics(
    depth: int,
    hybrid_run: dict,
    dense_cache: Dict[str, List[Dict[str, Any]]],
    sparse_cache: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Computes candidate-union diagnostics:
    - average number of unique candidates in dense ∪ sparse before fusion
    - number of final top-10 results that appear only in sparse candidates
    - number of final top-10 results that appear only in dense candidates
    - number appearing in both
    - number of queries where sparse introduced a relevant chunk missed by dense top-20
    """
    union_sizes = []
    dense_only_top10_total = 0
    sparse_only_top10_total = 0
    both_top10_total = 0

    sparse_rescued_relevant_queries = []

    for q_info in hybrid_run["queries"]:
        q = q_info["query"]
        expected = q_info["expected"]
        top_chunks = q_info["top_chunk_ids"]

        dense_cands = dense_cache[q][:depth]
        sparse_cands = sparse_cache[q][:depth]

        d_ids = set(c["chunk_id"] for c in dense_cands)
        s_ids = set(c["chunk_id"] for c in sparse_cands)
        union_ids = d_ids | s_ids

        union_sizes.append(len(union_ids))

        # Check top-10 chunks origin
        for cid in top_chunks:
            in_d = cid in d_ids
            in_s = cid in s_ids
            if in_d and in_s:
                both_top10_total += 1
            elif in_d:
                dense_only_top10_total += 1
            elif in_s:
                sparse_only_top10_total += 1

        # Check if sparse at this depth introduced a relevant chunk that dense top-20 missed
        dense_20_chaps = {c["chapter_number"] for c in dense_cache[q][:20]}
        if expected not in dense_20_chaps:
            # Dense top-20 completely missed the relevant chapter!
            # Did sparse have it?
            sparse_depth_chaps = {c["chapter_number"] for c in sparse_cands}
            if expected in sparse_depth_chaps:
                # Did it make it into the final hybrid top 10?
                if expected in q_info["top_chapters"]:
                    sparse_rescued_relevant_queries.append({
                        "idx": q_info["idx"],
                        "query": q,
                        "tier": q_info["tier"],
                        "expected": expected,
                        "final_rank": q_info["rank"],
                    })

    total_top10_evaluated = len(hybrid_run["queries"]) * 10

    return {
        "candidate_depth": depth,
        "avg_unique_candidates_in_union": round(float(np.mean(union_sizes)), 2),
        "total_top10_results_evaluated": total_top10_evaluated,
        "final_top10_dense_only_count": dense_only_top10_total,
        "final_top10_dense_only_pct": round((dense_only_top10_total / total_top10_evaluated) * 100, 1),
        "final_top10_sparse_only_count": sparse_only_top10_total,
        "final_top10_sparse_only_pct": round((sparse_only_top10_total / total_top10_evaluated) * 100, 1),
        "final_top10_both_count": both_top10_total,
        "final_top10_both_pct": round((both_top10_total / total_top10_evaluated) * 100, 1),
        "sparse_rescued_relevant_count": len(sparse_rescued_relevant_queries),
        "sparse_rescued_relevant_queries": sparse_rescued_relevant_queries,
    }


# =====================================================
# MAIN RUNNER
# =====================================================

def run_candidate_depth_experiment():
    integrity_data = verify_integrity()

    # Pre-cache dense and sparse candidates up to depth 100 for all 50 queries
    print("\n📦 Generating canonical dense and sparse search results up to depth 100 for 50 queries...")
    dense_cache: Dict[str, List[Dict[str, Any]]] = {}
    sparse_cache: Dict[str, List[Dict[str, Any]]] = {}

    for item in EVAL_QUERIES:
        q = item["query"]
        dense_cache[q] = dense_search(q, top_k=100, return_results=True, preset="baseline")
        sparse_cache[q] = sparse_search(q, top_k=100, return_results=True, preset="baseline", tokenizer="baseline")

    def make_dense_fn(depth: int):
        return lambda q, top_k=depth, return_results=True: dense_cache[q][:top_k]

    def make_sparse_fn(depth: int):
        return lambda q, top_k=depth, return_results=True: sparse_cache[q][:top_k]

    # 1. Fixed Reference: Dense Baseline
    dense_baseline = evaluate_retrieval_system(
        lambda q, return_results=True: dense_cache[q][:10],
        "Dense baseline (800/150)"
    )

    # 2. Fixed Reference: Sparse Baseline
    sparse_baseline = evaluate_retrieval_system(
        lambda q, return_results=True: sparse_cache[q][:10],
        "Sparse baseline (800/150)"
    )

    # 3. Hybrid Candidate Depth = 20 (Baseline Production Setting)
    hybrid_depth_20 = evaluate_retrieval_system(
        lambda q, return_results=True: hybrid_search(
            q,
            alpha=0.7,
            top_k=10,
            return_results=True,
            preset="baseline",
            tokenizer="baseline",
            dense_fn=make_dense_fn(20),
            sparse_fn=make_sparse_fn(20),
            candidate_depth=20,
        ),
        "Hybrid candidate_depth = 20 (Baseline, α=0.7)"
    )

    # 4. Hybrid Candidate Depth = 50
    hybrid_depth_50 = evaluate_retrieval_system(
        lambda q, return_results=True: hybrid_search(
            q,
            alpha=0.7,
            top_k=10,
            return_results=True,
            preset="baseline",
            tokenizer="baseline",
            dense_fn=make_dense_fn(50),
            sparse_fn=make_sparse_fn(50),
            candidate_depth=50,
        ),
        "Hybrid candidate_depth = 50 (α=0.7)"
    )

    # 5. Hybrid Candidate Depth = 100
    hybrid_depth_100 = evaluate_retrieval_system(
        lambda q, return_results=True: hybrid_search(
            q,
            alpha=0.7,
            top_k=10,
            return_results=True,
            preset="baseline",
            tokenizer="baseline",
            dense_fn=make_dense_fn(100),
            sparse_fn=make_sparse_fn(100),
            candidate_depth=100,
        ),
        "Hybrid candidate_depth = 100 (α=0.7)"
    )

    # Movement Analyses
    movement_20_to_50 = analyze_query_movement(hybrid_depth_20, hybrid_depth_50, "Hybrid Depth 20 -> Depth 50")
    movement_50_to_100 = analyze_query_movement(hybrid_depth_50, hybrid_depth_100, "Hybrid Depth 50 -> Depth 100")
    movement_20_to_100 = analyze_query_movement(hybrid_depth_20, hybrid_depth_100, "Hybrid Depth 20 -> Depth 100")

    # Candidate-Union Diagnostics
    diag_20 = compute_candidate_diagnostics(20, hybrid_depth_20, dense_cache, sparse_cache)
    diag_50 = compute_candidate_diagnostics(50, hybrid_depth_50, dense_cache, sparse_cache)
    diag_100 = compute_candidate_diagnostics(100, hybrid_depth_100, dense_cache, sparse_cache)

    systems = [
        dense_baseline,
        sparse_baseline,
        hybrid_depth_20,
        hybrid_depth_50,
        hybrid_depth_100,
    ]

    # Save to JSON
    results_json = {
        "experiment_metadata": {
            "title": "Controlled RRF Candidate-Depth Experiment",
            "date": "2026-09-16",
            "corpus": "baseline 800/150",
            "num_chunks": 4239,
            "num_queries": 50,
            "dense_model": "BAAI/bge-large-en-v1.5",
            "bm25_tokenizer": "baseline",
            "alpha": 0.7,
            "rrf_k": 60,
            "final_top_k": 10,
        },
        "candidate_depth_configurations": [20, 50, 100],
        "integrity_checks": integrity_data,
        "overall_metrics": {
            s["name"]: {
                "recall@5": s["recall@5"],
                "recall@10": s["recall@10"],
                "mrr": s["mrr"],
                "ndcg@10": s["ndcg@10"],
            }
            for s in systems
        },
        "tier_metrics": {
            s["name"]: {
                "tier_mrr": s["tier_mrr"],
                "tier_r5": s["tier_r5"],
                "tier_r10": s["tier_r10"],
                "tier_ndcg": s["tier_ndcg"],
            }
            for s in systems
        },
        "query_movement_statistics": {
            "depth_20_to_50": movement_20_to_50,
            "depth_50_to_100": movement_50_to_100,
            "depth_20_to_100": movement_20_to_100,
        },
        "candidate_union_diagnostics": {
            "depth_20": diag_20,
            "depth_50": diag_50,
            "depth_100": diag_100,
        },
        "systems": systems,
    }

    out_file = Path(__file__).resolve().parent / "retrieval_candidate_depth_results.json"
    out_file.write_text(json.dumps(results_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n💾 Benchmark results saved to: {out_file}")

    # =====================================================
    # TERMINAL SUMMARY TABLE
    # =====================================================
    print(f"\n{'═' * 95}")
    print(f"  CONTROLLED RRF CANDIDATE-DEPTH EXPERIMENT SUMMARY")
    print(f"  Corpus: Baseline 800/150 (4,239 chunks) | Queries: 50 | α=0.7 | RRF k=60 | Final K=10")
    print(f"{'═' * 95}")
    print(f"  {'Configuration':<42} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG@10':>9}  {'T1 MRR':>7} {'T2 MRR':>7} {'T3 MRR':>7}")
    print(f"  {'─' * 93}")
    for s in systems:
        tm = s["tier_mrr"]
        print(
            f"  {s['name']:<42} "
            f"{s['recall@5']:>6.3f} "
            f"{s['recall@10']:>6.3f} "
            f"{s['mrr']:>6.3f} "
            f"{s['ndcg@10']:>9.3f}  "
            f"{tm['1']:>7.3f} {tm['2']:>7.3f} {tm['3']:>7.3f}"
        )
    print(f"{'═' * 95}")

    # Candidate Union Diagnostics Summary Table
    print(f"\n{'─' * 95}")
    print(f"  CANDIDATE UNION & PARTICIPATION DIAGNOSTICS (Total Top-10 Slots = 500)")
    print(f"{'─' * 95}")
    print(f"  {'Candidate Depth':<20} {'Avg Union Size':<16} {'Both Dense & Sparse':<24} {'Dense-Only':<18} {'Sparse-Only':<16}")
    print(f"  {'─' * 93}")
    for d, diag in [("Depth 20", diag_20), ("Depth 50", diag_50), ("Depth 100", diag_100)]:
        print(
            f"  {d:<20} "
            f"{diag['avg_unique_candidates_in_union']:<16.2f} "
            f"{diag['final_top10_both_count']:>3} ({diag['final_top10_both_pct']}%)            "
            f"{diag['final_top10_dense_only_count']:>3} ({diag['final_top10_dense_only_pct']}%)     "
            f"{diag['final_top10_sparse_only_count']:>3} ({diag['final_top10_sparse_only_pct']}%)"
        )
    print(f"{'═' * 95}")

    # Movement summary
    print(f"\n{'─' * 95}")
    print(f"  QUERY-LEVEL MOVEMENT BREAKDOWN")
    print(f"{'─' * 95}")
    for mov in [movement_20_to_50, movement_50_to_100, movement_20_to_100]:
        print(f"  ▶ {mov['comparison']}:")
        print(f"      Tied       (=) : {mov['tied_count']}/{mov['total_queries']} ({mov['tied_pct']}%)  [T1: {mov['tied_tier_breakdown']['tier_1']}, T2: {mov['tied_tier_breakdown']['tier_2']}, T3: {mov['tied_tier_breakdown']['tier_3']}]")
        print(f"      Improved   (▲) : {mov['improved_count']}/{mov['total_queries']} ({mov['improved_pct']}%)  [T1: {mov['improved_tier_breakdown']['tier_1']}, T2: {mov['improved_tier_breakdown']['tier_2']}, T3: {mov['improved_tier_breakdown']['tier_3']}]")
        print(f"      Regressed  (▼) : {mov['regressed_count']}/{mov['total_queries']} ({mov['regressed_pct']}%)  [T1: {mov['regressed_tier_breakdown']['tier_1']}, T2: {mov['regressed_tier_breakdown']['tier_2']}, T3: {mov['regressed_tier_breakdown']['tier_3']}]")
        
        if mov["improved_queries"]:
            print("      Improved Queries:")
            for q in mov["improved_queries"]:
                print(f"        • [T{q['tier']}] \"{q['query'][:60]}...\" Rank: {q['base_rank']} -> {q['exp_rank']} (MRR: {q['base_mrr']:.3f} -> {q['exp_mrr']:.3f})")
        if mov["regressed_queries"]:
            print("      Regressed Queries:")
            for q in mov["regressed_queries"]:
                print(f"        • [T{q['tier']}] \"{q['query'][:60]}...\" Rank: {q['base_rank']} -> {q['exp_rank']} (MRR: {q['base_mrr']:.3f} -> {q['exp_mrr']:.3f})")
        print()
    print(f"{'═' * 95}\n")


if __name__ == "__main__":
    run_candidate_depth_experiment()
