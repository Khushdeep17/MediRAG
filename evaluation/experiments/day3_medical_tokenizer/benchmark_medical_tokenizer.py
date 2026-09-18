import json
import sys
from pathlib import Path
from typing import Any, Dict, List
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
    RETRIEVAL_K,
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

def analyze_query_movement(baseline_run: dict, experiment_run: dict, comparison_name: str) -> Dict[str, Any]:
    b_queries = baseline_run["queries"]
    e_queries = experiment_run["queries"]
    total = len(b_queries)

    improved = []
    regressed = []
    tied = []

    for b, e in zip(b_queries, e_queries):
        diff_mrr = round(e["mrr"] - b["mrr"], 4)
        diff_r5 = e["r5"] - b["r5"]
        diff_r10 = e["r10"] - b["r10"]
        diff_ndcg = round(e["ndcg"] - b["ndcg"], 4)

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

        # Primary movement discriminator is MRR (rank movement)
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
# MAIN RUNNER
# =====================================================

def run_benchmark():
    integrity_data = verify_integrity()

    # Pre-cache dense search results to guarantee 100% identical dense input to hybrid
    print("\n📦 Generating canonical baseline dense search results for 50 queries...")
    dense_cache: Dict[str, List[Dict[str, Any]]] = {}
    for item in EVAL_QUERIES:
        q = item["query"]
        dense_cache[q] = dense_search(q, top_k=RETRIEVAL_K, return_results=True, preset="baseline")

    def cached_dense_fn(query: str, top_k: int = 20, return_results: bool = True):
        return dense_cache[query][:top_k]

    # 1. Baseline Dense
    dense_baseline = evaluate_retrieval_system(
        lambda q, return_results=True: dense_cache[q][:10],
        "Dense baseline (800/150)"
    )

    # 2. Sparse Baseline Tokenizer
    sparse_baseline = evaluate_retrieval_system(
        lambda q, return_results=True: sparse_search(q, top_k=10, return_results=True, preset="baseline", tokenizer="baseline"),
        "Sparse baseline tokenizer (800/150)"
    )

    # 3. Sparse Medical Tokenizer
    sparse_medical = evaluate_retrieval_system(
        lambda q, return_results=True: sparse_search(q, top_k=10, return_results=True, preset="baseline", tokenizer="medical"),
        "Sparse medical tokenizer (800/150)"
    )

    # 4. Hybrid Baseline Tokenizer
    hybrid_baseline = evaluate_retrieval_system(
        lambda q, return_results=True: hybrid_search(
            q, alpha=0.7, top_k=10, return_results=True, preset="baseline", tokenizer="baseline", dense_fn=cached_dense_fn
        ),
        "Hybrid baseline tokenizer (800/150, α=0.7)"
    )

    # 5. Hybrid Medical Tokenizer
    hybrid_medical = evaluate_retrieval_system(
        lambda q, return_results=True: hybrid_search(
            q, alpha=0.7, top_k=10, return_results=True, preset="baseline", tokenizer="medical", dense_fn=cached_dense_fn
        ),
        "Hybrid medical tokenizer (800/150, α=0.7)"
    )

    systems = [
        dense_baseline,
        sparse_baseline,
        sparse_medical,
        hybrid_baseline,
        hybrid_medical,
    ]

    # Analyze Query Movement
    sparse_movement = analyze_query_movement(
        sparse_baseline, sparse_medical, "Sparse Baseline -> Sparse Medical"
    )
    hybrid_movement = analyze_query_movement(
        hybrid_baseline, hybrid_medical, "Hybrid Baseline -> Hybrid Medical"
    )

    # Prepare complete output JSON
    results_json = {
        "experiment_metadata": {
            "title": "Comparative Medical BM25 Retrieval Benchmark",
            "date": "2026-09-16",
            "chunk_preset": "baseline (800/150)",
            "num_chunks": 4239,
            "num_queries": 50,
            "dense_model": "BAAI/bge-large-en-v1.5",
            "dense_weight_alpha": 0.7,
            "rrf_k": 60,
            "retrieval_k": 10,
        },
        "tokenizer_descriptions": {
            "baseline": "Standard alphanumeric regex tokenizer: re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split() with NLTK English stopwords.",
            "medical": "Medical-aware BM25 tokenizer: Greek letter transliteration (α->alpha, β->beta, γ->gamma, δ->delta, κ->kappa), Linnaean genus-species abbreviation contraction (H. pylori -> h_pylori + pylori), decimal preservation (2.5 mg -> 2.5), slash-unit normalization (mg/dL -> mg/dl + mg + dl), and dual-emission hyphenated compounds (covid-19 -> covid-19 + covid, type-2 -> type-2 + type) with numeric noise suppression."
        },
        "configurations": [
            "Dense baseline (800/150)",
            "Sparse baseline tokenizer (800/150)",
            "Sparse medical tokenizer (800/150)",
            "Hybrid baseline tokenizer (800/150, α=0.7)",
            "Hybrid medical tokenizer (800/150, α=0.7)",
        ],
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
            "sparse_baseline_to_medical": sparse_movement,
            "hybrid_baseline_to_medical": hybrid_movement,
        },
        "systems": systems,
    }

    out_file = Path(__file__).resolve().parent / "retrieval_medical_tokenizer_results.json"
    out_file.write_text(json.dumps(results_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n💾 Benchmark results saved to: {out_file}")

    # =====================================================
    # TERMINAL SUMMARY TABLE
    # =====================================================
    print(f"\n{'═' * 90}")
    print(f"  CONTROLLED MEDICAL BM25 TOKENIZER BENCHMARK SUMMARY")
    print(f"  Corpus: Baseline 800/150 (4,239 chunks) | Queries: 50 | Hybrid α=0.7 | RRF k=60")
    print(f"{'═' * 90}")
    print(f"  {'Configuration':<38} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG@10':>9}  {'T1 MRR':>7} {'T2 MRR':>7} {'T3 MRR':>7}")
    print(f"  {'─' * 88}")
    for s in systems:
        tm = s["tier_mrr"]
        print(
            f"  {s['name']:<38} "
            f"{s['recall@5']:>6.3f} "
            f"{s['recall@10']:>6.3f} "
            f"{s['mrr']:>6.3f} "
            f"{s['ndcg@10']:>9.3f}  "
            f"{tm['1']:>7.3f} {tm['2']:>7.3f} {tm['3']:>7.3f}"
        )
    print(f"{'═' * 90}")

    # Movement summary
    print(f"\n{'─' * 90}")
    print(f"  QUERY-LEVEL MOVEMENT BREAKDOWN")
    print(f"{'─' * 90}")
    for mov in [sparse_movement, hybrid_movement]:
        print(f"  ▶ {mov['comparison']}:")
        print(f"      Tied       (=) : {mov['tied_count']}/{mov['total_queries']} ({mov['tied_pct']}%)  [T1: {mov['tied_tier_breakdown']['tier_1']}, T2: {mov['tied_tier_breakdown']['tier_2']}, T3: {mov['tied_tier_breakdown']['tier_3']}]")
        print(f"      Improved   (▲) : {mov['improved_count']}/{mov['total_queries']} ({mov['improved_pct']}%)  [T1: {mov['improved_tier_breakdown']['tier_1']}, T2: {mov['improved_tier_breakdown']['tier_2']}, T3: {mov['improved_tier_breakdown']['tier_3']}]")
        print(f"      Regressed  (▼) : {mov['regressed_count']}/{mov['total_queries']} ({mov['regressed_pct']}%)  [T1: {mov['regressed_tier_breakdown']['tier_1']}, T2: {mov['regressed_tier_breakdown']['tier_2']}, T3: {mov['regressed_tier_breakdown']['tier_3']}]")
        
        if mov["improved_queries"]:
            print("      Key Improved Queries:")
            for q in mov["improved_queries"][:5]:
                print(f"        • [T{q['tier']}] \"{q['query'][:60]}...\" Rank: {q['base_rank']} -> {q['exp_rank']} (MRR: {q['base_mrr']:.3f} -> {q['exp_mrr']:.3f})")
        if mov["regressed_queries"]:
            print("      Key Regressed Queries:")
            for q in mov["regressed_queries"][:5]:
                print(f"        • [T{q['tier']}] \"{q['query'][:60]}...\" Rank: {q['base_rank']} -> {q['exp_rank']} (MRR: {q['base_mrr']:.3f} -> {q['exp_mrr']:.3f})")
        print()
    print(f"{'═' * 90}\n")


if __name__ == "__main__":
    run_benchmark()
