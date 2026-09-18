"""
run_retrieval_benchmark.py

Thin CLI runner for executing MediRAG retrieval benchmarks.
Loads versioned query sets from evaluation/datasets/ and orchestrates
dense, sparse, and hybrid retrieval evaluations.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
from retrieval.dense import dense_search
from retrieval.sparse import sparse_search
from retrieval.fusion import hybrid_search
from evaluation.retrieval_metrics import (
    load_retrieval_dataset,
    load_retrieval_queries,
    evaluate_system,
    analyze_regressions,
    print_comparison_table,
    RETRIEVAL_K,
)

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "retrieval_queries_v2.json"


def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET,
    mode: str = "baseline",
    preset: str = "baseline",
    alpha: float = 0.7,
    k: int = RETRIEVAL_K,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute retrieval benchmark for the specified mode and dataset with full metadata."""
    dataset_data = load_retrieval_dataset(dataset_path)
    queries = dataset_data.get("queries", [])
    dataset_version = dataset_data.get("dataset_version", "2.0")
    dataset_name = dataset_data.get("dataset_name", "MediRAG Retrieval Benchmark v2")

    print(f"Loaded {len(queries)} evaluation queries from: {dataset_path} (Version: {dataset_version})")
    print(f"Mode: {mode} | Preset: {preset} | α: {alpha} | Depth K: {k}")

    all_results = []

    if mode in ("baseline", "compare"):
        print("\n--- Evaluating Baseline (800/150) ---")
        base_dense = evaluate_system(
            lambda q, return_results=True: dense_search(q, top_k=k, return_results=True, preset="baseline"),
            "Baseline Dense (800/150)",
            queries=queries,
            verbose=verbose,
            k=k,
        )
        base_sparse = evaluate_system(
            lambda q, return_results=True: sparse_search(q, top_k=k, return_results=True, preset="baseline"),
            "Baseline Sparse BM25 (800/150)",
            queries=queries,
            verbose=verbose,
            k=k,
        )
        base_hybrid = evaluate_system(
            lambda q, return_results=True: hybrid_search(q, alpha=alpha, top_k=k, return_results=True, preset="baseline"),
            f"Baseline Hybrid α={alpha} (800/150)",
            queries=queries,
            verbose=verbose,
            k=k,
        )
        all_results.extend([base_dense, base_sparse, base_hybrid])

    if mode in ("experiment", "compare"):
        print(f"\n--- Evaluating Experiment ({preset}) ---")
        exp_dense = evaluate_system(
            lambda q, return_results=True: dense_search(q, top_k=k, return_results=True, preset=preset),
            f"Experiment Dense ({preset})",
            queries=queries,
            verbose=verbose,
            k=k,
        )
        exp_sparse = evaluate_system(
            lambda q, return_results=True: sparse_search(q, top_k=k, return_results=True, preset=preset),
            f"Experiment Sparse ({preset})",
            queries=queries,
            verbose=verbose,
            k=k,
        )
        exp_hybrid = evaluate_system(
            lambda q, return_results=True: hybrid_search(q, alpha=alpha, top_k=k, return_results=True, preset=preset),
            f"Experiment Hybrid α={alpha} ({preset})",
            queries=queries,
            verbose=verbose,
            k=k,
        )
        all_results.extend([exp_dense, exp_sparse, exp_hybrid])

    print_comparison_table(all_results)

    if mode == "compare" and len(all_results) >= 6:
        analyze_regressions(all_results[0], all_results[3], system_label="Dense Retrieval")
        analyze_regressions(all_results[2], all_results[5], system_label=f"Hybrid Retrieval (α={alpha})")

    # Run metadata packaging
    now_iso = datetime.now().isoformat()
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_file = (
        "data/processed/merck_chunks_800_150.json"
        if preset == "baseline"
        else "data/processed/merck_chunks_450_64.json"
    )

    summary_metrics = {
        r["name"]: {
            "modern_metrics": r.get("modern_metrics", {}),
            "legacy_metrics": r.get("legacy_metrics", {}),
            "tier_summary": r.get("tier_summary", {}),
        }
        for r in all_results
    }

    report_payload = {
        "benchmark_name": "MediRAG Retrieval Benchmark v2",
        "timestamp": now_iso,
        "dataset_metadata": {
            "dataset_path": str(dataset_path),
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "query_count": len(queries),
        },
        "retrieval_configuration": {
            "mode": mode,
            "preset": preset,
            "fusion_alpha": alpha,
            "retrieval_depth_k": k,
            "rrf_constant": 60,
            "dense_model": "BAAI/bge-large-en-v1.5",
            "dense_index": "models/indices/faiss_flat_l2.index",
            "sparse_model": "BM25Okapi (baseline tokenizer)",
            "sparse_index": "models/indices/bm25_index.pkl",
            "chunk_dataset": chunk_file,
        },
        "summary_metrics": summary_metrics,
        "systems_evaluated": all_results,
    }

    out_file = output_path or (
        PROJECT_ROOT / "evaluation" / "results" / "current" / f"retrieval_benchmark_v2_{now_tag}.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)
    print(f"💾 Fresh retrieval benchmark results saved to: {out_file}")

    # Also update stable pointer in current/
    stable_file = PROJECT_ROOT / "evaluation" / "results" / "current" / "retrieval_benchmark_results.json"
    try:
        with open(stable_file, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2, ensure_ascii=False)
        print(f"💾 Updated active pointer: {stable_file}")
    except Exception as e:
        print(f"⚠️ Could not write active pointer: {e}")

    return report_payload


def main():
    parser = argparse.ArgumentParser(description="MediRAG Retrieval Benchmark Runner")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to retrieval query JSON dataset")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "experiment", "compare"], help="Benchmark execution mode")
    parser.add_argument("--preset", type=str, default="baseline", choices=["baseline", "experiment_450_64"], help="Corpus chunk preset")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid fusion alpha weight (default: 0.7)")
    parser.add_argument("--k", type=int, default=RETRIEVAL_K, help="Retrieval depth top-k (default: 30)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print per-query results")
    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        mode=args.mode,
        preset=args.preset,
        alpha=args.alpha,
        k=args.k,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
