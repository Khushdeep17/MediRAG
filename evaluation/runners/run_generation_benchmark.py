"""
run_generation_benchmark.py

Thin CLI runner for executing MediRAG generation benchmarks.
Loads versioned query sets from evaluation/datasets/ and orchestrates
retrieval + synthesis using the full 2,200c evidence contract.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.generation_eval import (
    load_generation_queries,
    chunk_to_dict,
    extract_cited_numbers,
)

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "generation_queries_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "current" / "generation_outputs.json"


def run_generation_benchmark(
    dataset_path: Path = DEFAULT_DATASET,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Execute generation benchmark over the query set."""
    queries = load_generation_queries(dataset_path)
    if limit is not None:
        queries = queries[:limit]

    print(f"Loaded {len(queries)} generation evaluation queries from: {dataset_path}")
    out_file = output_path or DEFAULT_OUTPUT

    if dry_run:
        print("🔍 Dry-run mode: Query set validated. Generation skipped.")
        return []

    # Defer import of generate_answer until runtime so dry-runs / help flags don't trigger heavy imports
    from generate import generate_answer

    all_results = []
    print(f"\nStarting generation benchmark ({len(queries)} queries)...")

    for i, q in enumerate(queries, 1):
        query_text = q["query"]
        expected_chap = q.get("expected_chapter") or q.get("relevant_chapter")
        tier = q["tier"]

        print(f"[{i:02d}/{len(queries):02d}] T{tier} (Ch.{expected_chap}): \"{query_text[:55]}...\"")

        try:
            answer, retrieved_chunks = generate_answer(query_text)
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            answer = f"ERROR: {e}"
            retrieved_chunks = []

        cited_numbers = extract_cited_numbers(answer)
        serialized_chunks = [chunk_to_dict(c, idx) for idx, c in enumerate(retrieved_chunks, 1)]
        retrieved_chapters = [c.get("chapter_number") for c in retrieved_chunks if c.get("chapter_number") is not None]

        chunk_by_idx = {c["source_index"]: c for c in serialized_chunks}
        cited_actual_chapters = [
            chunk_by_idx[n]["chapter_number"]
            for n in cited_numbers
            if n in chunk_by_idx and chunk_by_idx[n]["chapter_number"] is not None
        ]
        expected_cited = expected_chap in cited_actual_chapters

        citation_attribution = {}
        for num in cited_numbers:
            if num in chunk_by_idx:
                c = chunk_by_idx[num]
                citation_attribution[str(num)] = {
                    "source_index": num,
                    "chunk_id": c["chunk_id"],
                    "chapter_number": c["chapter_number"],
                    "chapter_title": c["chapter_title"],
                    "section_title": c["section_title"],
                    "content": c["content"],
                }
            else:
                citation_attribution[str(num)] = {
                    "source_index": num,
                    "error": "Citation number out of bounds (hallucinated citation index)",
                }

        record = {
            "query_id": q.get("id", i),
            "query": query_text,
            "tier": tier,
            "relevant_chapter": expected_chap,
            "expected_chapter": expected_chap,
            "generated_answer": answer,
            "retrieved_chunks": serialized_chunks,
            "retrieved_chapter_numbers": retrieved_chapters,
            "cited_source_indices": cited_numbers,
            "cited_chapter_numbers": cited_actual_chapters,
            "citation_attribution": citation_attribution,
            "expected_chapter_cited": expected_cited,
            "timestamp": str(datetime.utcnow()),
            "manual_accuracy_grade": None,
            "llm_faithfulness": None,
            "llm_completeness": None,
            "llm_medical_accuracy": None,
        }
        all_results.append(record)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Benchmark completed. Saved {len(all_results)} records → {out_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="MediRAG Generation Benchmark Runner")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to generation query JSON dataset")
    parser.add_argument("--output", type=Path, default=None, help="Path to save generation outputs JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to run")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and exit without calling generation")
    args = parser.parse_args()

    run_generation_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
