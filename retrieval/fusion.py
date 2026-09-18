from typing import Any, Dict, List, Optional
import numpy as np

from retrieval.dense import dense_search
from retrieval.sparse import sparse_search


# =====================================================
# CONFIG
# =====================================================

DEFAULT_TOP_K = 10
RRF_K = 60  # constant for Reciprocal Rank Fusion


# =====================================================
# HYBRID SEARCH (Weighted RRF Fusion)
# =====================================================

def hybrid_search(
    query: str,
    alpha: float = 0.5,
    top_k: int = DEFAULT_TOP_K,
    return_results: bool = False,
    verbose: bool = False,
    preset: str = "baseline",
    tokenizer: str = "baseline",
    dense_fn: Any = None,
    sparse_fn: Any = None,
    candidate_depth: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid retrieval using weighted Reciprocal Rank Fusion (RRF).

    alpha = weight for dense
    (1 - alpha) = weight for sparse
    preset = 'baseline' (800/150) or 'experiment_450_64' (450/64)
    tokenizer = 'baseline' or 'medical'
    candidate_depth = number of candidates to pull from dense/sparse before fusion (defaults to top_k * 2)
    """
    depth = candidate_depth if candidate_depth is not None else top_k * 2

    # -------- Retrieve from both systems --------
    if dense_fn is not None:
        dense_results = dense_fn(query, top_k=depth, return_results=True)
    else:
        dense_results = dense_search(query, top_k=depth, return_results=True, preset=preset)

    if sparse_fn is not None:
        sparse_results = sparse_fn(query, top_k=depth, return_results=True)
    else:
        sparse_results = sparse_search(
            query, top_k=depth, return_results=True, preset=preset, tokenizer=tokenizer
        )

    # -------- Build rank lookup --------
    fusion_scores = {}

    # Dense contribution
    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["chunk_id"]
        score = alpha * (1 / (RRF_K + rank))
        fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0) + score

    # Sparse contribution
    for rank, item in enumerate(sparse_results, start=1):
        chunk_id = item["chunk_id"]
        score = (1 - alpha) * (1 / (RRF_K + rank))
        fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0) + score

    # -------- Sort by fusion score --------
    ranked_chunk_ids = sorted(
        fusion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_chunk_ids = ranked_chunk_ids[:top_k]

    # -------- Build final result list --------
    final_results = []

    # Use dense result metadata as canonical source
    dense_lookup = {r["chunk_id"]: r for r in dense_results}
    sparse_lookup = {r["chunk_id"]: r for r in sparse_results}

    for rank, (chunk_id, fusion_score) in enumerate(ranked_chunk_ids, start=1):

        if chunk_id in dense_lookup:
            base = dense_lookup[chunk_id]
        else:
            base = sparse_lookup[chunk_id]

        result = {
            "rank": rank,
            "chunk_id": chunk_id,
            "chapter_number": base["chapter_number"],
            "chapter_title": base["chapter_title"],
            "content": base["content"],
            "score": float(fusion_score),
        }

        final_results.append(result)

    if return_results:
        return final_results

    # CLI Mode
    print("\n🔎 Query:", query)
    print("=" * 70)

    for r in final_results:
        preview = r["content"][:300].replace("\n", " ")
        print(f"\nRank {r['rank']}")
        print(f"Fusion Score: {r['score']:.6f}")
        print(f"Chapter: {r['chapter_number']} - {r['chapter_title']}")
        print(f"Preview: {preview}...")

    return final_results


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    test_query = "What are the manifestations of acute pancreatitis?"
    hybrid_search(test_query, alpha=0.6)