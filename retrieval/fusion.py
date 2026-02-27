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
):
    """
    alpha = weight for dense
    (1 - alpha) = weight for sparse
    """

    # -------- Retrieve from both systems --------
    dense_results = dense_search(query, top_k=top_k * 2, return_results=True)
    sparse_results = sparse_search(query, top_k=top_k * 2, return_results=True)

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

    test_query = "What are the causes and treatment of migraine?"
    hybrid_search(test_query, alpha=0.6)