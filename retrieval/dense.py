import json
import numpy as np
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import os
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===================================================
# CONFIG
# ===================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"

INDEX_PATH  = PROJECT_ROOT / "index" / "faiss.index"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "merck_chunks_800_150.json"
IDS_PATH    = PROJECT_ROOT / "embeddings" / "ids.json"

DEFAULT_TOP_K = 10
DEVICE = "cpu"


# ===================================================
# LOAD MODEL (loaded once)
# ===================================================

print("🔍 Loading dense model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
model.to(DEVICE)
model.eval()
print("✅ Dense model loaded.")


# ===================================================
# LOAD INDEX + METADATA
# ===================================================

print("📦 Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("📄 Loading chunks + id mapping...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(IDS_PATH, "r", encoding="utf-8") as f:
    id_map = json.load(f)

chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

print(f"📊 Total indexed vectors: {index.ntotal}")


# ===================================================
# EMBEDDING UTILITIES
# ===================================================

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def encode_query(query: str):

    query = "query: " + query  # Required for BGE

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs["attention_mask"])

    embeddings = embeddings.cpu().numpy().astype("float32")
    faiss.normalize_L2(embeddings)

    return embeddings


# ===================================================
# DENSE SEARCH
# ===================================================

def dense_search(query: str, top_k: int = DEFAULT_TOP_K, return_results: bool = False):

    query_vec = encode_query(query)
    scores, indices = index.search(query_vec, top_k)

    results = []

    for rank, (faiss_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):

        chunk_id = id_map[faiss_idx]
        chunk = chunk_lookup[chunk_id]

        result = {
            "rank": rank,
            "chunk_id": chunk_id,
            "chapter_number": chunk["chapter_number"],
            "chapter_title": chunk["chapter_title"],
            "content": chunk["content"],
            "score": float(score),
        }

        results.append(result)

    if return_results:
        return results

    # CLI Mode (pretty print)
    print("\n🔎 Query:", query)
    print("=" * 70)

    for r in results:
        preview = r["content"][:300].replace("\n", " ")
        print(f"\nRank {r['rank']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Chapter: {r['chapter_number']} - {r['chapter_title']}")
        print(f"Preview: {preview}...")

    return results


# ===================================================
# TEST MODE
# ===================================================

if __name__ == "__main__":
    test_query = "What are the causes and treatment of migraine?"
    dense_search(test_query)