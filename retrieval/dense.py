import json
import numpy as np
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch


# ===================================================
# CONFIG
# ===================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"

INDEX_PATH = Path("index/faiss.index")
IDS_PATH = Path("embeddings/ids.json")
CHUNKS_PATH = Path("data/processed/merck_chunks_800_150.json")

TOP_K = 5
DEVICE = "cpu"


# ===================================================
# LOAD MODEL
# ===================================================

print("🔍 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)
model.to(DEVICE)
model.eval()

print("✅ Model loaded.")


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

# Create fast lookup dictionary
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
    query = "query: " + query  # IMPORTANT for BGE

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
# SEARCH
# ===================================================

def search(query: str):

    query_vec = encode_query(query)
    scores, indices = index.search(query_vec, TOP_K)

    print("\n🔎 Query:", query)
    print("=" * 70)

    for rank, (faiss_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):

        chunk_id = id_map[faiss_idx]
        chunk = chunk_lookup[chunk_id]

        preview = chunk["content"][:300].replace("\n", " ")

        print(f"\nRank {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chapter: {chunk['chapter_number']} - {chunk['chapter_title']}")
        print(f"Preview: {preview}...")


# ===================================================
# TEST
# ===================================================

if __name__ == "__main__":

    test_query = "What are the causes and treatment of migraine?"
    search(test_query)