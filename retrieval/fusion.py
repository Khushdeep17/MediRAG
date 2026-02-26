import json
import numpy as np
import faiss
import re
import torch
from pathlib import Path
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords

# -----------------------------
# CONFIG
# -----------------------------

MODEL_NAME = "BAAI/bge-large-en-v1.5"

TOP_K = 5              # final results
RETRIEVAL_K = 20       # candidates from each retriever
RRF_K = 60             # RRF constant

WD = 0.6               # Dense weight
WS = 0.4               # Sparse weight

INDEX_PATH = Path("index/faiss.index")
CHUNKS_PATH = Path("data/processed/merck_chunks_800_150.json")
IDS_PATH = Path("embeddings/ids.json")

DEVICE = "cpu"

# -----------------------------
# LOAD CHUNKS + ID MAP
# -----------------------------

print("📄 Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("🆔 Loading FAISS id mapping...")
with open(IDS_PATH, "r", encoding="utf-8") as f:
    id_map = json.load(f)  # list: faiss_index -> chunk_id

# Fast lookups
chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
chunkid_to_faiss = {chunk_id: idx for idx, chunk_id in enumerate(id_map)}
documents = [chunk["content"] for chunk in chunks]

# -----------------------------
# LOAD DENSE MODEL + INDEX
# -----------------------------

print("🔍 Loading dense model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.to(DEVICE)
model.eval()

print("📦 Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

# -----------------------------
# BM25 BUILD
# -----------------------------

STOPWORDS = set(stopwords.words("english"))

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # removed numbers (less noise)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

print("🧠 Building BM25...")
tokenized_corpus = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# -----------------------------
# DENSE ENCODER
# -----------------------------

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )

def encode_query(query: str):
    query = "query: " + query

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

# -----------------------------
# HYBRID SEARCH (Weighted RRF)
# -----------------------------

def hybrid_search(query: str):

    # -------- Dense Retrieval --------
    dense_vec = encode_query(query)
    dense_scores, dense_indices = index.search(dense_vec, RETRIEVAL_K)

    # -------- Sparse Retrieval --------
    tokenized_query = tokenize(query)
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(sparse_scores)[::-1][:RETRIEVAL_K]

    # -------- Weighted RRF Fusion --------
    fusion_scores = {}

    # Dense contribution
    for rank, faiss_idx in enumerate(dense_indices[0], 1):
        score = WD * (1 / (RRF_K + rank))
        fusion_scores[faiss_idx] = fusion_scores.get(faiss_idx, 0) + score

    # Sparse contribution
    for rank, sparse_idx in enumerate(sparse_indices, 1):

        sparse_chunk_id = chunks[sparse_idx]["chunk_id"]

        if sparse_chunk_id in chunkid_to_faiss:
            faiss_idx = chunkid_to_faiss[sparse_chunk_id]
            score = WS * (1 / (RRF_K + rank))
            fusion_scores[faiss_idx] = fusion_scores.get(faiss_idx, 0) + score

    # Sort final ranking
    ranked = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = ranked[:TOP_K]

    # -------- Print Results --------
    print("\n🔎 Query:", query)
    print("=" * 70)

    for rank, (faiss_idx, score) in enumerate(top_results, 1):

        chunk_id = id_map[faiss_idx]
        chunk = chunk_lookup[chunk_id]
        preview = chunk["content"][:300].replace("\n", " ")

        print(f"\nRank {rank}")
        print(f"Fusion Score: {score:.6f}")
        print(f"Chapter: {chunk['chapter_number']} - {chunk['chapter_title']}")
        print(f"Preview: {preview}...")

# -----------------------------
# TEST
# -----------------------------

if __name__ == "__main__":

    test_query = "What are the causes and treatment of migraine?"
    hybrid_search(test_query)