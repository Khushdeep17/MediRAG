import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


# ===================================================
# CONFIG
# ===================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 8
DEVICE = "cpu"

CHUNKS_PATH = Path("data/processed/merck_chunks_800_150.json")

EMBEDDINGS_DIR = Path("embeddings")
INDEX_DIR = Path("index")

EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "ids.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================
# MODEL LOADING
# ===================================================

def load_model():
    print("🔍 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded.")
    return tokenizer, model


# ===================================================
# EMBEDDING UTILITIES
# ===================================================

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def embed_batch(texts, tokenizer, model):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    return embeddings.cpu().numpy()


# ===================================================
# MAIN INDEXING PIPELINE
# ===================================================

def main():

    # Load chunks
    print("📘 Loading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Deterministic ordering
    chunks = sorted(chunks, key=lambda x: x["chunk_id"])

    print(f"📄 Total chunks loaded: {len(chunks)}")

    tokenizer, model = load_model()

    all_embeddings = []
    id_mapping = []

    print("⚡ Generating embeddings...")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]

        # BGE passage prefix
        texts = [f"passage: {chunk['content']}" for chunk in batch]

        batch_embeddings = embed_batch(texts, tokenizer, model)

        all_embeddings.append(batch_embeddings)
        id_mapping.extend([chunk["chunk_id"] for chunk in batch])

    embeddings_matrix = np.vstack(all_embeddings).astype("float32")

    print(f"📏 Embedding matrix shape: {embeddings_matrix.shape}")

    # Normalize for cosine similarity
    print("📏 Normalizing embeddings (L2)...")
    faiss.normalize_L2(embeddings_matrix)

    # Save raw embeddings
    print("💾 Saving embeddings.npy...")
    np.save(EMBEDDINGS_PATH, embeddings_matrix)

    print("💾 Saving ids.json...")
    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f, indent=2)

    # Build FAISS index
    print("🏗️ Building FAISS IndexFlatIP...")
    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    print("💾 Saving FAISS index...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print("✅ Dense indexing complete.")
    print(f"📦 Total vectors indexed: {index.ntotal}")


# ===================================================
# ENTRYPOINT
# ===================================================

if __name__ == "__main__":
    main()