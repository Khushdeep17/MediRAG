import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------

CHUNKS_PATH = Path("data/processed/merck_chunks_800_150.json")
TOP_K = 5

# -----------------------------
# LOAD CHUNKS
# -----------------------------

print("📄 Loading chunks...")
chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

documents = [chunk["content"] for chunk in chunks]

# -----------------------------
# TOKENIZATION
# -----------------------------

import re
from nltk.corpus import stopwords
import nltk

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

STOPWORDS = set(stopwords.words("english"))

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

print("🧠 Tokenizing corpus...")
tokenized_corpus = [tokenize(doc) for doc in documents]

print("🏗 Building BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)

print("✅ BM25 ready.")

# -----------------------------
# SEARCH FUNCTION
# -----------------------------

def search(query: str):
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    print("\n🔎 Query:", query)
    print("=" * 70)

    for rank, idx in enumerate(top_indices, 1):
        chunk = chunks[idx]
        preview = chunk["content"][:300].replace("\n", " ")

        print(f"\nRank {rank}")
        print(f"Score: {scores[idx]:.4f}")
        print(f"Chapter: {chunk['chapter_number']} - {chunk['chapter_title']}")
        print(f"Preview: {preview}...")


# -----------------------------
# TEST
# -----------------------------

if __name__ == "__main__":

    test_query = "H. pylori eradication therapy"
    search(test_query)