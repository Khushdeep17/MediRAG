import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

import nltk
from nltk.corpus import stopwords


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "merck_chunks_800_150.json"
DEFAULT_TOP_K = 10


# ===================================================
# LOAD DATA
# ===================================================

print("📄 Loading chunks...")
chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

documents = [chunk["content"] for chunk in chunks]


# ===================================================
# TOKENIZATION
# ===================================================

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


print("🧠 Tokenizing corpus...")
tokenized_corpus = [tokenize(doc) for doc in documents]


# ===================================================
# BUILD BM25
# ===================================================

print("🏗 Building BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)
print("✅ BM25 ready.")


# ===================================================
# SPARSE SEARCH
# ===================================================

def sparse_search(query: str, top_k: int = DEFAULT_TOP_K, return_results: bool = False):

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]

        result = {
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "chapter_number": chunk["chapter_number"],
            "chapter_title": chunk["chapter_title"],
            "content": chunk["content"],
            "score": float(scores[idx]),
        }

        results.append(result)

    if return_results:
        return results

    # CLI mode
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
# TEST
# ===================================================

if __name__ == "__main__":
    test_query = "H. pylori eradication therapy"
    sparse_search(test_query)