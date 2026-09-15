import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set
import nltk
from nltk.corpus import stopwords
import numpy as np
from rank_bm25 import BM25Okapi

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPARSE_PRESETS = {
    "baseline": {
        "chunks_path": PROJECT_ROOT / "data" / "processed" / "merck_chunks_800_150.json",
    },
    "experiment_450_64": {
        "chunks_path": PROJECT_ROOT / "data" / "processed" / "merck_chunks_450_64.json",
    },
}

DEFAULT_CHUNKS_PATH = SPARSE_PRESETS["baseline"]["chunks_path"]
DEFAULT_TOP_K = 10


# ===================================================
# SPARSE RETRIEVER (LAZY INITIALIZATION)
# ===================================================

class SparseRetriever:
    """Lazy-loaded BM25 sparse retriever."""

    def __init__(self, chunks_path: Path = DEFAULT_CHUNKS_PATH):
        self.chunks_path = Path(chunks_path)
        self._chunks: Optional[List[Dict[str, Any]]] = None
        self._documents: Optional[List[str]] = None
        self._bm25: Optional[BM25Okapi] = None
        self._stopwords: Optional[Set[str]] = None

    def _get_stopwords(self) -> Set[str]:
        """Load English stopwords on demand."""
        if self._stopwords is None:
            try:
                self._stopwords = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                self._stopwords = set(stopwords.words("english"))
        return self._stopwords

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using current baseline regex and stopword filtering."""
        stop_words = self._get_stopwords()
        text_lower = text.lower()
        text_clean = re.sub(r"[^a-z0-9\s]", " ", text_lower)
        tokens = text_clean.split()
        return [t for t in tokens if t not in stop_words]

    def _load_index(self) -> None:
        """Load chunk data, tokenize corpus, and construct BM25 index on demand."""
        if self._bm25 is None or self._chunks is None:
            if not self.chunks_path.exists():
                raise FileNotFoundError(
                    f"❌ Chunks data not found at: {self.chunks_path}. "
                    "Please verify preprocessing outputs."
                )

            print("📄 Loading chunks...")
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)

            self._documents = [chunk["content"] for chunk in self._chunks]

            print("🧠 Tokenizing corpus...")
            tokenized_corpus = [self.tokenize(doc) for doc in self._documents]

            print("🏗 Building BM25 index...")
            self._bm25 = BM25Okapi(tokenized_corpus)
            print("✅ BM25 ready.")

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        return_results: bool = False,
    ) -> List[Dict[str, Any]]:
        """Perform BM25 sparse search over the tokenized corpus."""
        self._load_index()
        assert self._bm25 is not None and self._chunks is not None

        tokenized_query = self.tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._chunks[idx]
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

        # CLI mode (pretty print)
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
# GLOBAL SINGLETONS & CONVENIENCE FUNCTIONS
# ===================================================

_SPARSE_RETRIEVERS: Dict[str, SparseRetriever] = {}


def get_sparse_retriever(preset: str = "baseline") -> SparseRetriever:
    """Return or initialize a SparseRetriever singleton for the given preset."""
    global _SPARSE_RETRIEVERS
    if preset not in _SPARSE_RETRIEVERS:
        if preset not in SPARSE_PRESETS:
            raise ValueError(f"Unknown preset: '{preset}'. Available presets: {list(SPARSE_PRESETS.keys())}")
        config = SPARSE_PRESETS[preset]
        _SPARSE_RETRIEVERS[preset] = SparseRetriever(chunks_path=config["chunks_path"])
    return _SPARSE_RETRIEVERS[preset]


def tokenize(text: str, preset: str = "baseline") -> List[str]:
    """Tokenize text using the specified SparseRetriever preset singleton."""
    return get_sparse_retriever(preset=preset).tokenize(text)


def sparse_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    return_results: bool = False,
    preset: str = "baseline",
) -> List[Dict[str, Any]]:
    """Execute sparse search using the specified SparseRetriever preset singleton."""
    return get_sparse_retriever(preset=preset).search(query, top_k=top_k, return_results=return_results)


# ===================================================
# TEST
# ===================================================

if __name__ == "__main__":
    test_query = "H. pylori eradication therapy"
    print("Testing baseline preset:")
    sparse_search(test_query)