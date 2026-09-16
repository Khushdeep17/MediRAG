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
# MEDICAL NORMALIZATION PATTERNS
# ===================================================

GREEK_MAP = {
    'α': 'alpha', 'Α': 'alpha',
    'β': 'beta',  'Β': 'beta',
    'γ': 'gamma', 'Γ': 'gamma',
    'δ': 'delta', 'Δ': 'delta',
    'κ': 'kappa', 'Κ': 'kappa',
}

# Genus-species abbreviations: e.g. H. pylori, C. difficile, E. coli, S. aureus
# Matches single-letter genus followed by dot and species name (at least 3 characters)
GENUS_SPECIES_PATTERN = re.compile(r'(?<![a-zA-Z\.])([a-zA-Z])\.\s*([a-zA-Z]{3,})\b')

# Decimal numbers: e.g. 2.5, 0.5, 12.5
DECIMAL_PATTERN = re.compile(r'\b\d+\.\d+\b')

# Slash units/fractions: e.g. mg/dL, mL/min
UNIT_SLASH_PATTERN = re.compile(r'\b([a-zA-Z]+)/([a-zA-Z]+)\b')

# Hyphenated medical compound pattern: e.g. covid-19, type-2, beta-blocker, tnf-alpha, il-6, ldl-c
HYPHEN_PATTERN = re.compile(r'\b([a-zA-Z0-9]+)-([a-zA-Z0-9]+)\b')


# ===================================================
# SPARSE RETRIEVER (LAZY INITIALIZATION)
# ===================================================

class SparseRetriever:
    """Lazy-loaded BM25 sparse retriever with selectable tokenizers."""

    def __init__(self, chunks_path: Path = DEFAULT_CHUNKS_PATH, tokenizer: str = "baseline"):
        self.chunks_path = Path(chunks_path)
        self.tokenizer = tokenizer
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

    def tokenize_baseline(self, text: str) -> List[str]:
        """Tokenize text using current baseline regex and stopword filtering (intact)."""
        stop_words = self._get_stopwords()
        text_lower = text.lower()
        text_clean = re.sub(r"[^a-z0-9\s]", " ", text_lower)
        tokens = text_clean.split()
        return [t for t in tokens if t not in stop_words]

    def tokenize_medical(self, text: str) -> List[str]:
        """
        Tokenize text with medical-aware normalization:
        - Transliterates Greek characters (α->alpha, β->beta, etc.)
        - Normalizes genus-species abbreviations without orphan letters (H. pylori -> hpylori, pylori)
        - Preserves decimal numbers (2.5 -> '2.5')
        - Dual-emits hyphenated compounds without orphan numeric noise (COVID-19 -> 'covid-19', 'covid')
        - Normalizes slash expressions (mg/dL -> 'mg_dl', 'mg', 'dl')
        - Preserves alphanumeric tokens (HbA1c, G6PD, B12)
        - Applies NLTK stopwords and eliminates orphan single-letter noise
        """
        stop_words = self._get_stopwords()

        # 1. Greek letter normalization
        for g_char, replacement in GREEK_MAP.items():
            if g_char in text:
                text = text.replace(g_char, replacement)

        # 2. Lowercase
        text = text.lower()

        # 3. Filter out generic Latin editorial abbreviations (e.g. / i.e.)
        text = re.sub(r'\b(?:e\.g\.|i\.e\.)\b', ' ', text)

        # 4. Normalize Genus. species abbreviations
        # E.g. "h. pylori" -> "h_pylori pylori" (emits compound + species, no orphan 'h')
        text = GENUS_SPECIES_PATTERN.sub(r'\1_\2 \2', text)

        # 5. Normalize slash units / expressions
        # E.g. "mg/dl" -> "mg_dl mg dl"
        text = UNIT_SLASH_PATTERN.sub(r'\1_\2 \1 \2', text)

        # 6. Protect Decimal Numbers
        # Replace '.' in decimals with a placeholder so it survives punctuation cleanup
        decimals = set(DECIMAL_PATTERN.findall(text))
        for dec in decimals:
            placeholder = dec.replace('.', '_dec_')
            text = text.replace(dec, placeholder)

        # 7. Dual-emission for hyphenated terms
        def replace_hyphen(m):
            left, right = m.group(1), m.group(2)
            compound = f"{left}-{right}"
            # If right is numeric (e.g. covid-19, type-2, il-6):
            if right.isdigit():
                # Emit compound + alpha constituent (DO NOT emit bare number noise)
                if left.isalpha() and len(left) > 1:
                    return f"{compound} {left}"
                return compound
            # If left is numeric:
            if left.isdigit():
                if right.isalpha() and len(right) > 1:
                    return f"{compound} {right}"
                return compound
            # Both are alphabetic: emit compound + both constituents (if > 1 char)
            parts = [compound]
            if len(left) > 1:
                parts.append(left)
            if len(right) > 1:
                parts.append(right)
            return " ".join(parts)

        text = HYPHEN_PATTERN.sub(replace_hyphen, text)

        # 8. Clean all remaining punctuation (retain alphanumeric, hyphens, underscores)
        text_clean = re.sub(r"[^a-z0-9_\-\s]", " ", text)

        # 9. Split and restore decimals, filter stopwords and orphan single-characters
        tokens = []
        for t in text_clean.split():
            if "_dec_" in t:
                t = t.replace("_dec_", ".")

            # Stopword filter
            if t in stop_words:
                continue

            # Eliminate orphan single-letter noise
            if len(t) <= 1:
                continue

            tokens.append(t)

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using the selected tokenizer strategy ('baseline' or 'medical')."""
        if self.tokenizer == "medical":
            return self.tokenize_medical(text)
        return self.tokenize_baseline(text)

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


def get_sparse_retriever(preset: str = "baseline", tokenizer: str = "baseline") -> SparseRetriever:
    """Return or initialize a SparseRetriever singleton for the given preset and tokenizer strategy."""
    global _SPARSE_RETRIEVERS
    cache_key = f"{preset}_{tokenizer}"
    if cache_key not in _SPARSE_RETRIEVERS:
        if preset not in SPARSE_PRESETS:
            raise ValueError(f"Unknown preset: '{preset}'. Available presets: {list(SPARSE_PRESETS.keys())}")
        config = SPARSE_PRESETS[preset]
        _SPARSE_RETRIEVERS[cache_key] = SparseRetriever(
            chunks_path=config["chunks_path"],
            tokenizer=tokenizer,
        )
    return _SPARSE_RETRIEVERS[cache_key]


def tokenize(text: str, preset: str = "baseline", tokenizer: str = "baseline") -> List[str]:
    """Tokenize text using the specified SparseRetriever preset and tokenizer strategy."""
    return get_sparse_retriever(preset=preset, tokenizer=tokenizer).tokenize(text)


def tokenize_medical(text: str) -> List[str]:
    """Convenience function for direct medical-aware tokenization."""
    return get_sparse_retriever(preset="baseline", tokenizer="medical").tokenize_medical(text)


def sparse_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    return_results: bool = False,
    preset: str = "baseline",
    tokenizer: str = "baseline",
) -> List[Dict[str, Any]]:
    """Execute sparse search using the specified SparseRetriever preset and tokenizer strategy."""
    return get_sparse_retriever(preset=preset, tokenizer=tokenizer).search(
        query, top_k=top_k, return_results=return_results
    )


# ===================================================
# TEST
# ===================================================

if __name__ == "__main__":
    test_query = "H. pylori eradication therapy"
    print("Testing baseline preset (baseline tokenizer):")
    sparse_search(test_query, tokenizer="baseline")
    print("\nTesting baseline preset (medical tokenizer):")
    sparse_search(test_query, tokenizer="medical")