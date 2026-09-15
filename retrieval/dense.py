import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===================================================
# CONFIG & PRESETS
# ===================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"

PRESETS = {
    "baseline": {
        "index_path": PROJECT_ROOT / "index" / "faiss.index",
        "chunks_path": PROJECT_ROOT / "data" / "processed" / "merck_chunks_800_150.json",
        "ids_path": PROJECT_ROOT / "embeddings" / "ids.json",
    },
    "experiment_450_64": {
        "index_path": PROJECT_ROOT / "index" / "merck_450_64.index",
        "chunks_path": PROJECT_ROOT / "data" / "processed" / "merck_chunks_450_64.json",
        "ids_path": PROJECT_ROOT / "embeddings" / "ids_450_64.json",
    },
}

DEFAULT_INDEX_PATH = PRESETS["baseline"]["index_path"]
DEFAULT_CHUNKS_PATH = PRESETS["baseline"]["chunks_path"]
DEFAULT_IDS_PATH = PRESETS["baseline"]["ids_path"]

DEFAULT_TOP_K = 10
DEVICE = "cpu"


# ===================================================
# DENSE RETRIEVER (LAZY INITIALIZATION)
# ===================================================

class DenseRetriever:
    """Lazy-loaded dense retriever using BGE embeddings and FAISS."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_path: Path = DEFAULT_INDEX_PATH,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        ids_path: Path = DEFAULT_IDS_PATH,
        device: str = DEVICE,
    ):
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.ids_path = Path(ids_path)
        self.device = device

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._index: Optional[faiss.Index] = None
        self._chunks: Optional[List[Dict[str, Any]]] = None
        self._id_map: Optional[List[str]] = None
        self._chunk_lookup: Optional[Dict[str, Dict[str, Any]]] = None

    def _load_model(self) -> None:
        """Load tokenizer and embedding model on demand."""
        if self._tokenizer is None or self._model is None:
            print("🔍 Loading dense model...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )
            self._model.to(self.device)
            self._model.eval()
            print("✅ Dense model loaded.")

    def _load_index(self) -> None:
        """Load FAISS index, chunk metadata, and ID mappings on demand."""
        if self._index is None or self._chunks is None or self._id_map is None:
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"❌ FAISS index not found at: {self.index_path}. "
                    "Please run `python indexing/dense_faiss.py` first."
                )
            if not self.chunks_path.exists():
                raise FileNotFoundError(
                    f"❌ Chunks data not found at: {self.chunks_path}. "
                    "Please verify preprocessing outputs."
                )
            if not self.ids_path.exists():
                raise FileNotFoundError(
                    f"❌ Chunk IDs mapping not found at: {self.ids_path}. "
                    "Please run `python indexing/dense_faiss.py` first."
                )

            print("📦 Loading FAISS index...")
            self._index = faiss.read_index(str(self.index_path))

            print("📄 Loading chunks + id mapping...")
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)

            with open(self.ids_path, "r", encoding="utf-8") as f:
                self._id_map = json.load(f)

            self._chunk_lookup = {chunk["chunk_id"]: chunk for chunk in self._chunks}
            print(f"📊 Total indexed vectors: {self._index.ntotal}")

    @staticmethod
    def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask weighting."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode and L2-normalize a query string into an embedding vector."""
        self._load_model()
        assert self._tokenizer is not None and self._model is not None

        query_formatted = "query: " + query  # Required for current BGE baseline

        inputs = self._tokenizer(
            query_formatted,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])

        embeddings_np = embeddings.cpu().numpy().astype("float32")
        faiss.normalize_L2(embeddings_np)
        return embeddings_np

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        return_results: bool = False,
    ) -> List[Dict[str, Any]]:
        """Perform dense semantic search against the FAISS index."""
        self._load_index()
        assert self._index is not None and self._id_map is not None and self._chunk_lookup is not None

        query_vec = self.encode_query(query)
        scores, indices = self._index.search(query_vec, top_k)

        results: List[Dict[str, Any]] = []
        for rank, (faiss_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            chunk_id = self._id_map[faiss_idx]
            chunk = self._chunk_lookup[chunk_id]

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
# GLOBAL SINGLETONS & CONVENIENCE FUNCTIONS
# ===================================================

_RETRIEVERS: Dict[str, DenseRetriever] = {}


def get_retriever(preset: str = "baseline") -> DenseRetriever:
    """Return or initialize a DenseRetriever singleton for the given preset."""
    global _RETRIEVERS
    if preset not in _RETRIEVERS:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: '{preset}'. Available presets: {list(PRESETS.keys())}")
        config = PRESETS[preset]
        _RETRIEVERS[preset] = DenseRetriever(
            index_path=config["index_path"],
            chunks_path=config["chunks_path"],
            ids_path=config["ids_path"],
        )
    return _RETRIEVERS[preset]


def encode_query(query: str, preset: str = "baseline") -> np.ndarray:
    """Encode query using the specified DenseRetriever preset singleton."""
    return get_retriever(preset=preset).encode_query(query)


def dense_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    return_results: bool = False,
    preset: str = "baseline",
) -> List[Dict[str, Any]]:
    """Execute dense search using the specified DenseRetriever preset singleton."""
    return get_retriever(preset=preset).search(query, top_k=top_k, return_results=return_results)


# ===================================================
# TEST MODE
# ===================================================

if __name__ == "__main__":
    test_query = "What are the causes and treatment of migraine?"
    print("Testing baseline preset:")
    dense_search(test_query)