import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Optional, List, Dict, Any, Set
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ===================================================
# CONFIG & DEFAULTS
# ===================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_THREADS = None
DEFAULT_CHECKPOINT_INTERVAL = 5
DEVICE = "cpu"

DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "merck_chunks_800_150.json"
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"

DEFAULT_EMBEDDINGS_PATH = DEFAULT_EMBEDDINGS_DIR / "embeddings.npy"
DEFAULT_IDS_PATH = DEFAULT_EMBEDDINGS_DIR / "ids.json"
DEFAULT_FAISS_INDEX_PATH = DEFAULT_INDEX_DIR / "faiss.index"


# ===================================================
# MODEL LOADING
# ===================================================

def load_model(device: str = DEVICE):
    print("🔍 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    model.to(device)
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


def embed_batch(texts: List[str], tokenizer, model, device: str = DEVICE) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    return embeddings.cpu().numpy().astype("float32")


# ===================================================
# CHECKPOINT MANAGEMENT (INCREMENTAL & ATOMIC)
# ===================================================

def _atomic_write_json(data: Dict[str, Any], path: Path) -> None:
    """Atomically write JSON data using a temporary file to prevent corruption."""
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, path)


def _load_and_validate_checkpoint(
    checkpoint_dir: Path,
    chunks_path: Path,
    total_chunks: int,
    batch_size: int,
) -> Optional[Dict[str, Any]]:
    """
    Validate checkpoint manifest against current configuration fingerprint.
    Returns manifest dictionary if valid, raises ValueError if corrupt/mismatched.
    """
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    print(f"🔍 Found checkpoint manifest at: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise ValueError(f"❌ Failed to parse checkpoint manifest at {manifest_path}: {e}")

    # Configuration fingerprint validation
    checks = [
        ("chunks_path", Path(manifest.get("chunks_path", "")).resolve() == chunks_path.resolve(),
         f"chunks_path mismatch: manifest has '{manifest.get('chunks_path')}' vs current '{chunks_path}'"),
        ("total_chunks", manifest.get("total_chunks") == total_chunks,
         f"total_chunks mismatch: manifest has {manifest.get('total_chunks')} vs current {total_chunks}"),
        ("batch_size", manifest.get("batch_size") == batch_size,
         f"batch_size mismatch: manifest has {manifest.get('batch_size')} vs current {batch_size}"),
        ("model_name", manifest.get("model_name") == MODEL_NAME,
         f"model_name mismatch: manifest has '{manifest.get('model_name')}' vs current '{MODEL_NAME}'"),
        ("embedding_dim", manifest.get("embedding_dim") == EMBEDDING_DIM,
         f"embedding_dim mismatch: manifest has {manifest.get('embedding_dim')} vs current {EMBEDDING_DIM}"),
    ]

    for name, ok, msg in checks:
        if not ok:
            raise ValueError(
                f"❌ Checkpoint validation failed [{name}]: {msg}. "
                "Refusing to resume from incompatible checkpoint."
            )

    # Verify batch part files exist on disk and are not empty
    completed_batches: List[int] = manifest.get("completed_batches", [])
    for b_idx in completed_batches:
        part_file = checkpoint_dir / f"batch_{b_idx:05d}.npy"
        if not part_file.exists():
            raise ValueError(
                f"❌ Checkpoint inconsistency: batch file '{part_file}' recorded in manifest but missing on disk!"
            )
        if part_file.stat().st_size == 0:
            raise ValueError(
                f"❌ Checkpoint inconsistency: batch file '{part_file}' is 0 bytes (corrupted)!"
            )

    return manifest


# ===================================================
# MAIN INDEXING PIPELINE
# ===================================================

def build_dense_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    ids_path: Path = DEFAULT_IDS_PATH,
    index_path: Path = DEFAULT_FAISS_INDEX_PATH,
    checkpoint_dir: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_threads: Optional[int] = DEFAULT_NUM_THREADS,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
) -> None:
    """
    Generate dense BGE embeddings and construct a FAISS IndexFlatIP vector index.
    Supports incremental disk checkpointing and thermal-safe CPU multi-threading.
    """
    chunks_path = Path(chunks_path).resolve()
    embeddings_path = Path(embeddings_path).resolve()
    ids_path = Path(ids_path).resolve()
    index_path = Path(index_path).resolve()

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure PyTorch CPU threads safely only if explicitly specified
    if num_threads is not None and num_threads > 0:
        try:
            torch.set_num_threads(num_threads)
            print(f"⚙️ PyTorch CPU thread count set to: {num_threads}")
        except Exception as e:
            print(f"⚠️ Could not set PyTorch thread count: {e}")

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"❌ Chunks data not found at: {chunks_path}. "
            "Please verify preprocessing outputs."
        )

    # Load chunks with deterministic ordering
    print(f"📘 Loading chunks from: {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunks = sorted(chunks, key=lambda x: x["chunk_id"])
    total_chunks = len(chunks)
    print(f"📄 Total chunks loaded: {total_chunks}")

    # Prepare batches
    total_batches = (total_chunks + batch_size - 1) // batch_size
    batch_slices = [
        (b_idx, chunks[b_idx * batch_size : min((b_idx + 1) * batch_size, total_chunks)])
        for b_idx in range(total_batches)
    ]

    # Initialize or resume checkpointing state
    completed_batch_set: Set[int] = set()
    in_memory_parts: List[np.ndarray] = []

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        manifest = _load_and_validate_checkpoint(
            checkpoint_dir=checkpoint_dir,
            chunks_path=chunks_path,
            total_chunks=total_chunks,
            batch_size=batch_size,
        )

        if manifest is not None:
            completed_batch_set = set(manifest.get("completed_batches", []))
            completed_chunk_count = manifest.get("completed_chunk_count", 0)
            pct = (completed_chunk_count / total_chunks) * 100.0 if total_chunks else 0.0
            print(
                f"🔄 RESUMING: Found valid checkpoint with {completed_chunk_count}/{total_chunks} "
                f"chunks ({len(completed_batch_set)}/{total_batches} batches, {pct:.1f}% complete)."
            )

    batches_to_process = [
        (b_idx, batch) for b_idx, batch in batch_slices if b_idx not in completed_batch_set
    ]

    if batches_to_process:
        tokenizer, model = load_model()
        print(f"⚡ Generating embeddings (batch_size={batch_size}, remaining_batches={len(batches_to_process)}/{total_batches})...")

        batches_since_checkpoint = 0

        for b_idx, batch in tqdm(batches_to_process, desc="Embedding batches"):
            # BGE passage prefix (exact preservation)
            texts = [f"passage: {chunk['content']}" for chunk in batch]
            batch_embeddings = embed_batch(texts, tokenizer, model)

            if checkpoint_dir is not None:
                # 1. Save batch array incrementally FIRST (~65 KB per batch, never the full matrix)
                batch_file = checkpoint_dir / f"batch_{b_idx:05d}.npy"
                np.save(batch_file, batch_embeddings)

                completed_batch_set.add(b_idx)
                batches_since_checkpoint += 1

                # 2. Update checkpoint manifest atomically AFTER batch is persisted
                if batches_since_checkpoint >= checkpoint_interval or b_idx == total_batches - 1:
                    completed_ids = [
                        chunk["chunk_id"]
                        for bi, b in batch_slices
                        if bi in completed_batch_set
                        for chunk in b
                    ]
                    next_batch_idx = min(
                        [bi for bi in range(total_batches) if bi not in completed_batch_set],
                        default=total_batches,
                    )
                    manifest_data = {
                        "chunks_path": str(chunks_path),
                        "total_chunks": total_chunks,
                        "model_name": MODEL_NAME,
                        "embedding_dim": EMBEDDING_DIM,
                        "batch_size": batch_size,
                        "next_batch_idx": next_batch_idx,
                        "completed_chunk_count": len(completed_ids),
                        "completed_batches": sorted(list(completed_batch_set)),
                        "completed_ids": completed_ids,
                    }
                    _atomic_write_json(manifest_data, checkpoint_dir / "manifest.json")
                    batches_since_checkpoint = 0
            else:
                in_memory_parts.append(batch_embeddings)
                completed_batch_set.add(b_idx)

    # ---------------------------------------------------
    # FINAL ASSEMBLY & VERIFICATION
    # ---------------------------------------------------
    print("\n📦 Assembling final embedding matrix...")

    if checkpoint_dir is not None:
        # Load batch parts in deterministic sequential order
        ordered_parts = []
        for b_idx, batch in batch_slices:
            part_file = checkpoint_dir / f"batch_{b_idx:05d}.npy"
            if not part_file.exists():
                raise FileNotFoundError(f"❌ Missing batch part file: {part_file}")
            ordered_parts.append(np.load(part_file))

        embeddings_matrix = np.vstack(ordered_parts).astype("float32")
    else:
        embeddings_matrix = np.vstack(in_memory_parts).astype("float32")

    id_mapping = [chunk["chunk_id"] for chunk in chunks]

    # Validation Checks
    print(f"📏 Validating embedding matrix shape: {embeddings_matrix.shape}...")
    if embeddings_matrix.shape != (total_chunks, EMBEDDING_DIM):
        raise ValueError(
            f"❌ Embedding shape mismatch: expected ({total_chunks}, {EMBEDDING_DIM}), got {embeddings_matrix.shape}"
        )

    if len(id_mapping) != total_chunks:
        raise ValueError(
            f"❌ ID mapping count mismatch: expected {total_chunks}, got {len(id_mapping)}"
        )

    if np.isnan(embeddings_matrix).any():
        raise ValueError("❌ NaN values detected in embedding matrix!")

    if np.isinf(embeddings_matrix).any():
        raise ValueError("❌ Infinite values detected in embedding matrix!")

    # Normalize for cosine similarity (IndexFlatIP)
    print("📏 Normalizing embeddings (L2)...")
    faiss.normalize_L2(embeddings_matrix)

    # Save final raw embeddings
    print(f"💾 Saving embeddings to: {embeddings_path}...")
    np.save(embeddings_path, embeddings_matrix)

    # Save final IDs
    print(f"💾 Saving IDs to: {ids_path}...")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f, indent=2)

    # Build FAISS index
    print("🏗️ Building FAISS IndexFlatIP...")
    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    if index.ntotal != total_chunks:
        raise ValueError(f"❌ FAISS vector count mismatch: expected {total_chunks}, got {index.ntotal}")

    print(f"💾 Saving FAISS index to: {index_path}...")
    faiss.write_index(index, str(index_path))

    # Clean up checkpoint files only after all final artifacts are successfully verified
    if checkpoint_dir is not None and checkpoint_dir.exists():
        print(f"🧹 Cleaning up temporary checkpoint directory: {checkpoint_dir}...")
        try:
            shutil.rmtree(checkpoint_dir)
            print("✅ Checkpoint directory cleaned.")
        except Exception as e:
            print(f"⚠️ Could not remove checkpoint directory: {e}")

    print("✅ Dense indexing complete and fully verified.")
    print(f"📦 Total vectors indexed: {index.ntotal}")


# ===================================================
# ENTRYPOINT
# ===================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dense BGE FAISS index from chunks with incremental checkpointing.")
    parser.add_argument("--chunks-path", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks JSON")
    parser.add_argument("--embeddings-path", type=str, default=str(DEFAULT_EMBEDDINGS_PATH), help="Path to save embeddings .npy")
    parser.add_argument("--ids-path", type=str, default=str(DEFAULT_IDS_PATH), help="Path to save chunk IDs JSON")
    parser.add_argument("--index-path", type=str, default=str(DEFAULT_FAISS_INDEX_PATH), help="Path to save FAISS index")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to store incremental batch checkpoints")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size (baseline default: 32)")
    parser.add_argument("--threads", type=int, default=DEFAULT_NUM_THREADS, help="PyTorch CPU thread count limit (default: None, unconstrained)")
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help="Sync manifest every N batches")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    build_dense_index(
        chunks_path=Path(args.chunks_path),
        embeddings_path=Path(args.embeddings_path),
        ids_path=Path(args.ids_path),
        index_path=Path(args.index_path),
        checkpoint_dir=ckpt_dir,
        batch_size=args.batch_size,
        num_threads=args.threads,
        checkpoint_interval=args.checkpoint_interval,
    )