import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merck_structured.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150

MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Common section headings in medical textbook chapters
KNOWN_SECTION_HEADINGS = [
    "Approach to the Patient With",
    "Symptoms and Signs",
    "Pathophysiology",
    "Diagnosis",
    "Treatment",
    "Etiology",
    "Prognosis",
    "Prevention",
    "Key Points",
    "Evaluation",
    "History:",
    "Physical examination:",
]


# ---------------------------------------------------
# LOAD TOKENIZER (LAZY)
# ---------------------------------------------------

_tokenizer: Optional[AutoTokenizer] = None


def get_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Lazily load and return the HuggingFace tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        print("🔍 Loading BGE tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


# ---------------------------------------------------
# CHUNKING FUNCTION
# ---------------------------------------------------

def chunk_tokens(tokens: List[int], chunk_size: int, overlap: int) -> List[List[int]]:
    """Split a list of tokens into overlapping slices."""
    chunks = []
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk = tokens[start:end]

        if len(chunk) < 50:  # Avoid tiny trailing chunks
            break

        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def detect_section_heading(text: str) -> Optional[str]:
    """Identify if a known clinical section heading appears near the start of the chunk."""
    text_prefix = text[:200]
    for heading in KNOWN_SECTION_HEADINGS:
        if heading.lower() in text_prefix.lower():
            return heading.rstrip(":")
    return None


# ---------------------------------------------------
# CHUNK GENERATION & STATS
# ---------------------------------------------------

def create_chunks(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    input_path: Path = INPUT_PATH,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Generate chunks from structured chapter JSON with configurable size and overlap."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(
            f"❌ Structured chapter JSON not found at: {input_path}. "
            "Please run `python preprocessing/section_parser.py` first."
        )

    if output_path is None:
        output_path = OUTPUT_DIR / f"merck_chunks_{chunk_size}_{overlap}.json"

    print(f"📘 Loading chapter-level JSON from {input_path}...")
    chapters = json.loads(input_path.read_text(encoding="utf-8"))

    all_chunks: List[Dict[str, Any]] = []
    token_lengths: List[int] = []

    print(f"✂️ Token-aware chunking (size={chunk_size}, overlap={overlap}) in progress...\n")
    tokenizer = get_tokenizer()

    for chapter in tqdm(chapters):
        chapter_num = chapter["chapter_number"]
        chapter_title = chapter["chapter_title"]
        content = chapter["content"]

        tokens = tokenizer.encode(content, add_special_tokens=False)
        token_chunks = chunk_tokens(tokens, chunk_size, overlap)

        for idx, token_chunk in enumerate(token_chunks):
            text_chunk = tokenizer.decode(token_chunk)
            token_len = len(token_chunk)
            token_lengths.append(token_len)

            section_heading = detect_section_heading(text_chunk)
            breadcrumb = f"Chapter {chapter_num}: {chapter_title}"
            if section_heading:
                breadcrumb += f" > {section_heading}"

            chunk_record: Dict[str, Any] = {
                "chunk_id": f"{chapter_num}_{idx}",
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "section_title": section_heading,
                "breadcrumb": breadcrumb,
                "content": text_chunk,
                "token_length": token_len,
                "char_length": len(text_chunk),
            }
            all_chunks.append(chunk_record)

    # Statistical evaluation
    token_arr = np.array(token_lengths)
    total_count = len(all_chunks)
    min_tokens = int(np.min(token_arr))
    max_tokens = int(np.max(token_arr))
    mean_tokens = float(np.mean(token_arr))
    median_tokens = float(np.median(token_arr))
    p90_tokens = float(np.percentile(token_arr, 90))
    p95_tokens = float(np.percentile(token_arr, 95))
    gt_512 = int(np.sum(token_arr > 512))
    le_512 = int(np.sum(token_arr <= 512))

    print("\n" + "=" * 60)
    print(f"✅ Chunking Complete: {chunk_size}/{overlap}")
    print("=" * 60)
    print(f"📦 Total chunk count       : {total_count}")
    print(f"📊 Minimum token count     : {min_tokens}")
    print(f"📊 Maximum token count     : {max_tokens}")
    print(f"📊 Mean token count        : {mean_tokens:.2f}")
    print(f"📊 Median token count      : {median_tokens:.2f}")
    print(f"📊 P90 token count         : {p90_tokens:.2f}")
    print(f"📊 P95 token count         : {p95_tokens:.2f}")
    print(f"🚨 Chunks > 512 tokens     : {gt_512}")
    print(f"✅ Chunks <= 512 tokens    : {le_512}")
    print("=" * 60)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {total_count} chunks to: {output_path}")
    return all_chunks


# ---------------------------------------------------
# CLI
# ---------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurable token-aware chunking.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Token chunk size")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Token overlap")
    parser.add_argument("--output-path", type=str, default=None, help="Custom output JSON path")
    args = parser.parse_args()

    custom_out = Path(args.output_path) if args.output_path else None
    create_chunks(chunk_size=args.chunk_size, overlap=args.overlap, output_path=custom_out)