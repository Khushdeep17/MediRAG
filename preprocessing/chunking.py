import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

INPUT_PATH = Path("data/processed/merck_structured.json")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800
OVERLAP = 150

OUTPUT_PATH = OUTPUT_DIR / f"merck_chunks_{CHUNK_SIZE}_{OVERLAP}.json"

MODEL_NAME = "BAAI/bge-large-en-v1.5"


# ---------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------

print("🔍 Loading BGE tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------
# CHUNKING FUNCTION
# ---------------------------------------------------

def chunk_tokens(tokens: List[int], chunk_size: int, overlap: int) -> List[List[int]]:
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


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":

    print("📘 Loading chapter-level JSON...")
    chapters = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    all_chunks = []
    token_lengths = []

    print("✂️ Token-aware chunking in progress...\n")

    for chapter in tqdm(chapters):

        chapter_num = chapter["chapter_number"]
        chapter_title = chapter["chapter_title"]
        content = chapter["content"]

        tokens = tokenizer.encode(content, add_special_tokens=False)
        token_chunks = chunk_tokens(tokens, CHUNK_SIZE, OVERLAP)

        for idx, token_chunk in enumerate(token_chunks):

            text_chunk = tokenizer.decode(token_chunk)
            token_len = len(token_chunk)

            token_lengths.append(token_len)

            all_chunks.append({
                "chunk_id": f"{chapter_num}_{idx}",
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "content": text_chunk,
                "token_length": token_len,
                "char_length": len(text_chunk)
            })

    avg_token_len = sum(token_lengths) / len(token_lengths)

    print("\n✅ Chunking complete.")
    print(f"📦 Total chunks created: {len(all_chunks)}")
    print(f"📊 Average token length: {avg_token_len:.2f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"💾 Saved to: {OUTPUT_PATH}")