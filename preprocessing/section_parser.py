import re
import json
from pathlib import Path
from typing import List, Dict


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

INPUT_PATH = Path("data/processed/merck_cleaned.txt")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "merck_structured.json"

CHAPTER_REGEX = re.compile(r"^Chapter\s+(\d+)\.\s+(.+)")


# ---------------------------------------------------
# CORE PARSER
# ---------------------------------------------------

def parse_chapters(text: str) -> List[Dict]:
    """
    Parse cleaned Merck corpus into chapter-level structured records.
    No fragile section heuristics.
    Deterministic + stable.
    """

    records: List[Dict] = []

    current_chapter_number = None
    current_chapter_title = None
    current_content: List[str] = []

    lines = text.splitlines()

    for line in lines:
        line_stripped = line.strip()

        chapter_match = CHAPTER_REGEX.match(line_stripped)

        # -----------------------------
        # New Chapter Detected
        # -----------------------------
        if chapter_match:

            # Save previous chapter
            if current_chapter_number is not None:
                chapter_text = " ".join(current_content).strip()

                if chapter_text:
                    records.append({
                        "chapter_number": current_chapter_number,
                        "chapter_title": current_chapter_title,
                        "content": chapter_text,
                        "char_length": len(chapter_text)
                    })

            # Initialize new chapter
            current_chapter_number = int(chapter_match.group(1))
            current_chapter_title = chapter_match.group(2).strip()
            current_content = []

            continue

        # Ignore content before first chapter
        if current_chapter_number is None:
            continue

        # Collect non-empty lines
        if line_stripped:
            current_content.append(line_stripped)

    # -----------------------------
    # Save Final Chapter
    # -----------------------------
    if current_chapter_number is not None:
        chapter_text = " ".join(current_content).strip()

        if chapter_text:
            records.append({
                "chapter_number": current_chapter_number,
                "chapter_title": current_chapter_title,
                "content": chapter_text,
                "char_length": len(chapter_text)
            })

    return records


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------

if __name__ == "__main__":

    print("📘 Loading cleaned corpus...")
    text = INPUT_PATH.read_text(encoding="utf-8")

    print("🔍 Parsing chapters (stable mode)...")
    records = parse_chapters(text)

    print(f"✅ Total chapters parsed: {len(records)}")

    if records:
        avg_length = sum(r["char_length"] for r in records) // len(records)
        print(f"📊 Average chapter length (chars): {avg_length}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"💾 Structured JSON saved to: {OUTPUT_PATH}")