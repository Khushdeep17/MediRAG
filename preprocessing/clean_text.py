import re
from pathlib import Path
from typing import List
from tqdm import tqdm
from pypdf import PdfReader


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

PDF_PATH = Path("data/merck_manual.pdf")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "merck_cleaned.txt"

# Keep ONLY core medical pages
START_PAGE = 53
END_PAGE = 3655


# ---------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """
    Normalize spacing while preserving structure.
    """
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------
# HEADER / FOOTER REMOVAL
# ---------------------------------------------------

def remove_headers_footers(text: str) -> str:
    """
    Remove recurring header/footer artifacts.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line_strip = line.strip()

        if not line_strip:
            cleaned.append("")
            continue

        if "Merck Manual" in line_strip:
            continue

        if "Copyright" in line_strip:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------
# STRUCTURAL NOISE REMOVAL (SAFE)
# ---------------------------------------------------

def remove_structural_noise(text: str) -> str:
    """
    Remove structural references without harming
    biomedical content.
    """

    # Remove structural bracket references only
    text = re.sub(
        r"\[(Table|Fig|Figure|See).*?\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove page-based cross references like (see p. 123)
    text = re.sub(
        r"\(see p\..*?\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove standalone numeric page lines (1–4 digits)
    text = re.sub(r"\n\d{1,4}\n", "\n", text)

    return text


# ---------------------------------------------------
# LINE BREAK FIXING (HEADING SAFE)
# ---------------------------------------------------

def fix_line_breaks(text: str) -> str:
    """
    Fix PDF line breaks while preserving headings.
    """

    # Fix hyphenated word splits
    text = re.sub(r"-\n", "", text)

    # Merge broken sentences only when:
    # previous char not period
    # next line starts lowercase
    text = re.sub(
        r"(?<![.\n])\n(?=[a-z])",
        " ",
        text,
    )

    return text


# ---------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------

def extract_main_content(pdf_path: Path) -> List[str]:
    """
    Extract and clean core medical content pages.
    """

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    print(f"📄 Total PDF pages: {total_pages}")
    print(f"📘 Extracting pages {START_PAGE} → {END_PAGE}")

    cleaned_pages = []

    for i in tqdm(range(START_PAGE - 1, END_PAGE), desc="Processing pages"):
        page = reader.pages[i]
        raw_text = page.extract_text()

        if not raw_text:
            continue

        text = normalize_whitespace(raw_text)
        text = remove_headers_footers(text)
        text = remove_structural_noise(text)
        text = fix_line_breaks(text)
        text = normalize_whitespace(text)

        cleaned_pages.append(text)

    return cleaned_pages


# ---------------------------------------------------
# FINAL CORPUS POST-PROCESSING
# ---------------------------------------------------

def assemble_corpus(pages: List[str]) -> str:
    """
    Combine pages and trim non-medical front matter.
    """

    full_text = "\n\n".join(pages)

    # Trim everything before Chapter 1
    chapter_start = full_text.find("Chapter 1.")
    if chapter_start != -1:
        full_text = full_text[chapter_start:]

    return full_text.strip()


# ---------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------

def save_clean_corpus(text: str, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Final clean corpus saved at: {output_path}")


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------

if __name__ == "__main__":
    pages = extract_main_content(PDF_PATH)
    corpus = assemble_corpus(pages)
    save_clean_corpus(corpus, OUTPUT_PATH)