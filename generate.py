import os
import re
from groq import Groq
from dotenv import load_dotenv
from retrieval.fusion import hybrid_search

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME    = "openai/gpt-oss-120b"
TOP_K_CONTEXT = 5
MAX_TOKENS    = 700

# =====================================================
# LOAD ENV + CLIENT
# =====================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=api_key)

# =====================================================
# FORMAT CONTEXT
# =====================================================

def format_context(chunks: list) -> str:
    formatted = []
    for i, chunk in enumerate(chunks[:TOP_K_CONTEXT], 1):
        chapter_number = chunk["chapter_number"]
        chunk_id = chunk.get("chunk_id", f"{chapter_number}-{i:02d}")
        formatted.append(
            f"[{i}]\n"
            f"Chapter {chapter_number}: {chunk['chapter_title']}\n"
            f"Chunk ID: {chunk_id}\n"
            f"{chunk['content'][:1200]}"
        )
    return "\n\n---\n\n".join(formatted)

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """\
You are a medical retrieval-augmented QA assistant. Answer only using the \
retrieved context and do not use outside medical knowledge. If the context \
does not contain the answer, say: "Not covered in provided context." Be \
accurate, concise, and well structured. Cite each paragraph with the most \
relevant source. Prefer one source citation over several unless the sources \
contribute distinct information. Answer the user's question directly instead \
of summarizing the entire disease.\
"""

# =====================================================
# BUILD PROMPT
# =====================================================

def build_prompt(query: str, context_text: str) -> str:
    return f"""Answer the question using ONLY the provided context.

Answer the user's question directly, not the entire disease. Include only sections
that are relevant to the question. Use Markdown headings beginning with
`## Overview`; make Overview a short two-sentence paragraph, not a bullet list.
Treatment questions may use `## Acute Management` and `## Long-term Management`
when the context supports those distinctions. Symptom,
cause, definition, and comparison questions should receive the headings that best
fit the question.

Use bullet points where helpful. Keep the answer concise, approximately 200–300
words. Support factual statements with inline citations using exactly [1] through
[5]. Cite each paragraph with the most relevant source and prefer one citation
over multiple citations unless they provide distinct information. Do not use
alternative citation markers or source labels such as (Source 1). Do not add a
References section; inline citations are sufficient.
Avoid unsupported claims and say "Not covered in provided context." when the
context does not contain the requested information.

---

## CONTEXT

{context_text}

---

## QUESTION

{query}

---

## ANSWER
"""

# =====================================================
# CLEAN MODEL OUTPUT
# =====================================================

def clean_answer(raw: str) -> str:
    """Strip chain-of-thought <think> blocks if the model emits them."""
    if "<think>" in raw:
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        else:
            raw = raw.split("<think>")[0].strip()

    # Normalize model-generated citation variants to the app's [N] format.
    raw = re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", raw)

    # The UI already exposes retrieved sources, so inline citations are enough.
    raw = re.split(r"(?im)^\s*(?:#{1,6}\s*)?References\s*:?\s*$", raw, maxsplit=1)[0]
    return raw.strip()

# =====================================================
# GENERATE ANSWER
# =====================================================

def generate_answer(query: str, verbose: bool = False):

    # 1️⃣ Hybrid Retrieval
    retrieved_chunks = hybrid_search(
        query,
        return_results=True,
        verbose=verbose
    )

    if not retrieved_chunks:
        return "No relevant context retrieved.", []

    retrieved_chunks = retrieved_chunks[:TOP_K_CONTEXT]

    # 2️⃣ Format context
    context_text = format_context(retrieved_chunks)

    # 3️⃣ Build prompt
    prompt = build_prompt(query, context_text)

    # 4️⃣ Call Groq
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0.25,         # Slightly lower — tighter grounding, less creative expansion
        max_tokens=MAX_TOKENS,
        frequency_penalty=0.2,    # Prevents "lifestyle adjustments / lifestyle modifications" type repetition
        presence_penalty=0,
    )

    if verbose:
        print(f"Generation finish reason: {completion.choices[0].finish_reason}")

    raw_answer = completion.choices[0].message.content

    if not raw_answer:
        return "⚠️ Model returned an empty response.", retrieved_chunks

    answer = clean_answer(raw_answer)

    return answer, retrieved_chunks

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    query = "What are the causes and treatment of migraine?"

    print("\n🔎 Running Hybrid Retrieval + Groq Generation...\n")

    answer, sources = generate_answer(query, verbose=True)

    print("=" * 70)
    print("📌 Generated Answer:\n")
    print(answer)

    print("\n" + "=" * 70)
    print("📚 Retrieved Sources:\n")

    for i, chunk in enumerate(sources, 1):
        print(f"  [{i}] Chapter {chunk['chapter_number']} — {chunk['chapter_title']}")