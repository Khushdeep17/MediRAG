import os
import re
from groq import Groq
from dotenv import load_dotenv
from retrieval.fusion import hybrid_search

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME    = "openai/gpt-oss-120b"
TOP_K_CONTEXT = 4
MAX_TOKENS    = 1250
CHUNK_CHAR_LIMIT = 2200
FUSION_ALPHA  = 0.7
MAX_RETRIES   = 3

# =====================================================
# LOAD ENV + CLIENT (LAZY INITIALIZATION)
# =====================================================

_client = None


def get_groq_client() -> Groq:
    """Lazily instantiate and return the Groq API client."""
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ GROQ_API_KEY not found in environment variables. "
                "Please configure GROQ_API_KEY in your .env file or environment."
            )
        _client = Groq(api_key=api_key)
    return _client


# =====================================================
# FORMAT CONTEXT
# =====================================================

def format_context(chunks: list) -> str:
    formatted = []
    for i, chunk in enumerate(chunks[:TOP_K_CONTEXT], 1):
        chapter_number = chunk.get("chapter_number", "?")
        chapter_title = chunk.get("chapter_title", "Unknown")
        chunk_id = chunk.get("chunk_id", f"{chapter_number}-{i:02d}")
        # Up to 2200 chars gives complete clinical context while respecting Groq TPM limits
        content = chunk.get("content", "").strip()[:CHUNK_CHAR_LIMIT]
        formatted.append(
            f"[{i}]\n"
            f"Chapter {chapter_number}: {chapter_title}\n"
            f"Chunk ID: {chunk_id}\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(formatted)

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """\
You are an expert medical retrieval-augmented QA assistant. Your mission is to provide \
strictly grounded, accurate, structured, and clinically precise answers using ONLY \
the provided medical reference text.

Core Guidelines:
1. Strict Grounding: Rely strictly on the provided context excerpts. Do not use outside medical knowledge or assume facts not present in the excerpts.
2. Direct Focus: Address the specific question directly. Do not summarize the entire disease or provide generic boilerplate if not asked.
3. Accurate Inline Citations: Every substantive factual claim, symptom, diagnostic test, or treatment must be followed by its source citation using [1], [2], [3], or [4] corresponding to the excerpt where the fact appears.
4. Structured Formatting: Use clear Markdown headings (starting with ## Overview), well-organized paragraphs, and bullet points for lists.
5. Partial or Missing Coverage: If the retrieved excerpts do not contain enough information to answer a specific aspect of the query, explicitly state what is not covered in the excerpts rather than speculating or hallucinating.\
"""

# =====================================================
# BUILD PROMPT
# =====================================================

def build_prompt(query: str, context_text: str) -> str:
    return f"""Answer the following medical question based strictly on the provided retrieved context.

Instructions:
- Begin with a short 2-3 sentence `## Overview` answering the central query directly.
- Organize the remainder of your answer with logical Markdown headings that fit the question (for example: `## Symptoms & Clinical Presentation`, `## Causes & Etiology`, `## Treatment & Management` [with `### Acute Management` and `### Long-Term Management` when applicable], or `## Diagnostic Evaluation`).
- Use clear bullet points for key symptoms, mechanisms, risk factors, or medication classes.
- Place inline citations like [1] or [2] immediately after each factual sentence or bullet item. Use only standard brackets like [1] (never 【1】 or (Source 1)).
- Keep the answer concise and high-yield, approximately 300–400 words.
- If an aspect of the question is not mentioned in the context, explicitly state: "The provided context does not mention [aspect]." Do not fabricate information.
- Do not include an overall References or Sources list at the end (inline citations are sufficient).

---

## RETRIEVED MEDICAL CONTEXT

{context_text}

---

## USER QUESTION

{query}

---

## GROUNDED ANSWER
"""

# =====================================================
# CLEAN MODEL OUTPUT
# =====================================================

def clean_answer(raw: str) -> str:
    """Clean model output, normalize citations, and remove redundant sections."""
    if not raw:
        return ""

    # Strip chain-of-thought <think> blocks if the model emits them.
    if "<think>" in raw:
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        else:
            raw = raw.split("<think>")[0].strip()

    # Normalize weird whitespace before citations (e.g., \u202f[1], \xa0[1]) to standard space
    raw = re.sub(r"[\u202f\u00a0\u2000-\u200b]+(?=\[\d+\])", " ", raw)

    # Normalize model-generated citation variants like 【1】 or 【1†source】 to [1]
    raw = re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", raw)
    raw = re.sub(r"\[Source\s*(\d+)\]", r"[\1]", raw, flags=re.IGNORECASE)

    # Clean unclosed citation brackets at the end if truncated
    raw = re.sub(r"[【\[]\d*$", "", raw).strip()
    raw = re.sub(r"[【\[]\d+[^】\]]*$", "", raw).strip()

    # Remove duplicate consecutive citations like [1][1] -> [1]
    raw = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", raw)

    # Remove any trailing References or Sources list
    raw = re.split(r"(?im)^\s*(?:#{1,6}\s*)?(?:References|Sources)\s*:?\s*$", raw, maxsplit=1)[0]

    return raw.strip()

# =====================================================
# GENERATE ANSWER
# =====================================================

def generate_answer(query: str, verbose: bool = False):
    import time

    # 1️⃣ Hybrid Retrieval (using optimal alpha=0.7)
    retrieved_chunks = hybrid_search(
        query,
        alpha=FUSION_ALPHA,
        top_k=TOP_K_CONTEXT,
        return_results=True,
        verbose=verbose
    )

    if not retrieved_chunks:
        return "No relevant context retrieved.", []

    retrieved_chunks = retrieved_chunks[:TOP_K_CONTEXT]

    # 2️⃣ Format context (full passages)
    context_text = format_context(retrieved_chunks)

    # 3️⃣ Build prompt
    prompt = build_prompt(query, context_text)

    # 4️⃣ Call Groq with rate-limit retry support
    client = get_groq_client()
    completion = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
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
                temperature=0.2,         # Low temperature for precise grounding
                max_tokens=MAX_TOKENS,
                frequency_penalty=0.15,
                presence_penalty=0,
            )
            break
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                wait_time = 4.0 * attempt
                if verbose:
                    print(f"⏳ Rate limit encountered. Waiting {wait_time:.1f}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait_time)
            else:
                raise e

    if completion is None:
        return "⚠️ Service temporarily busy due to rate limits. Please try again in a few seconds.", retrieved_chunks

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

    query = "What are the manifestations of acute pancreatitis?"

    print("\n🔎 Running Hybrid Retrieval + Groq Generation...\n")

    answer, sources = generate_answer(query, verbose=True)

    print("=" * 70)
    print("📌 Generated Answer:\n")
    print(answer)

    print("\n" + "=" * 70)
    print("📚 Retrieved Sources:\n")

    for i, chunk in enumerate(sources, 1):
        print(f"  [{i}] Chapter {chunk['chapter_number']} — {chunk['chapter_title']}")