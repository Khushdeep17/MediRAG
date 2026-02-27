import os
from groq import Groq
from dotenv import load_dotenv
from retrieval.fusion import hybrid_search

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME    = "qwen/qwen3-32b"
TOP_K_CONTEXT = 5
MAX_TOKENS    = 2000

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
        formatted.append(
            f"[Source {i}] Chapter {chunk['chapter_number']} — {chunk['chapter_title']}\n"
            f"{chunk['content'][:1200]}"
        )
    return "\n\n---\n\n".join(formatted)

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """\
You are a precise medical writing assistant. You produce structured, readable \
answers using ONLY the information explicitly present in the provided context. \
You balance explanatory paragraphs with selective bullet points. \
You never infer, expand, or use terminology that is not directly supported \
by the source text — even if it seems medically correct.\
"""

# =====================================================
# BUILD PROMPT
# =====================================================

def build_prompt(query: str, context_text: str) -> str:
    return f"""Answer the medical question below using ONLY the provided context. Follow the structure and rules exactly.

---

## RESPONSE STRUCTURE

### 1. Overview
2–3 sentences giving a direct, high-level answer. No bullet points. No medical jargon without explanation.

### 2. Symptoms
Open with a short paragraph describing the symptom pattern as described in the context.
Then list specific symptoms as bullets — each bullet must be a complete thought, not just a word.
Include subtypes if mentioned (e.g., aura vs. no aura).

### 3. Causes
Open with a paragraph describing the underlying mechanisms or triggers as stated in the context.
Follow with bullets for distinct causes or risk factors.
⚠️ Use only terminology that appears directly in the retrieved text. Do not paraphrase into more specific medical language than what the source uses.

### 4. Treatment
Split this section into two clearly labelled sub-sections:

**Acute Management** — treatments used during an active attack.
Start with a paragraph, then list medications or interventions as bullets.

**Preventive Management** — strategies used to reduce frequency long-term.
Start with a paragraph, then list approaches as bullets.
Do NOT merge these two — they serve different goals and must stay separate.

### 5. Medical Terms
After the main answer, add a short "Key Terms" block. Define any medical terms used in the response in plain English — one line per term.
Only include terms that were actually used in your answer.

### 6. Closing Note
1–2 sentences on when to seek professional medical advice or what affects patient outcomes.

---

## GROUNDING RULES (critical)

- Use ONLY information explicitly present in the context. Do not expand, infer, or fill gaps with background medical knowledge — even if you are confident it is correct.
- If a mechanism or term is not in the source text, do not include it. Use the source's own phrasing where possible.
- Add inline citations [1], [2] after every specific fact, referencing the Source number from the context.
- If the context does not contain enough information to fill a section, write: "Not covered in provided context." — do not fabricate.
- Do not repeat the question or restate these instructions in your answer.

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
        presence_penalty=0.1,     # Encourages covering all sections (symptoms ≠ causes ≠ treatment)
    )

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