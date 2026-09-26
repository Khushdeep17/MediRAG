import os
import sys
import json
import re
from datetime import datetime

# --- Fix import path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from generate import generate_answer   # returns (answer, retrieved_chunks)

# =====================================================
# CONFIG
# =====================================================

OUTPUT_FILE = "evaluation/generation_outputs.json"

# =====================================================
# 20 BALANCED QUERIES — 7 Tier-1, 7 Tier-2, 6 Tier-3
# Manually picked for medical diversity
# =====================================================

GENERATION_QUERIES = [

    # ── TIER 1: Direct (7) ────────────────────────────────────────────────────
    {"query": "What are the causes and treatment of migraine?",               "relevant_chapter": 178, "tier": 1},
    {"query": "What are the symptoms of Parkinson disease?",                  "relevant_chapter": 183, "tier": 1},
    {"query": "How is asthma treated?",                                       "relevant_chapter": 191, "tier": 1},
    {"query": "What causes iron deficiency anemia?",                          "relevant_chapter": 105, "tier": 1},
    {"query": "How is diabetes mellitus managed?",                            "relevant_chapter":  99, "tier": 1},
    {"query": "What are the risk factors and treatment of hypertension?",     "relevant_chapter": 208, "tier": 1},
    {"query": "What causes peptic ulcer disease and how is it treated?",      "relevant_chapter":  13, "tier": 1},

    # ── TIER 2: Indirect / Clinical Framing (7) ───────────────────────────────
    {"query": "A patient presents with recurring severe headaches with nausea and light sensitivity. What is the diagnosis and management?",
                                                                              "relevant_chapter": 178, "tier": 2},
    {"query": "What neurological condition causes resting tremor, rigidity, and bradykinesia?",
                                                                              "relevant_chapter": 183, "tier": 2},
    {"query": "How is reversible airflow obstruction treated in adults?",     "relevant_chapter": 191, "tier": 2},
    {"query": "A patient has low hemoglobin and low ferritin — what are the causes and treatment?",
                                                                              "relevant_chapter": 105, "tier": 2},
    {"query": "What clinical signs suggest an underactive thyroid and how is it corrected?",
                                                                              "relevant_chapter":  93, "tier": 2},
    {"query": "What lifestyle and pharmacological interventions reduce blood pressure?",
                                                                              "relevant_chapter": 208, "tier": 2},
    {"query": "What are the first and second line therapies for H. pylori related gastric disease?",
                                                                              "relevant_chapter":  13, "tier": 2},

    # ── TIER 3: Hard / Multi-concept (6) ──────────────────────────────────────
    {"query": "What mechanisms explain why certain medications worsen thyroid function?",
                                                                              "relevant_chapter":  93, "tier": 3},
    {"query": "How does electrolyte imbalance affect cardiac rhythm?",        "relevant_chapter":  97, "tier": 3},
    {"query": "How does portal hypertension develop in patients with liver fibrosis?",
                                                                              "relevant_chapter":  27, "tier": 3},
    {"query": "What clotting cascade abnormalities lead to bleeding in liver disease?",
                                                                              "relevant_chapter": 111, "tier": 3},
    {"query": "What distinguishes angina pectoris from acute myocardial ischemia on presentation?",
                                                                              "relevant_chapter": 210, "tier": 3},
    {"query": "What is the mechanism behind refeeding syndrome in malnourished patients?",
                                                                              "relevant_chapter":   3, "tier": 3},
]

# =====================================================
# HELPERS
# =====================================================

def extract_cited_numbers(text: str) -> list[int]:
    """Extract all [N] citation numbers from generated answer text."""
    return sorted(set(int(n) for n in re.findall(r'\[(\d+)\]', text)))


def chunk_to_dict(chunk: dict) -> dict:
    """Serialize a retrieved chunk to a JSON-safe dict."""
    return {
        "chapter_number": chunk.get("chapter_number"),
        "chapter_title" : chunk.get("chapter_title", ""),
        "content_snippet": chunk.get("content", "")[:500],   # keep JSON small
    }

# =====================================================
# MAIN — Run generation for all 20 queries
# =====================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_results = []

    print(f"\n{'═' * 60}")
    print(f"  MediRAG — Generation Evaluation Data Collection")
    print(f"  Queries : {len(GENERATION_QUERIES)}")
    print(f"  Output  : {OUTPUT_FILE}")
    print(f"{'═' * 60}\n")

    for idx, item in enumerate(GENERATION_QUERIES, 1):

        query            = item["query"]
        expected_chapter = item["relevant_chapter"]
        tier             = item["tier"]

        print(f"[{idx:02d}/20] Tier {tier} | {query[:72]}")

        # ── Generate (internally calls hybrid retrieval + Groq) ────────────────
        answer, retrieved_chunks = generate_answer(query, verbose=False)

        retrieved_chapter_numbers = [c["chapter_number"] for c in retrieved_chunks[:5]]
        cited_numbers             = extract_cited_numbers(answer)

        # Expected chapter cited = did model cite source that maps to expected chapter?
        # cited_numbers are 1-indexed positions in the retrieved list
        cited_actual_chapters = [
            retrieved_chunks[i - 1]["chapter_number"]
            for i in cited_numbers
            if 0 < i <= len(retrieved_chunks)
        ]
        expected_chapter_cited = expected_chapter in cited_actual_chapters

        record = {
            "id"                      : idx,
            "query"                   : query,
            "tier"                    : tier,
            "expected_chapter"        : expected_chapter,
            "retrieved_chapters"      : retrieved_chapter_numbers,
            "retrieved_chunks"        : [chunk_to_dict(c) for c in retrieved_chunks[:5]],
            "generated_answer"        : answer,
            "cited_source_numbers"    : cited_numbers,
            "cited_actual_chapters"   : cited_actual_chapters,
            "expected_chapter_cited"  : expected_chapter_cited,
            "timestamp"               : str(datetime.utcnow()),
            # ── Grading fields (filled by manual_grades.py / llm_judge.py) ────
            "manual_accuracy_grade"   : None,   # 0=wrong, 1=partial, 2=correct
            "llm_faithfulness"        : None,   # 1–5
            "llm_completeness"        : None,   # 1–5
            "llm_medical_accuracy"    : None,   # 1–5
        }

        all_results.append(record)

        # Progress print
        hit = "✅" if expected_chapter in retrieved_chapter_numbers else "❌"
        cite = "📎" if expected_chapter_cited else "  "
        print(f"       Retrieved: {retrieved_chapter_numbers}  {hit}")
        print(f"       Cited chapters: {cited_actual_chapters}  {cite}")
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"{'═' * 60}")
    print(f"✅ Saved {len(all_results)} records → {OUTPUT_FILE}")
    print(f"   Next steps:")
    print(f"   1. python evaluation/auto_metrics.py     ← automated scoring")
    print(f"   2. python evaluation/llm_judge.py        ← LLM-as-Judge scoring")
    print(f"   3. python evaluation/manual_grades.py    ← fill in your 0/1/2 grades")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
