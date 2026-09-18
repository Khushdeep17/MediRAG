import sys
from pathlib import Path
import json
import re
import numpy as np
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_CURRENT = PROJECT_ROOT / "evaluation" / "results" / "current"
INPUT_FILE = RESULTS_CURRENT / "generation_outputs.json"
if not INPUT_FILE.exists():
    INPUT_FILE = PROJECT_ROOT / "evaluation" / "generation_outputs.json"
OUTPUT_FILE = RESULTS_CURRENT / "auto_metrics_results.json"

# ------------------------------
# Secondary (Lexical) Settings
# ------------------------------

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","of","in","on","at",
    "to","for","with","by","from","as","or","and","but","not",
    "this","that","these","those","it","its","which","who",
    "what","how","when","where","patient","treatment","disease",
    "condition","symptoms","cause","causes","used","also",
}

MIN_TERM_LENGTH = 4

def tokenize(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) >= MIN_TERM_LENGTH}

def lexical_grounded_rate(answer_tokens, context_tokens):
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"❌ Input file not found: {INPUT_FILE}. "
        "Please run `python evaluation/generation_eval.py` first."
    )

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)

retrieval_hits = 0
citation_accuracy = 0
citation_consistency = 0
struct_grounded = 0

lexical_scores = []
tier_stats = defaultdict(lambda: {"count": 0, "citation_correct": 0, "lexical_scores": []})

per_query_results = []

# ------------------------------
# Main Loop
# ------------------------------

for idx, item in enumerate(data, 1):

    rec_id = item.get("id", item.get("question_id", idx))
    expected = item.get("expected_chapter")
    retrieved = set(item.get("retrieved_chapters", []))
    cited = set(item.get("cited_actual_chapters", []))
    tier = item.get("tier", 1)

    tier_stats[tier]["count"] += 1

    # 1️⃣ Retrieval Hit
    retrieval_hit = expected in retrieved
    if retrieval_hit:
        retrieval_hits += 1

    # 2️⃣ Citation Accuracy (Legacy Chapter-Presence Proxy)
    citation_correct = bool(item.get("expected_chapter_cited", False))
    if citation_correct:
        citation_accuracy += 1
        tier_stats[tier]["citation_correct"] += 1

    # 3️⃣ Citation Consistency (Legacy Tautological Proxy)
    citation_consistent = cited.issubset(retrieved) if cited else True
    if citation_consistent:
        citation_consistency += 1

    # 4️⃣ Structural Grounded (Legacy Conjunction Heuristic)
    structural_grounded = (
        retrieval_hit and
        citation_correct and
        citation_consistent
    )

    if structural_grounded:
        struct_grounded += 1

    # 5️⃣ Context Utilization / Lexical Grounding (Full Evidence)
    answer = item.get("generated_answer", "")
    chunks = item.get("retrieved_chunks", [])
    # Prefer full generation-visible content; fallback gracefully to content_snippet
    context_text = " ".join(
        (c.get("content") or c.get("content_snippet", "")).strip()[:2200] for c in chunks
    )

    ans_tokens = tokenize(answer)
    ctx_tokens = tokenize(context_text)

    lexical_score = lexical_grounded_rate(ans_tokens, ctx_tokens)
    lexical_scores.append(lexical_score)
    tier_stats[tier]["lexical_scores"].append(lexical_score)

    # Save per-query result with unified schema matching manual_grades.py
    per_query_results.append({
        "id": rec_id,
        "question_id": rec_id,
        "tier": tier,
        # Native metric names
        "retrieval_hit": bool(retrieval_hit),
        "citation_correct": bool(citation_correct),
        "citation_consistent": bool(citation_consistent),
        "structural_grounded": bool(structural_grounded),
        "lexical_grounded_rate": round(float(lexical_score), 3),
        # Consumer keys expected by manual_grades.py / generation_eval_report.json
        "auto_grounded_rate": 1.0 if structural_grounded else 0.0,
        "auto_citation_accuracy": 1.0 if citation_correct else 0.0,
        "auto_context_utilization": round(float(lexical_score), 3),
        "auto_hallucination_risk": 0.0 if structural_grounded else 1.0,
        # Methodological reliability status
        "metric_status": {
            "grounded_rate": "legacy_structural_heuristic",
            "citation_accuracy": "legacy_chapter_presence_proxy",
            "citation_consistency": "legacy_tautological_proxy",
            "context_utilization": "lexical_overlap_diagnostic",
            "hallucination_risk": "inverted_structural_heuristic",
        }
    })

# ------------------------------
# Final Metrics
# ------------------------------

retrieval_rate = retrieval_hits / total if total else 0
citation_rate = citation_accuracy / total if total else 0
consistency_rate = citation_consistency / total if total else 0
struct_grounded_rate = struct_grounded / total if total else 0
hallucination_rate = 1.0 - struct_grounded_rate
lexical_avg = float(np.mean(lexical_scores)) if lexical_scores else 0.0

print("\n" + "═" * 60)
print("MEDIRAG — AUTOMATED METRICS SUMMARY")
print("═" * 60)

print("\nDIAGNOSTIC UTILIZATION METRICS (Full Retrieved Evidence)")
print(f"Context Utilization (Lexical Overlap) : {lexical_avg:.2%}")
print(f"Retrieval Hit Rate                   : {retrieval_rate:.2%}")

print("\nLEGACY HEURISTIC METRICS (Methodologically Deprecated)")
print(f"Citation Accuracy (Chapter proxy)    : {citation_rate:.2%}")
print(f"Citation Consistency (Tautological)  : {consistency_rate:.2%}")
print(f"Structural Grounded Rate (Heuristic) : {struct_grounded_rate:.2%}")
print(f"Hallucination Rate (1 - Structural)  : {hallucination_rate:.2%}")

print("\nTier-wise Breakdown:")
for tier in sorted(tier_stats.keys()):
    count = tier_stats[tier]["count"]
    correct = tier_stats[tier]["citation_correct"]
    tier_rate = correct / count if count else 0
    tier_lex = float(np.mean(tier_stats[tier]["lexical_scores"])) if tier_stats[tier]["lexical_scores"] else 0.0
    print(f"  Tier {tier}: CitAcc={tier_rate:.2%} | CtxUtil={tier_lex:.2%} (n={count})")

# ------------------------------
# Save Output
# ------------------------------

output = {
    "schema_version": "2.0",
    "summary": {
        "retrieval_hit_rate": round(retrieval_rate, 3),
        "citation_accuracy": round(citation_rate, 3),
        "citation_consistency": round(consistency_rate, 3),
        "structural_grounded_rate": round(struct_grounded_rate, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "context_utilization_rate": round(float(lexical_avg), 3),
        "lexical_grounded_rate": round(float(lexical_avg), 3),
    },
    "reliability_notes": {
        "citation_accuracy": "DEPRECATED / UNRELIABLE: Measures whether expected chapter was cited; does not verify sentence-level claim attribution.",
        "citation_consistency": "DEPRECATED / UNRELIABLE: Measures whether cited indices are within retrieved set; largely tautological.",
        "structural_grounded_rate": "DEPRECATED / UNRELIABLE: Conjunction of retrieval hit, chapter-presence citation, and consistency. Not a claim-level NLI grounding metric.",
        "hallucination_rate": "DEPRECATED / UNRELIABLE: Defined as (1 - structural_grounded_rate); not a claim-level hallucination metric.",
        "context_utilization_rate": "DIAGNOSTIC: Lexical unigram overlap between generated answer and full retrieved context."
    },
    "per_query": per_query_results,
    "per_question": per_query_results,   # Alias for backward compatibility
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved → {OUTPUT_FILE}\n")