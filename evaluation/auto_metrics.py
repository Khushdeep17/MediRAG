import json
import re
import numpy as np
from collections import defaultdict
import os

os.makedirs("evaluation", exist_ok=True)

INPUT_FILE  = "evaluation/generation_outputs.json"
OUTPUT_FILE = "evaluation/auto_metrics_results.json"

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

# ------------------------------
# Load Data
# ------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)

retrieval_hits = 0
citation_accuracy = 0
citation_consistency = 0
struct_grounded = 0

lexical_scores = []
tier_stats = defaultdict(lambda: {"count": 0, "citation_correct": 0})

per_question_results = []

# ------------------------------
# Main Loop
# ------------------------------

for idx, item in enumerate(data):

    expected = item["expected_chapter"]
    retrieved = set(item["retrieved_chapters"])
    cited = set(item["cited_actual_chapters"])
    tier = item["tier"]

    tier_stats[tier]["count"] += 1

    # 1️⃣ Retrieval Hit
    retrieval_hit = expected in retrieved
    if retrieval_hit:
        retrieval_hits += 1

    # 2️⃣ Citation Accuracy
    citation_correct = item["expected_chapter_cited"]
    if citation_correct:
        citation_accuracy += 1
        tier_stats[tier]["citation_correct"] += 1

    # 3️⃣ Citation Consistency
    citation_consistent = cited.issubset(retrieved)
    if citation_consistent:
        citation_consistency += 1

    # 4️⃣ Structural Grounded
    structural_grounded = (
        retrieval_hit and
        citation_correct and
        citation_consistent
    )

    if structural_grounded:
        struct_grounded += 1

    # 5️⃣ Lexical Metric
    answer = item.get("generated_answer", "")
    chunks = item.get("retrieved_chunks", [])
    context_text = " ".join(c.get("content_snippet", "") for c in chunks)

    ans_tokens = tokenize(answer)
    ctx_tokens = tokenize(context_text)

    lexical_score = lexical_grounded_rate(ans_tokens, ctx_tokens)
    lexical_scores.append(lexical_score)

    # Save per-question result
    per_question_results.append({
        "question_id": item.get("question_id", idx),
        "tier": tier,
        "retrieval_hit": retrieval_hit,
        "citation_correct": citation_correct,
        "citation_consistent": citation_consistent,
        "structural_grounded": structural_grounded,
        "lexical_grounded_rate": round(float(lexical_score), 3)
    })

# ------------------------------
# Final Metrics
# ------------------------------

retrieval_rate = retrieval_hits / total if total else 0
citation_rate = citation_accuracy / total if total else 0
consistency_rate = citation_consistency / total if total else 0
struct_grounded_rate = struct_grounded / total if total else 0
hallucination_rate = 1 - struct_grounded_rate
lexical_avg = np.mean(lexical_scores) if lexical_scores else 0

print("\n" + "═" * 60)
print("MEDIRAG — FINAL AUTOMATED METRICS")
print("═" * 60)

print("\nPRIMARY METRICS")
print(f"Retrieval@5 Hit Rate        : {retrieval_rate:.2%}")
print(f"Citation Accuracy           : {citation_rate:.2%}")
print(f"Citation Consistency        : {consistency_rate:.2%}")
print(f"Structural Grounded Rate    : {struct_grounded_rate:.2%}")
print(f"Hallucination Rate          : {hallucination_rate:.2%}")

print("\nSECONDARY DIAGNOSTIC METRIC")
print(f"Lexical Grounded Rate (avg) : {lexical_avg:.2%}")

print("\nTier-wise Citation Accuracy:")
for tier in sorted(tier_stats.keys()):
    count = tier_stats[tier]["count"]
    correct = tier_stats[tier]["citation_correct"]
    tier_rate = correct / count if count else 0
    print(f"  Tier {tier}: {tier_rate:.2%}")

# ------------------------------
# Save Output
# ------------------------------

output = {
    "summary": {
        "retrieval_hit_rate": round(retrieval_rate, 3),
        "citation_accuracy": round(citation_rate, 3),
        "citation_consistency": round(consistency_rate, 3),
        "structural_grounded_rate": round(struct_grounded_rate, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "lexical_grounded_rate": round(float(lexical_avg), 3),
    },
    "per_question": per_question_results
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved → {OUTPUT_FILE}\n")