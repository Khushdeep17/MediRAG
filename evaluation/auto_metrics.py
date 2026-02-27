import os
import sys
import json
import re
import numpy as np
from collections import defaultdict

# --- Fix import path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE  = "evaluation/generation_outputs.json"
OUTPUT_FILE = "evaluation/auto_metrics_results.json"

# Medical stopwords — not useful for grounding checks
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "or", "and", "but", "not",
    "this", "that", "these", "those", "it", "its", "which", "who",
    "what", "how", "when", "where", "patient", "treatment", "disease",
    "condition", "symptoms", "cause", "causes", "used", "also", "may",
    "often", "typically", "include", "including", "such", "both",
}

MIN_TERM_LENGTH = 4   # ignore very short tokens

# =====================================================
# HELPERS
# =====================================================

def tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, filter stopwords and short tokens."""
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) >= MIN_TERM_LENGTH}


def grounded_rate(answer_tokens: set, context_tokens: set) -> float:
    """
    What fraction of answer's medical terms appear in the retrieved context?
    Score of 1.0 = fully grounded, 0.0 = completely ungrounded.
    """
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def context_utilization(answer_tokens: set, context_tokens: set) -> float:
    """
    What fraction of the retrieved context was actually used in the answer?
    High = model used most of the context.
    Low  = model used only a small slice.
    """
    if not context_tokens:
        return 0.0
    used = answer_tokens & context_tokens
    return len(used) / len(context_tokens)


def citation_accuracy(record: dict) -> float:
    """
    Did the model cite the correct (expected) source chapter?
    Binary: 1.0 if expected chapter appeared in cited chapters, else 0.
    """
    return float(record.get("expected_chapter_cited", False))


def answer_length_score(answer: str) -> dict:
    """Basic answer quality proxy — word count + section headers."""
    words    = len(answer.split())
    headers  = len(re.findall(r'#{1,3}\s+\w+', answer))
    bullets  = len(re.findall(r'^\s*[-*]', answer, re.MULTILINE))
    return {"word_count": words, "header_count": headers, "bullet_count": bullets}


def hallucination_risk_score(grounded: float) -> str:
    """
    Map grounded rate to qualitative hallucination risk tier.
    Conservative thresholds for medical content.
    """
    if grounded >= 0.80: return "Low"
    elif grounded >= 0.65: return "Moderate"
    else: return "High"

# =====================================================
# MAIN
# =====================================================

def main():

    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Run generation_eval.py first.")
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        records = json.load(f)

    print(f"\n{'═' * 60}")
    print(f"  MediRAG — Automated Generation Metrics")
    print(f"  Input  : {INPUT_FILE} ({len(records)} records)")
    print(f"{'═' * 60}\n")

    scored_records  = []
    tier_scores     = defaultdict(lambda: defaultdict(list))

    for rec in records:
        query   = rec["query"]
        tier    = rec["tier"]
        answer  = rec.get("generated_answer", "")
        chunks  = rec.get("retrieved_chunks", [])

        # Build token sets
        answer_tokens  = tokenize(answer)
        context_text   = " ".join(c.get("content_snippet", "") for c in chunks)
        context_tokens = tokenize(context_text)

        # ── Compute metrics ────────────────────────────────────────────────────
        gr   = round(grounded_rate(answer_tokens, context_tokens), 3)
        cu   = round(context_utilization(answer_tokens, context_tokens), 3)
        ca   = citation_accuracy(rec)
        hall = hallucination_risk_score(gr)
        aq   = answer_length_score(answer)

        rec_scored = {
            **rec,
            "auto_grounded_rate"      : gr,
            "auto_context_utilization": cu,
            "auto_citation_accuracy"  : ca,
            "auto_hallucination_risk" : hall,
            "auto_answer_stats"       : aq,
        }
        scored_records.append(rec_scored)

        tier_scores[tier]["grounded_rate"].append(gr)
        tier_scores[tier]["context_utilization"].append(cu)
        tier_scores[tier]["citation_accuracy"].append(ca)

        # Per-query print
        hit_sym = "✅" if gr >= 0.80 else ("⚠️ " if gr >= 0.65 else "❌")
        print(f"[{rec['id']:02d}] T{tier} {hit_sym}  Grounded={gr:.3f}  CtxUtil={cu:.3f}  "
              f"CitAccuracy={int(ca)}  HallRisk={hall}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_gr  = [r["auto_grounded_rate"]       for r in scored_records]
    all_cu  = [r["auto_context_utilization"] for r in scored_records]
    all_ca  = [r["auto_citation_accuracy"]   for r in scored_records]

    print(f"\n{'═' * 60}")
    print(f"  AUTOMATED METRICS SUMMARY  (n={len(records)})")
    print(f"{'═' * 60}")
    print(f"  Grounded Rate        : {np.mean(all_gr):.3f}  (target: ≥ 0.85)")
    print(f"  Context Utilization  : {np.mean(all_cu):.3f}")
    print(f"  Citation Accuracy    : {np.mean(all_ca):.3f}  (target: ≥ 0.90)")
    print(f"  Hallucination Rate   : {1 - np.mean(all_gr):.3f}  (target: ≤ 0.15)")
    print(f"\n  ── By Tier ────────────────────────────────────────")
    for tier in [1, 2, 3]:
        tname = {1:"Direct", 2:"Indirect", 3:"Hard"}[tier]
        ts    = tier_scores[tier]
        if ts["grounded_rate"]:
            print(f"  Tier {tier} {tname:<10} "
                  f"Grounded={np.mean(ts['grounded_rate']):.3f}  "
                  f"CitAcc={np.mean(ts['citation_accuracy']):.3f}")
    print(f"{'═' * 60}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "summary": {
            "n_queries"           : len(records),
            "grounded_rate"       : round(float(np.mean(all_gr)), 3),
            "context_utilization" : round(float(np.mean(all_cu)), 3),
            "citation_accuracy"   : round(float(np.mean(all_ca)), 3),
            "hallucination_rate"  : round(float(1 - np.mean(all_gr)), 3),
            "by_tier": {
                str(t): {
                    "grounded_rate"     : round(float(np.mean(tier_scores[t]["grounded_rate"])), 3),
                    "citation_accuracy" : round(float(np.mean(tier_scores[t]["citation_accuracy"])), 3),
                }
                for t in [1, 2, 3] if tier_scores[t]["grounded_rate"]
            }
        },
        "per_query": scored_records,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved → {OUTPUT_FILE}")
    print(f"   Next: python evaluation/llm_judge.py\n")


if __name__ == "__main__":
    main()
