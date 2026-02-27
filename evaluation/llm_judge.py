import os
import sys
import json
import re
import time
import numpy as np
from collections import defaultdict

# --- Fix import path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE   = "evaluation/generation_outputs.json"
OUTPUT_FILE  = "evaluation/llm_judge_results.json"
JUDGE_MODEL  = "qwen/qwen3-32b"
RETRY_DELAY  = 2     # seconds between retries on API failure
MAX_RETRIES  = 3

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =====================================================
# JUDGE PROMPT
# =====================================================

JUDGE_SYSTEM = """\
You are a strict medical evaluation expert. You evaluate AI-generated medical answers \
against their source context. You always respond with ONLY a valid JSON object — \
no explanation, no markdown, no preamble.\
"""

def build_judge_prompt(query: str, context: str, answer: str) -> str:
    return f"""Evaluate this AI-generated medical answer against the provided source context.

Score each dimension from 1 to 5:

**Faithfulness** (1–5)
How well does the answer stay within the information provided in the context?
5 = Every claim is directly supported by context.
1 = Answer contains significant information not present in context.

**Completeness** (1–5)
How thoroughly does the answer address the question given the available context?
5 = All relevant aspects from context are covered.
1 = Major relevant information from context is missing.

**Medical Accuracy** (1–5)
How medically sound is the answer based on the context provided?
5 = Medically precise with correct terminology and relationships.
1 = Contains medical errors or misleading statements.

Also provide:
**reasoning**: One sentence explaining the main strength or weakness.
**hallucination_flag**: true if answer contains claims clearly absent from context, else false.

---

CONTEXT (retrieved medical text):
{context[:1500]}

QUESTION:
{query}

GENERATED ANSWER:
{answer[:1200]}

---

Respond with ONLY this JSON (no other text):
{{
  "faithfulness": <int 1-5>,
  "completeness": <int 1-5>,
  "medical_accuracy": <int 1-5>,
  "reasoning": "<one sentence>",
  "hallucination_flag": <true|false>
}}"""

# =====================================================
# CALL JUDGE
# =====================================================

def call_judge(query: str, context: str, answer: str) -> dict | None:
    """Call LLM judge with retries. Returns parsed dict or None on failure."""

    prompt = build_judge_prompt(query, context, answer)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,    # Near-deterministic for consistent scoring
                max_tokens=300,
            )

            raw = response.choices[0].message.content.strip()

            # Strip <think> blocks if model emits them
            if "<think>" in raw and "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            # Strip markdown fences
            raw = re.sub(r"```(?:json)?", "", raw).strip()

            parsed = json.loads(raw)

            # Validate expected keys
            required = {"faithfulness", "completeness", "medical_accuracy",
                        "reasoning", "hallucination_flag"}
            if not required.issubset(parsed.keys()):
                raise ValueError(f"Missing keys: {required - parsed.keys()}")

            # Clamp scores to 1–5
            for key in ["faithfulness", "completeness", "medical_accuracy"]:
                parsed[key] = max(1, min(5, int(parsed[key])))

            return parsed

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"   ⚠️  Parse error (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"   ⚠️  API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)

    print(f"   ❌ Failed after {MAX_RETRIES} attempts — skipping.")
    return None

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
    print(f"  MediRAG — LLM-as-Judge Evaluation")
    print(f"  Model  : {JUDGE_MODEL}")
    print(f"  Input  : {INPUT_FILE} ({len(records)} records)")
    print(f"{'═' * 60}\n")

    judged_records = []
    tier_scores    = defaultdict(lambda: defaultdict(list))
    failed         = 0

    for rec in records:
        idx    = rec["id"]
        query  = rec["query"]
        tier   = rec["tier"]
        answer = rec.get("generated_answer", "")
        chunks = rec.get("retrieved_chunks", [])

        context_text = "\n\n".join(
            f"[Source {i+1}] Ch.{c['chapter_number']} — {c['chapter_title']}\n{c['content_snippet']}"
            for i, c in enumerate(chunks)
        )

        print(f"[{idx:02d}/20] T{tier} | {query[:65]}")

        scores = call_judge(query, context_text, answer)

        if scores:
            rec_judged = {
                **rec,
                "llm_faithfulness"     : scores["faithfulness"],
                "llm_completeness"     : scores["completeness"],
                "llm_medical_accuracy" : scores["medical_accuracy"],
                "llm_reasoning"        : scores["reasoning"],
                "llm_hallucination_flag": scores["hallucination_flag"],
                "llm_avg_score"        : round(
                    (scores["faithfulness"] + scores["completeness"] + scores["medical_accuracy"]) / 3, 2
                ),
            }

            f_score = scores["faithfulness"]
            c_score = scores["completeness"]
            m_score = scores["medical_accuracy"]
            hall    = "🚩" if scores["hallucination_flag"] else "  "

            print(f"       Faith={f_score}/5  Complete={c_score}/5  MedAcc={m_score}/5  {hall}")
            print(f"       {scores['reasoning'][:80]}")

            for key in ["faithfulness", "completeness", "medical_accuracy"]:
                tier_scores[tier][key].append(scores[key])

        else:
            rec_judged = {**rec, "llm_faithfulness": None, "llm_completeness": None,
                          "llm_medical_accuracy": None, "llm_reasoning": "FAILED",
                          "llm_hallucination_flag": None, "llm_avg_score": None}
            failed += 1

        judged_records.append(rec_judged)
        print()
        time.sleep(0.5)   # polite rate limiting

    # ── Aggregate ──────────────────────────────────────────────────────────────
    valid = [r for r in judged_records if r["llm_faithfulness"] is not None]

    faith_scores = [r["llm_faithfulness"]     for r in valid]
    comp_scores  = [r["llm_completeness"]      for r in valid]
    macc_scores  = [r["llm_medical_accuracy"]  for r in valid]
    avg_scores   = [r["llm_avg_score"]         for r in valid]
    hall_flags   = [r["llm_hallucination_flag"] for r in valid]

    print(f"{'═' * 60}")
    print(f"  LLM JUDGE SUMMARY  (n={len(valid)}/{len(records)} scored)")
    print(f"{'═' * 60}")
    print(f"  Faithfulness     : {np.mean(faith_scores):.2f} / 5.0")
    print(f"  Completeness     : {np.mean(comp_scores):.2f} / 5.0")
    print(f"  Medical Accuracy : {np.mean(macc_scores):.2f} / 5.0")
    print(f"  Average Score    : {np.mean(avg_scores):.2f} / 5.0")
    print(f"  Hallucination 🚩 : {sum(hall_flags)}/{len(valid)} answers flagged "
          f"({sum(hall_flags)/len(valid)*100:.0f}%)")
    if failed:
        print(f"  ⚠️  Failed queries  : {failed}")
    print(f"\n  ── By Tier ────────────────────────────────────────")
    for t in [1, 2, 3]:
        tname = {1:"Direct", 2:"Indirect", 3:"Hard"}[t]
        ts    = tier_scores[t]
        if ts["faithfulness"]:
            print(f"  Tier {t} {tname:<10}  "
                  f"Faith={np.mean(ts['faithfulness']):.2f}  "
                  f"Complete={np.mean(ts['completeness']):.2f}  "
                  f"MedAcc={np.mean(ts['medical_accuracy']):.2f}")
    print(f"{'═' * 60}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "summary": {
            "n_scored"           : len(valid),
            "n_failed"           : failed,
            "faithfulness"       : round(float(np.mean(faith_scores)), 2),
            "completeness"       : round(float(np.mean(comp_scores)), 2),
            "medical_accuracy"   : round(float(np.mean(macc_scores)), 2),
            "avg_score"          : round(float(np.mean(avg_scores)), 2),
            "hallucination_count": int(sum(hall_flags)),
            "hallucination_rate" : round(float(sum(hall_flags) / len(valid)), 3),
            "by_tier": {
                str(t): {
                    "faithfulness"    : round(float(np.mean(tier_scores[t]["faithfulness"])), 2),
                    "completeness"    : round(float(np.mean(tier_scores[t]["completeness"])), 2),
                    "medical_accuracy": round(float(np.mean(tier_scores[t]["medical_accuracy"])), 2),
                }
                for t in [1, 2, 3] if tier_scores[t]["faithfulness"]
            }
        },
        "per_query": judged_records,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved → {OUTPUT_FILE}")
    print(f"   Next: python evaluation/manual_grades.py\n")


if __name__ == "__main__":
    main()
