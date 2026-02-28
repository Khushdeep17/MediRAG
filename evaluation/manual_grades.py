"""
manual_grades.py

Interactive CLI to fill in your manual accuracy grades (0/1/2) for each generated answer.
Run AFTER generation_eval.py has produced generation_outputs.json.

Grades:
  0 = Wrong / medically incorrect
  1 = Partially correct (right direction, missing key info)
  2 = Fully correct

After grading, computes final combined metrics report.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

# --- Fix import path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# =====================================================
# CONFIG
# =====================================================

GEN_FILE        = "evaluation/generation_outputs.json"
AUTO_FILE       = "evaluation/auto_metrics_results.json"
JUDGE_FILE      = "evaluation/llm_judge_results.json"
GRADES_FILE     = "evaluation/manual_grades.json"
FINAL_REPORT    = "evaluation/generation_eval_report.json"

# =====================================================
# GRADING MODE — set to False to just load existing
# =====================================================

INTERACTIVE = True   # Set False if grades.json already filled

# =====================================================
# INTERACTIVE GRADING
# =====================================================

def run_interactive_grading(records: list) -> dict:
    """Walk through each answer and collect a 0/1/2 grade from the user."""

    existing = {}
    if os.path.exists(GRADES_FILE):
        with open(GRADES_FILE) as f:
            existing = json.load(f)
        already_graded = sum(1 for v in existing.values() if v is not None)
        if already_graded > 0:
            resume = input(f"\n⚡ Found {already_graded} existing grades. Resume? (y/n): ").strip().lower()
            if resume != 'y':
                existing = {}

    grades = dict(existing)

    print(f"\n{'═'*65}")
    print("  MANUAL GRADING — Read each answer and score 0 / 1 / 2")
    print("  0 = Incorrect   1 = Partially correct   2 = Fully correct")
    print("  Press Enter without typing to skip (keep existing grade).")
    print(f"{'═'*65}\n")

    for rec in records:
        idx   = str(rec["id"])
        query = rec["query"]
        ans   = rec.get("generated_answer", "")
        tier  = rec["tier"]

        if idx in grades and grades[idx] is not None:
            print(f"[{rec['id']:02d}/20] T{tier} ✓ Already graded: {grades[idx]}  |  {query[:60]}")
            continue

        print(f"\n{'─'*65}")
        print(f"[{rec['id']:02d}/20]  TIER {tier}")
        print(f"QUERY  : {query}")
        print(f"{'─'*65}")
        print(f"ANSWER :\n{ans}")
        
        while True:
            raw = input("Grade (0/1/2) or 's' to skip: ").strip().lower()
            if raw == 's' or raw == '':
                grades[idx] = None
                break
            if raw in ('0', '1', '2'):
                grades[idx] = int(raw)
                break
            print("  ⚠️  Enter 0, 1, 2, or 's'")

        # Save after each grade so progress is never lost
        with open(GRADES_FILE, "w") as f:
            json.dump(grades, f, indent=2)

    return grades

# =====================================================
# COMBINED REPORT
# =====================================================

def build_final_report(records, grades, auto_data, judge_data):
    """Merge manual grades + auto metrics + LLM judge into one report."""

    # Build lookup dicts
    auto_lookup  = {str(r["id"]): r for r in auto_data.get("per_query", [])}
    judge_lookup = {str(r["id"]): r for r in judge_data.get("per_query", [])}

    combined = []
    tier_stats = defaultdict(lambda: defaultdict(list))

    for rec in records:
        idx    = str(rec["id"])
        grade  = grades.get(idx)
        auto   = auto_lookup.get(idx, {})
        judge  = judge_lookup.get(idx, {})
        tier   = rec["tier"]

        entry = {
            "id"                    : rec["id"],
            "query"                 : rec["query"],
            "tier"                  : tier,
            "expected_chapter"      : rec["expected_chapter"],
            # Manual
            "manual_accuracy_grade" : grade,
            # Auto
            "grounded_rate"         : auto.get("auto_grounded_rate"),
            "context_utilization"   : auto.get("auto_context_utilization"),
            "citation_accuracy"     : auto.get("auto_citation_accuracy"),
            "hallucination_risk"    : auto.get("auto_hallucination_risk"),
            # LLM Judge
            "llm_faithfulness"      : judge.get("llm_faithfulness"),
            "llm_completeness"      : judge.get("llm_completeness"),
            "llm_medical_accuracy"  : judge.get("llm_medical_accuracy"),
            "llm_avg_score"         : judge.get("llm_avg_score"),
            "llm_hallucination_flag": judge.get("llm_hallucination_flag"),
        }
        combined.append(entry)

        # Accumulate for tier stats
        if grade is not None:
            tier_stats[tier]["manual_accuracy"].append(grade)
        if entry["grounded_rate"] is not None:
            tier_stats[tier]["grounded_rate"].append(entry["grounded_rate"])
        if entry["llm_faithfulness"] is not None:
            tier_stats[tier]["llm_faithfulness"].append(entry["llm_faithfulness"])

    # ── Overall stats ──────────────────────────────────────────────────────────
    graded = [e for e in combined if e["manual_accuracy_grade"] is not None]
    auto_valid = [e for e in combined if e["grounded_rate"] is not None]
    judge_valid = [e for e in combined if e["llm_faithfulness"] is not None]

    manual_scores = [e["manual_accuracy_grade"] for e in graded]
    answer_accuracy_pct = round(sum(manual_scores) / (len(manual_scores) * 2) * 100, 1) if manual_scores else None
    fully_correct_pct   = round(sum(1 for s in manual_scores if s == 2) / len(manual_scores) * 100, 1) if manual_scores else None

    summary = {
        "n_total"             : len(records),
        "n_graded_manual"     : len(graded),
        # Manual
        "answer_accuracy_pct" : answer_accuracy_pct,
        "fully_correct_pct"   : fully_correct_pct,
        # Auto
        "grounded_rate"       : round(float(np.mean([e["grounded_rate"] for e in auto_valid])), 3) if auto_valid else None,
        "citation_accuracy"   : round(float(np.mean([e["citation_accuracy"] for e in auto_valid])), 3) if auto_valid else None,
        "hallucination_rate"  : round(1 - float(np.mean([e["grounded_rate"] for e in auto_valid])), 3) if auto_valid else None,
        # LLM Judge
        "llm_faithfulness"    : round(float(np.mean([e["llm_faithfulness"] for e in judge_valid])), 2) if judge_valid else None,
        "llm_completeness"    : round(float(np.mean([e["llm_completeness"] for e in judge_valid])), 2) if judge_valid else None,
        "llm_medical_accuracy": round(float(np.mean([e["llm_medical_accuracy"] for e in judge_valid])), 2) if judge_valid else None,
        # Tier breakdown
        "by_tier": {}
    }

    for t in [1, 2, 3]:
        ts = tier_stats[t]
        summary["by_tier"][str(t)] = {
            "manual_accuracy_pct" : round(sum(ts["manual_accuracy"]) / (len(ts["manual_accuracy"]) * 2) * 100, 1)
                                    if ts["manual_accuracy"] else None,
            "grounded_rate"       : round(float(np.mean(ts["grounded_rate"])), 3) if ts["grounded_rate"] else None,
            "llm_faithfulness"    : round(float(np.mean(ts["llm_faithfulness"])), 2) if ts["llm_faithfulness"] else None,
        }

    return {"summary": summary, "per_query": combined}

# =====================================================
# PRINT FINAL SUMMARY
# =====================================================

def print_final_summary(report):
    s = report["summary"]
    print(f"\n{'╔' + '═'*62 + '╗'}")
    print(f"║  {'MediRAG — Generation Evaluation Final Report':<60}║")
    print(f"{'╠' + '═'*62 + '╣'}")
    print(f"║  {'MANUAL GRADING':<40}{'':>20}║")
    if s["answer_accuracy_pct"] is not None:
        print(f"║  Answer Accuracy   : {s['answer_accuracy_pct']:>5.1f}%  (0/1/2 scale){'':<22}║")
        print(f"║  Fully Correct     : {s['fully_correct_pct']:>5.1f}%{'':<33}║")
    print(f"{'╠' + '═'*62 + '╣'}")
    print(f"║  {'AUTOMATED METRICS':<40}{'':>20}║")
    if s["grounded_rate"]:
        print(f"║  Grounded Rate     : {s['grounded_rate']:.3f}{'':<35}║")
        print(f"║  Citation Accuracy : {s['citation_accuracy']:.3f}{'':<35}║")
        print(f"║  Hallucination Rate: {s['hallucination_rate']:.3f}{'':<35}║")
    print(f"{'╠' + '═'*62 + '╣'}")
    print(f"║  {'LLM-AS-JUDGE':<40}{'':>20}║")
    if s["llm_faithfulness"]:
        print(f"║  Faithfulness      : {s['llm_faithfulness']:.2f} / 5.0{'':<31}║")
        print(f"║  Completeness      : {s['llm_completeness']:.2f} / 5.0{'':<31}║")
        print(f"║  Medical Accuracy  : {s['llm_medical_accuracy']:.2f} / 5.0{'':<31}║")
    print(f"{'╚' + '═'*62 + '╝'}\n")

# =====================================================
# MAIN
# =====================================================

def main():

    if not os.path.exists(GEN_FILE):
        print(f"❌ {GEN_FILE} not found. Run generation_eval.py first.")
        sys.exit(1)

    with open(GEN_FILE, encoding="utf-8") as f:
        records = json.load(f)

    # Load auto + judge results (optional — report works without them)
    auto_data  = json.load(open(AUTO_FILE))  if os.path.exists(AUTO_FILE)  else {"per_query": []}
    judge_data = json.load(open(JUDGE_FILE)) if os.path.exists(JUDGE_FILE) else {"per_query": []}

    if INTERACTIVE:
        grades = run_interactive_grading(records)
    else:
        if not os.path.exists(GRADES_FILE):
            print(f"❌ {GRADES_FILE} not found and INTERACTIVE=False.")
            sys.exit(1)
        with open(GRADES_FILE) as f:
            grades = json.load(f)

    # Build and save final report
    report = build_final_report(records, grades, auto_data, judge_data)

    with open(FINAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_final_summary(report)
    print(f"✅ Final report saved → {FINAL_REPORT}")
    print(f"   Now run the evaluation notebook to visualize everything.\n")


if __name__ == "__main__":
    main()
