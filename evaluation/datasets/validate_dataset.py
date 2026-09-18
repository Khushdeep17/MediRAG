"""
validate_dataset.py

Automated Quality-Control Validator for MediRAG Evaluation Datasets.
Verifies structural schema constraints, chunk ID existence in corpus,
evidence span text substring alignment, and clinical integrity rules.
"""

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "merck_chunks_800_150.json"


class DatasetValidator:
    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks_path = chunks_path
        self._corpus_chunks: Dict[str, Dict[str, Any]] = {}
        self._load_corpus()

    def _load_corpus(self):
        if not self.chunks_path.exists():
            print(f"⚠️ Warning: Corpus chunks not found at {self.chunks_path}. Skipping chunk existence verification.")
            return
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunk_list = json.load(f)
        for c in chunk_list:
            self._corpus_chunks[c["chunk_id"]] = c

    def validate_retrieval_dataset_v2(self, dataset_path: Path) -> List[str]:
        """Validate a Retrieval Dataset v2 JSON file."""
        errors = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for req_key in ["dataset_name", "dataset_version", "query_count", "queries"]:
            if req_key not in data:
                errors.append(f"Missing required top-level key: '{req_key}'")

        queries = data.get("queries", [])
        if len(queries) != data.get("query_count"):
            errors.append(f"Header query_count ({data.get('query_count')}) != actual queries length ({len(queries)})")

        seen_qids = set()
        seen_texts = set()

        for i, q in enumerate(queries, 1):
            qid = q.get("id")
            if qid is None:
                errors.append(f"Query index {i} missing 'id'")
                continue
            if qid in seen_qids:
                errors.append(f"Duplicate query id: {qid}")
            seen_qids.add(qid)

            query_text = q.get("query", "").strip()
            if not query_text:
                errors.append(f"Query {qid}: Empty query text")
            if query_text in seen_texts:
                errors.append(f"Query {qid}: Duplicate query text: '{query_text}'")
            seen_texts.add(query_text)

            tier = q.get("tier")
            if tier not in (1, 2, 3):
                errors.append(f"Query {qid}: Invalid tier {tier} (must be 1, 2, or 3)")

            exp_chaps = q.get("expected_chapters", [])
            if not exp_chaps or not isinstance(exp_chaps, list):
                errors.append(f"Query {qid}: 'expected_chapters' must be a non-empty list of integers")

            # Validate relevant_chunks if annotated
            rel_chunks = q.get("relevant_chunks", [])
            seen_chunk_ids = set()
            for rc in rel_chunks:
                cid = rc.get("chunk_id")
                if not cid:
                    errors.append(f"Query {qid}: Missing chunk_id in relevant_chunks")
                    continue
                if cid in seen_chunk_ids:
                    errors.append(f"Query {qid}: Duplicate chunk_id in relevant_chunks: '{cid}'")
                seen_chunk_ids.add(cid)

                rel_score = rc.get("relevance")
                if rel_score not in (0, 1, 2):
                    errors.append(f"Query {qid}, chunk {cid}: Relevance score must be 0, 1, or 2, got {rel_score}")

                if self._corpus_chunks:
                    if cid not in self._corpus_chunks:
                        errors.append(f"Query {qid}: Chunk '{cid}' does not exist in corpus")
                    else:
                        c_data = self._corpus_chunks[cid]
                        c_chap = c_data.get("chapter_number")
                        # If relevance is 2, it is expected to align with expected_chapters
                        if rel_score == 2 and exp_chaps and c_chap not in exp_chaps:
                            errors.append(f"Query {qid}: Chunk '{cid}' has chapter {c_chap} which is not in expected_chapters {exp_chaps}")

            # Validate evidence_spans
            spans = q.get("evidence_spans", [])
            for sp in spans:
                scid = sp.get("chunk_id")
                stext = sp.get("text", "").strip()
                if not scid:
                    errors.append(f"Query {qid}: Missing chunk_id in evidence_spans")
                    continue
                if not stext:
                    errors.append(f"Query {qid}: Empty text in evidence_spans for chunk {scid}")
                    continue

                if self._corpus_chunks and scid in self._corpus_chunks:
                    c_content = self._corpus_chunks[scid].get("content", "").lower()
                    if stext.lower() not in c_content:
                        errors.append(f"Query {qid}: Evidence span text '{stext[:40]}...' is not a substring of chunk {scid}")

                start = sp.get("approx_char_start")
                end = sp.get("approx_char_end")
                if start is not None and end is not None:
                    if start >= end:
                        errors.append(f"Query {qid}: approx_char_start ({start}) >= approx_char_end ({end}) in chunk {scid}")

        return errors

    def validate_generation_dataset_v2(self, dataset_path: Path) -> List[str]:
        """Validate a Generation Dataset v2 JSON file."""
        errors = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for req_key in ["dataset_name", "dataset_version", "query_count", "queries"]:
            if req_key not in data:
                errors.append(f"Missing required top-level key: '{req_key}'")

        queries = data.get("queries", [])
        if len(queries) != data.get("query_count"):
            errors.append(f"Header query_count ({data.get('query_count')}) != actual queries length ({len(queries)})")

        seen_qids = set()
        seen_texts = set()

        for i, q in enumerate(queries, 1):
            qid = q.get("id")
            if qid is None:
                errors.append(f"Query index {i} missing 'id'")
                continue
            if qid in seen_qids:
                errors.append(f"Duplicate query id: {qid}")
            seen_qids.add(qid)

            query_text = q.get("query", "").strip()
            if not query_text:
                errors.append(f"Query {qid}: Empty query text")
            if query_text in seen_texts:
                errors.append(f"Query {qid}: Duplicate query text: '{query_text}'")
            seen_texts.add(query_text)

            tier = q.get("tier")
            if tier not in (1, 2, 3):
                errors.append(f"Query {qid}: Invalid tier {tier}")

            exp_chaps = q.get("expected_chapters", [])
            if not exp_chaps or not isinstance(exp_chaps, list):
                errors.append(f"Query {qid}: 'expected_chapters' must be a non-empty list of integers")

            aspects = q.get("answer_aspects", [])
            seen_aspect_ids = set()
            for asp in aspects:
                aid = asp.get("aspect_id")
                if not aid:
                    errors.append(f"Query {qid}: Missing aspect_id")
                    continue
                if aid in seen_aspect_ids:
                    errors.append(f"Query {qid}: Duplicate aspect_id: '{aid}'")
                seen_aspect_ids.add(aid)

                crit = asp.get("criticality")
                if crit not in ("essential", "recommended", "safety_critical"):
                    errors.append(f"Query {qid}, aspect {aid}: Invalid criticality '{crit}'")

        return errors


def main():
    validator = DatasetValidator()
    ret_v2_path = PROJECT_ROOT / "evaluation" / "datasets" / "retrieval_queries_v2.json"
    gen_v2_path = PROJECT_ROOT / "evaluation" / "datasets" / "generation_queries_v2.json"

    print(f"Validating {ret_v2_path.name}...")
    errors_ret = validator.validate_retrieval_dataset_v2(ret_v2_path)
    if errors_ret:
        print(f"❌ Found {len(errors_ret)} retrieval validation errors:")
        for e in errors_ret[:10]:
            print(f"  • {e}")
        sys.exit(1)
    else:
        print(f"✅ {ret_v2_path.name} passed! 0 errors detected.")

    print(f"Validating {gen_v2_path.name}...")
    errors_gen = validator.validate_generation_dataset_v2(gen_v2_path)
    if errors_gen:
        print(f"❌ Found {len(errors_gen)} generation validation errors:")
        for e in errors_gen[:10]:
            print(f"  • {e}")
        sys.exit(1)
    else:
        print(f"✅ {gen_v2_path.name} passed! 0 errors detected.")


if __name__ == "__main__":
    main()
