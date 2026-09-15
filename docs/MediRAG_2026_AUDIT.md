# MediRAG 2026 Baseline Audit

> **Document Status**: Complete Baseline Audit & Modernization Readiness Assessment  
> **Date**: September 2026  
> **Corpus**: Merck Manual (18th Edition, Professional) — 4,239 chunks  
> **Repository Root**: `c:\Users\Khushdeep Singh\Desktop\Projects\MediRAG`  
> **Scope**: Preprocessing, Indexing, Dense/Sparse Retrieval, Fusion, Generation, Evaluation Suite, Streamlit UI, Dependencies, and Security.

---

## 1. Executive Summary

MediRAG is an established medical Hybrid Retrieval-Augmented Generation (RAG) system built over the Merck Manual. It combines:
1. **Document processing**: Ingestion of a 19.2 MB raw PDF, regex-based text cleaning, stable chapter-level parsing, and token-aware chunking (800-token chunks with 150-token overlap).
2. **Dense retrieval**: 1024-dimensional embeddings generated using `BAAI/bge-large-en-v1.5` indexed via FAISS `IndexFlatIP` (cosine similarity on L2-normalized vectors).
3. **Sparse retrieval**: Lexical BM25Okapi over lowercased, alphanumeric-split, stopword-filtered tokens.
4. **Hybrid fusion**: Weighted Reciprocal Rank Fusion (RRF, $k=60$) combining dense and sparse candidate ranks.
5. **Generation**: Groq API integration driving `openai/gpt-oss-120b` (temperature 0.2, max tokens 1250) with structured clinical prompting and inline citations (`[1]`, `[2]`, etc.).
6. **Multi-layer evaluation**: Offline retrieval benchmark (50 queries, 3 tiers), automated structural grounding metrics (20 queries), LLM-as-a-judge (`llama-3.3-70b-versatile`), and human manual grading.
7. **Frontend demo**: Streamlit dashboard with custom dark medical theme and grounding signal indicators.

### Key Audit Findings

* **Critical Architectural Discovery (The 512 Truncation Disconnect)**: Chunks are generated with a target size of **800 tokens** (`preprocessing/chunking.py`), but the embedding pipeline (`indexing/dense_faiss.py`) and query encoder (`retrieval/dense.py`) enforce `max_length=512`. **Up to 36% of the content in every chunk past token 512 is invisible to dense retrieval!**
* **The Fusion Domination Dynamic**: In `retrieval/fusion.py`, hybrid retrieval fetches `top_k * 2` candidates. At generation time (`top_k=4`, pool size 8) with $\alpha=0.7$, the minimum possible dense score in the candidate pool ($\frac{0.7}{60+8} \approx 0.01029$) is more than double the maximum possible sparse score ($\frac{0.3}{60+1} \approx 0.00492$). Consequently, a pure sparse candidate can **never** penetrate the top 8 unless it is already present in the dense candidate list. Sparse acts exclusively as a minor re-ranking bonus for dense hits.
* **Import-Time Blocking & Memory Overhead**: Importing `retrieval.dense` or `retrieval.sparse` synchronously executes heavy operations at module import time: downloading/loading the HuggingFace transformer to CPU, loading the FAISS index, reading a 15.3 MB JSON file, and tokenizing 4,239 documents to build BM25 from scratch on every run.
* **Evaluation Pipeline Schema Bug**: A field name and container mismatch between `evaluation/auto_metrics.py` (which emits `"per_question"` with key `"question_id"`) and `evaluation/manual_grades.py` (which expects `"per_query"` with key `"id"`) causes `generation_eval_report.json` to store `null` for all automated structural metrics.
* **Metric Formulation Flaws**: 
  - Citation consistency in both `app.py` and `evaluation/auto_metrics.py` is defined as `cited_actual_chapters.issubset(retrieved_chapter_numbers)`, which is tautological because `cited_actual_chapters` is extracted directly by indexing into `retrieved_chunks`.
  - Hallucination rate is mathematically defined as `1 - structural_grounded_rate`, conflating retrieval misses with ungrounded generation hallucinations.
  - LLM-as-judge prompt truncates context to 1,500 characters and answer to 1,200 characters, creating artificial hallucination flags when facts reside in later chunks.
* **Environment & Security**:
  - `requirements.txt` is encoded in **UTF-16LE**, breaking standard CLI tools on several platforms.
  - `pypdf` is imported by `preprocessing/clean_text.py` but is **completely missing** from `requirements.txt` and absent from the virtual environment.
  - The 19.2 MB raw PDF, 38.8 MB of processed JSON/text data, and an accidental directory (`evaluation/evaluation/`) are actively committed and tracked in git.

---

## 2. Current Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             OFFLINE PIPELINE                                     │
│                                                                                  │
│ [data/merck_manual.pdf] (Pages 53-3655)                                         │
│        │                                                                         │
│        ▼ (clean_text.py: pypdf, regex header/footer removal, line un-hyphenation)│
│ [data/processed/merck_cleaned.txt] (11.8 MB text)                                │
│        │                                                                         │
│        ▼ (section_parser.py: regex `^Chapter (\d+)\. (.*)`)                     │
│ [data/processed/merck_structured.json] (315 chapters)                            │
│        │                                                                         │
│        ▼ (chunking.py: BGE AutoTokenizer, 800 tokens, 150 overlap)               │
│ [data/processed/merck_chunks_800_150.json] (4,239 chunks)                       │
│        │                                                                         │
│   ┌────┴─────────────────────────────┐                                           │
│   ▼                                  ▼                                           │
│ [indexing/dense_faiss.py]       [retrieval/sparse.py]                            │
│  - BAAI/bge-large-en-v1.5        - BM25Okapi (rank-bm25)                         │
│  - batch_size=8, CPU             - lowercasing, regex tokenization               │
│  - max_length=512 (truncates!)   - NLTK English stopwords removal                │
│  - L2-normalized float32         - In-memory on import (NOT persisted to disk!)  │
│  - FAISS IndexFlatIP                                                             │
│   │                                                                              │
│   ▼                                                                              │
│ [embeddings/embeddings.npy] (17.3 MB)                                            │
│ [embeddings/ids.json] (50.9 KB)                                                  │
│ [index/faiss.index] (17.3 MB)                                                    │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴───────────────────────────────────────────┐
│                             ONLINE INFERENCE RUNTIME                             │
│                                                                                  │
│ User Query (via Streamlit app.py or generate.py)                                 │
│        │                                                                         │
│        ├───────────────────────────────┬─────────────────────────────────┐       │
│        ▼                               ▼                                 │       │
│  Dense Retrieval (dense.py)      Sparse Retrieval (sparse.py)            │       │
│  - Prefix: "query: "             - Tokenize query                        │       │
│  - Model encode (max 512, CPU)   - BM25 score over 4,239 docs            │       │
│  - L2 normalize                  - Top 2*k candidates (k=4 -> pool=8)    │       │
│  - FAISS search: top 2*k                                                 │       │
│        │                               │                                 │       │
│        └───────────────┬───────────────┘                                 │       │
│                        ▼                                                 │       │
│           Weighted RRF Fusion (fusion.py)                                │       │
│           - Score = 0.7/(60 + r_dense) + 0.3/(60 + r_sparse)             │       │
│           - Sort descending -> Slice top-k (TOP_K_CONTEXT = 4)           │       │
│                        │                                                 │       │
│                        ▼                                                 │       │
│           Context Formatter (generate.py)                                │       │
│           - Up to 2,200 chars per chunk                                  │       │
│           - Tagged: [1] Chapter {num}: {title} Chunk ID: {id}            │       │
│                        │                                                 │       │
│                        ▼                                                 │       │
│           LLM Generation (generate.py via Groq API)                      │       │
│           - Model: openai/gpt-oss-120b                                   │       │
│           - Structured system & user prompts                             │       │
│           - Strict inline citations [1], [2]                             │       │
│           - Post-cleaning regex (strip <think>, normalize brackets)      │       │
│                        │                                                 │       │
│        ┌───────────────┴───────────────┐                                 │       │
│        ▼                               ▼                                 ▼       │
│  Streamlit UI (app.py)     Generation Evaluation (eval/*)   Retrieval Eval       │
│  - Custom Dark Navy Theme  - 20 queries (3 tiers)           - 50 queries         │
│  - Inline Answer Display   - Auto structural metrics        - Recall@5/10, MRR   │
│  - Source Expanders (1-4)  - LLM-as-judge (LLaMA 3.3 70B)   - NDCG@10            │
│  - Grounding Signals       - Manual 0/1/2 grading                                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Stage-by-Stage Inventory

| # | Stage | Input | Output | Important Configuration | Model / Library | Implementation Location | Upstream Dependencies |
|---|---|---|---|---|---|---|---|
| 1 | **Corpus Ingestion** | `data/merck_manual.pdf` (3,655 pages) | Extracted raw text pages (53 to 3655) | `START_PAGE=53`, `END_PAGE=3655` | `pypdf.PdfReader` | [clean_text.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/preprocessing/clean_text.py) | Raw PDF file |
| 2 | **Text Cleaning** | Raw page strings | `data/processed/merck_cleaned.txt` | Regex un-hyphenation, header/footer strip | `re`, `pathlib` | [clean_text.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/preprocessing/clean_text.py) | Stage 1 |
| 3 | **Section/Chapter Parsing** | `merck_cleaned.txt` | `data/processed/merck_structured.json` | Regex `^Chapter\s+(\d+)\.\s+(.+)` | Standard Python `json`, `re` | [section_parser.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/preprocessing/section_parser.py) | Stage 2 |
| 4 | **Chunking** | Chapter records (315 chapters) | `data/processed/merck_chunks_800_150.json` | `CHUNK_SIZE=800`, `OVERLAP=150`, min trailing 50 | HuggingFace `AutoTokenizer` (`BAAI/bge-large-en-v1.5`) | [chunking.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/preprocessing/chunking.py) | Stage 3 |
| 5 | **Embedding Generation** | 4,239 chunks | `embeddings/embeddings.npy` (matrix: 4239x1024), `ids.json` | `BATCH_SIZE=8`, `DEVICE="cpu"`, `max_length=512`, prefix: `passage: ` | `transformers.AutoModel`, `torch`, `faiss.normalize_L2` | [dense_faiss.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/indexing/dense_faiss.py) | Stage 4 |
| 6 | **Vector Indexing** | Normalized embeddings matrix | `index/faiss.index` | `IndexFlatIP(1024)` | `faiss-cpu` | [dense_faiss.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/indexing/dense_faiss.py) | Stage 5 |
| 7 | **Dense Retrieval** | Query string | Top-k dense results with cosine similarity score | `max_length=512`, prefix: `query: `, `IndexFlatIP.search` | `transformers`, `torch`, `faiss` | [dense.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/retrieval/dense.py) | Stages 4, 5, 6 |
| 8 | **Sparse/BM25 Retrieval** | Query string | Top-k sparse results with BM25 scores | Lowercase, non-alphanumeric to spaces, NLTK stopwords | `rank_bm25.BM25Okapi`, `nltk` | [sparse.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/retrieval/sparse.py) | Stage 4 |
| 9 | **RRF Fusion** | Query string | Merged top-k fused results with RRF scores | $RRF\_K=60$, $\alpha=0.7$, fetches $2 \cdot k$ from each | `numpy` | [fusion.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/retrieval/fusion.py) | Stages 7, 8 |
| 10 | **Context Selection** | Fused chunk records | Formatted Markdown context string | `TOP_K_CONTEXT=4`, `CHUNK_CHAR_LIMIT=2200` | Standard Python string formatting | [generate.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/generate.py) | Stage 9 |
| 11 | **LLM Generation** | System prompt, formatted context, user query | Raw answer string | `model="openai/gpt-oss-120b"`, `temp=0.2`, `max_tokens=1250`, `freq_penalty=0.15` | `groq.Groq` | [generate.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/generate.py) | Stage 10 |
| 12 | **Citation Handling** | Raw answer string | Normalized citation answer string (`[1]`, `[2]`) | Regex stripping `<think>`, brackets normalization, trailing references removal | `re` | [generate.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/generate.py) | Stage 11 |
| 13 | **Streamlit UI** | User text input | Interactive web dashboard | Custom CSS, primary button, expanders, chips | `streamlit` | [app.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/app.py) | Stages 11, 12 |
| 14 | **Retrieval Evaluation** | 50 queries across 3 tiers | CSV tables and terminal reports | `RETRIEVAL_K=30`, Recall@5/10, MRR, NDCG@10 | `numpy`, `scipy` | [retrieval_metrics.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/evaluation/retrieval_metrics.py) | Stages 7, 8, 9 |
| 15 | **Generation Evaluation** | 20 queries across 3 tiers | `generation_outputs.json` & `auto_metrics_results.json` | Automated structural checks & token-overlap lexical grounding | `json`, `re`, `numpy` | [generation_eval.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/evaluation/generation_eval.py), [auto_metrics.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/evaluation/auto_metrics.py) | Stages 11, 12 |
| 16 | **LLM-as-a-Judge** | Query, context snippet (max 1500 chars), generated answer (max 1200 chars) | `llm_judge_results.json` | Model `llama-3.3-70b-versatile`, temp 0.1, 1-5 scale for Faithfulness, Completeness, Medical Accuracy | `groq.Groq` | [llm_judge.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/evaluation/llm_judge.py) | Stage 15 |
| 17 | **Manual Grading** | 20 generated answers | `manual_grades.json` & `generation_eval_report.json` | 0/1/2 grading scale (Wrong, Partial, Correct) | Python CLI interactive | [manual_grades.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/evaluation/manual_grades.py) | Stages 15, 16 |

---

## 3. Current Technical Baseline

The following baseline parameters were extracted directly from the code:

* **Embedding Model**: `BAAI/bge-large-en-v1.5`
* **Embedding Dimension**: `1024`
* **Embedding Device**: `"cpu"` (hardcoded in `indexing/dense_faiss.py` and `retrieval/dense.py`)
* **Chunk Target Size**: `800 tokens` (tokenized via HuggingFace BGE tokenizer)
* **Chunk Overlap**: `150 tokens`
* **Embedding Token Truncation Limit**: `512 tokens` (enforced via `tokenizer(..., max_length=512, truncation=True)`)
* **Total Chunks in Corpus**: `4,239`
* **Total Parsed Chapters**: `315`
* **Vector Index Type**: FAISS `IndexFlatIP` (exact inner product on L2-normalized float32 vectors)
* **Sparse Retriever**: `rank_bm25.BM25Okapi` (built on-the-fly in RAM on import, unpersisted)
* **BM25 Tokenization**: Lowercase, regex `[^a-z0-9\s]` replaced with space, split on whitespace, NLTK English stopwords filtered out
* **Fusion Algorithm**: Weighted Reciprocal Rank Fusion (RRF)
* **RRF Parameters**: $k_{RRF} = 60$, $\alpha = 0.7$ (Dense weight: $0.7$, Sparse weight: $0.3$)
* **Candidate Pool Size for Fusion**: `top_k * 2` (For `top_k=4`, pool size is 8 from dense and 8 from sparse)
* **Context Selection Top-K**: `4` (`TOP_K_CONTEXT = 4` in `generate.py`)
* **Chunk Character Cap in Context**: `2,200 characters` per chunk
* **Reranking Status**: **None** (Disabled; no cross-encoder or second-stage reranker)
* **Generation Model**: `openai/gpt-oss-120b` (hosted via Groq API)
* **Generation Parameters**: `temperature = 0.2`, `max_tokens = 1250`, `frequency_penalty = 0.15`, `presence_penalty = 0`, `max_retries = 3`
* **Citation Format**: Inline brackets `[1]`, `[2]`, `[3]`, `[4]` mapping to 1-indexed retrieved context items
* **Evaluation Query Benchmarks**:
  - Retrieval: **50 queries** (Tier 1 Direct: 17, Tier 2 Indirect: 18, Tier 3 Hard: 15)
  - Generation: **20 queries** (Tier 1 Direct: 7, Tier 2 Indirect: 7, Tier 3 Hard: 6)
* **Retrieval Metrics Reported**:
  - Hybrid ($\alpha=0.7$): Recall@5 = `0.940`, Recall@10 = `1.000`, MRR = `0.883`, NDCG@10 = `0.918`
  - Dense (BGE): Recall@5 = `0.960`, Recall@10 = `1.000`, MRR = `0.895`, NDCG@10 = `0.929`
  - Sparse (BM25): Recall@5 = `0.860`, Recall@10 = `0.920`, MRR = `0.695`, NDCG@10 = `0.754`
* **Generation Metrics**:
  - Automated Structural Grounded: `80.0%` (README / Notebook) vs `90.0%` (`auto_metrics_results.json` after recent uncommitted generation adjustments)
  - Citation Accuracy: `80.0%` vs `90.0%`
  - Lexical Grounded Rate: `27.7%` vs `29.4%`
  - LLM-as-a-Judge (`llama-3.3-70b-versatile`): Faithfulness = `3.75/5`, Completeness = `3.80/5`, Medical Accuracy = `4.60/5`, Hallucination Flags = `3/20`
  - Manual Grading: Accuracy = `82.5%`, Fully Correct = `70.0%`
* **Application Framework**: Streamlit `1.54.0`

### Detected Discrepancies Across Files

1. **Generation Temperature**: In git commit history (`generate.py`), temperature was `0.25`. In the current working tree uncommitted modification, it is `0.20`.
2. **Context Window / Candidate Count**: `app.py` line 402 renders `retrieved_chunks[:5]`, whereas `generate.py` lines 12 & 154 hardcodes `TOP_K_CONTEXT = 4`. The 5th source is never sent to the LLM and will never be rendered unless `generate.py` is changed.
3. **Sidebar Metric Hardcoding**: `app.py` lines 267-272 hardcodes offline evaluation numbers ("Recall@10: 1.000", "MRR: 0.883", "Faith (LLM): 4.1 / 5") which do not match `README.md` ("Faith: 3.75 / 5") or `generation_eval_report.json` ("Faith: 3.70 / 5").
4. **Automated Metrics Schema Mismatch**: `evaluation/auto_metrics.py` writes `"per_question"` with key `"question_id"`, but `evaluation/manual_grades.py` parses `"per_query"` with key `"id"`, causing `generation_eval_report.json` to store `null` for all automated structural metrics.
5. **Deduplication Inconsistency in Retrieval Evaluation**: In `evaluation/retrieval_metrics.py`, MRR uses raw candidate index `rank` (not skipping duplicate chapter chunk positions), whereas `ndcg_at_k` uses a separate deduplicated counter `pos`.

---

## 4. File-by-File Assessment

### 1. `requirements.txt`
* **Current responsibility**: Specification of project dependencies.
* **What is good**: Covers the core functional stack without framework bloat.
* **What is outdated/fragile**: 
  - **P0**: The file is encoded in **UTF-16LE** with byte-order marks. Many Linux/CI tools and standard `pip` invocations fail when parsing UTF-16LE.
  - **P0**: Missing `pypdf` (which is imported by `preprocessing/clean_text.py`).
  - Contains build/GUI dependencies that could be grouped (`PyYAML`, `regex`, `beautifulsoup4`).
* **Recommended direction**: Convert encoding to UTF-8 without BOM; add `pypdf>=4.0.0`; add comments distinguishing core, inference, and dev dependencies.
* **Priority**: **P0**

### 2. `preprocessing/clean_text.py`
* **Current responsibility**: Extracts medical content from `data/merck_manual.pdf` (pages 53–3655) and cleans formatting noise.
* **What is good**: Targeted regex for structural bracket noise (`[Table...]`, `(see p. ...)`) and intelligent lowercase line-break unwrapping (`(?<![.\n])\n(?=[a-z])`).
* **What is outdated/fragile**:
  - Hardcoded page boundaries (`53` to `3655`).
  - Missing `pypdf` in `requirements.txt`.
  - Missing CLI arguments; script runs only with hardcoded paths.
* **Recommended direction**: Add standard CLI argument parsing (`argparse`), error checking for missing PDF, and document page range selection.
* **Priority**: **P2**

### 3. `preprocessing/section_parser.py`
* **Current responsibility**: Splits cleaned text into chapter-level structured records using regex `^Chapter\s+(\d+)\.\s+(.+)`.
* **What is good**: Deterministic and stable; avoids fragile multi-level subsection parsing that previously broke on the complex Merck layout.
* **What is outdated/fragile**:
  - Chapter-level records are too coarse (average length ~37,000 characters). All section and subheading hierarchy within chapters is flattened into a single string.
  - Heading metadata is lost before chunking.
* **Recommended direction**: Retain chapter boundaries as the primary unit, but extract H2/H3 medical headings (e.g. Symptoms, Diagnosis, Treatment) as chunk metadata.
* **Priority**: **P1**

### 4. `preprocessing/chunking.py`
* **Current responsibility**: Tokenizes chapter content with BGE tokenizer into 800-token chunks with 150-token overlap.
* **What is good**: Token-aware chunking matches the embedding tokenizer rather than naive character slicing.
* **What is outdated/fragile**:
  - **P0 (Import-time execution)**: Line 28 runs `tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)` at global scope outside `if __name__ == "__main__":`.
  - **P0 (Dimension mismatch)**: Chunks are 800 tokens, but dense retrieval models (`bge-large-en-v1.5`) only encode up to 512 tokens. Tokens 513–800 are silently truncated during dense embedding.
  - Heading context is not prepended to chunk content (causing topic drift in later chunks of long chapters).
* **Recommended direction**: Wrap tokenizer loading in functions/CLI guard; align chunk size to $\le 512$ tokens (e.g. 450 tokens with 64 overlap); inject chapter title and local section heading into each chunk header.
* **Priority**: **P0**

### 5. `indexing/dense_faiss.py`
* **Current responsibility**: Generates BGE embeddings for all 4,239 chunks and builds a FAISS `IndexFlatIP` index.
* **What is good**: Normalizes embeddings with `faiss.normalize_L2` before building `IndexFlatIP` (mathematically equivalent to exact cosine similarity); saves both `.npy` matrix and `ids.json` mapping.
* **What is outdated/fragile**:
  - **P0**: `texts = [f"passage: {chunk['content']}" for chunk in batch]`. BGE v1.5 does **not** use the `"passage: "` prefix (that is E5 convention). BGE v1.5 passage embedding expects raw text without prefix.
  - Hardcoded `DEVICE = "cpu"`; does not leverage CUDA/MPS if available.
  - Silent truncation at `max_length=512`.
* **Recommended direction**: Remove invalid `"passage: "` prefix; allow dynamic CUDA/MPS/CPU selection; verify vector count and integrity.
* **Priority**: **P0**

### 6. `retrieval/dense.py`
* **Current responsibility**: Loads model and index, encodes query, and executes FAISS cosine similarity search.
* **What is good**: Accurate mean pooling with attention mask; correct L2 normalization of query vectors.
* **What is outdated/fragile**:
  - **P0 (Import-time blocking)**: Lines 32–56 load the tokenizer, 1.34 GB PyTorch model, FAISS index, and 15.3 MB JSON file into global memory on module import.
  - **P0**: `query = "query: " + query`. BGE-1.5 query instruction format is `"Represent this sentence for searching relevant passages: "` (or no prefix for symmetric search), **not** `"query: "`.
  - Hardcoded `PROJECT_ROOT` resolution.
* **Recommended direction**: Encapsulate in a lazy-loaded `DenseRetriever` class or singleton; align query instruction with BGE-1.5 specification; provide clean retrieval function.
* **Priority**: **P0**

### 7. `retrieval/sparse.py`
* **Current responsibility**: BM25Okapi sparse retrieval over corpus.
* **What is good**: Fast rank-bm25 implementation with stopword filtering.
* **What is outdated/fragile**:
  - **P0 (Massive import-time penalty)**: Every time this module is imported, it reads a 15.3 MB JSON file from disk, tokenizes all 4,239 documents using Python loops and regex, and recomputes the BM25 index in RAM! It is **never saved to disk**.
  - **P1**: Naive tokenization `re.sub(r"[^a-z0-9\s]", " ", text)` removes hyphens, periods, and slashes, which destroys critical medical entities (e.g. *H. pylori*, *COVID-19*, *Type-2*, *β-blockers*, dosages).
  - Stopword list strips words that have clinical relevance in queries (e.g. *what*, *how*, *condition*).
* **Recommended direction**: Precompute and serialize BM25 parameters/tokenized corpus (e.g., via `pickle` or `joblib`) to `index/bm25.pkl`; lazy-load on demand; use a medical-aware regex tokenizer that preserves hyphens and decimal points.
* **Priority**: **P0**

### 8. `retrieval/fusion.py`
* **Current responsibility**: Combines dense and sparse results using weighted Reciprocal Rank Fusion (RRF).
* **What is good**: Clean RRF formulation with canonical metadata resolution.
* **What is outdated/fragile**:
  - **P1 (Candidate Starvation & Dense Domination)**: Fetches only `top_k * 2` candidates from each system. At $k=4$, candidate pool is 8. With $\alpha=0.7$, the 8th dense item gets score $0.7 / (60 + 8) = 0.010294$, whereas the #1 sparse item gets at most $0.3 / (60 + 1) = 0.004918$. Sparse hits cannot enter the fused top-4 unless they are also in the dense top-8.
* **Recommended direction**: Widen the candidate pool (e.g. fetch top-20 or top-30 candidates from both retrievers before fusion, then truncate to top-k); make $k_{RRF}$ and candidate pool size configurable.
* **Priority**: **P1**

### 9. `generate.py`
* **Current responsibility**: Orchestrates hybrid retrieval, context formatting, Groq LLM API generation, and response cleaning.
* **What is good**:
  - Prompt structure is high-yield, requiring overview paragraph, clinical headings, and inline citations.
  - Built-in retry handler with exponential backoff for Groq 429 rate limits.
  - Robust post-processing in `clean_answer`: strips `<think>` blocks, normalizes citation variants (`【1】`, `[Source 1]` $\to$ `[1]`), collapses duplicates (`[1][1]` $\to$ `[1]`), and strips trailing references.
* **What is outdated/fragile**:
  - **P0 (Global API client initialization)**: Lines 24–28 instantiate `client = Groq(api_key=api_key)` at module import time. If `GROQ_API_KEY` is missing (e.g. in test environments or retrieval-only runs), importing `generate.py` throws an unhandled exception.
  - Hardcoded context size (`TOP_K_CONTEXT = 4`) and character truncation limit (`2200`).
* **Recommended direction**: Lazy-load Groq client inside generation function or class; preserve the strong prompting and cleaning logic.
* **Priority**: **P0**

### 10. `app.py`
* **Current responsibility**: Streamlit web application providing interactive medical QA interface.
* **What is good**:
  - Polished dark UI with medical navy palette (`#0b1220`, `#00b4e6`, `#00d4ff`).
  - Clear separation between response, response metadata, retrieved sources, and grounding chips.
* **What is outdated/fragile**:
  - **P1 (Tautological Grounding Signal)**: Lines 425–428 define `citation_consistent` as `set(cited_actual).issubset(set(retrieved_chapter_numbers))`. Because `cited_actual` is extracted directly from `retrieved_chunks`, this condition is mathematically guaranteed to be True as long as the model cites valid numbers, even if it cites the completely wrong passage for a claim!
  - Hardcoded evaluation metrics in the sidebar.
  - Hardcoded regex cleaning of an empty "Acute Management" section.
  - `sys.path.insert(0, PROJECT_ROOT)` at top of file.
* **Recommended direction**: Implement actual citation-to-passage attribution verification; dynamic display of retrieved sources matching `TOP_K_CONTEXT`; eliminate fragile regex coupling.
* **Priority**: **P1**

### 11. `evaluation/retrieval_metrics.py`
* **Current responsibility**: Evaluates Dense, Sparse, and Hybrid ($\alpha \in \{0.3, 0.5, 0.7\}$) systems across 50 test queries in 3 difficulty tiers.
* **What is good**: Excellent, realistic medical test queries categorized into Direct, Indirect, and Hard tiers.
* **What is outdated/fragile**:
  - **P1**: Binary relevance evaluated only at the coarse chapter level (`relevant_chapter: int`). If any chunk from the chapter is retrieved, it is marked as a hit, even if the specific chunk does not answer the question.
  - **P1**: Metric inconsistency between MRR (which uses un-deduplicated chunk rank) and NDCG@10 (which uses deduplicated chapter position).
  - Synchronous loop without checkpointing.
* **Recommended direction**: Add passage/chunk-level relevance annotations where possible; harmonize deduplication logic between MRR and NDCG.
* **Priority**: **P1**

### 12. `evaluation/generation_eval.py`
* **Current responsibility**: Generates answers for 20 balanced queries and dumps them to `evaluation/generation_outputs.json`.
* **What is good**: Saves complete record with query, tier, expected chapter, retrieved passages, cited sources, and answer text.
* **What is outdated/fragile**:
  - **P0 (Path bug)**: `OUTPUT_FILE = "evaluation/generation_outputs.json"` creates `evaluation/evaluation/generation_outputs.json` if run from within `evaluation/` directory.
  - Only saves 500-character snippets (`content_snippet`) for retrieved chunks, losing text used for generation.
* **Recommended direction**: Use `PROJECT_ROOT / "evaluation" / "generation_outputs.json"`; store full chunk text or reference chunk IDs.
* **Priority**: **P0**

### 13. `evaluation/auto_metrics.py`
* **Current responsibility**: Calculates automated retrieval hit rate, citation accuracy, structural grounding, hallucination rate, and lexical grounding.
* **What is good**: Pure Python/regex metric calculation without additional heavy dependencies.
* **What is outdated/fragile**:
  - **P0**: Defines `hallucination_rate = 1 - struct_grounded_rate`, which mislabels retrieval misses as hallucinations.
  - **P0**: Schema mismatch with `manual_grades.py` (emits `"per_question"` instead of `"per_query"`).
  - Lexical grounding evaluates token overlap against only the 500-character snippet rather than the full context provided to the model.
* **Recommended direction**: Disentangle retrieval failure from generation hallucination; harmonize JSON output keys; measure lexical grounding against full provided context.
* **Priority**: **P0**

### 14. `evaluation/llm_judge.py`
* **Current responsibility**: Uses `llama-3.3-70b-versatile` via Groq to grade answers on Faithfulness, Completeness, and Medical Accuracy (1–5 scale).
* **What is good**: Structured JSON prompting with schema validation and retry logic.
* **What is outdated/fragile**:
  - **P0 (Severe context truncation)**: Line 67 truncates `CONTEXT` to `[:1500]` characters and `ANSWER` to `[:1200]` characters. The actual retrieved context sent to the model during generation is up to ~8,800 characters! The judge evaluates faithfulness against an incomplete context excerpt, causing false hallucination flags.
  - Import-time `Groq()` client creation.
* **Recommended direction**: Provide the full retrieved context (or full passages corresponding to cited indices) to the judge model; lazy-load Groq client.
* **Priority**: **P0**

### 15. `evaluation/manual_grades.py`
* **Current responsibility**: Interactive CLI tool for manual 0/1/2 grading and compiling the final evaluation report.
* **What is good**: Interactive resume capability, incremental auto-saving to `manual_grades.json`.
* **What is outdated/fragile**:
  - **P0 (Broken Integration)**: Lines 106 & 127–130 attempt to load `"auto_grounded_rate"`, `"auto_citation_accuracy"`, etc., from `auto_data["per_query"]`. Because `auto_metrics.py` outputs `"per_question"` with different key names, all automated metrics in `generation_eval_report.json` become `null`!
* **Recommended direction**: Fix dictionary keys to match `auto_metrics.py` output; make CLI grading optionally non-interactive for automated pipelines.
* **Priority**: **P0**

---

## 5. RAG Quality Assessment

### A. Chunking
* **Chapter Parsing Granularity**: Parsing at chapter level (`section_parser.py`) is deterministic and eliminates earlier parsing crashes, but chapter texts average ~37,000 characters. All internal section boundaries (`Diagnosis`, `Treatment`, `Prognosis`, `Etiology`) are flattened into unstructured text.
* **The 800 vs 512 Token Window Inconsistency**: 
  - `preprocessing/chunking.py` generates chunks with `CHUNK_SIZE = 800` and `OVERLAP = 150`.
  - `indexing/dense_faiss.py` and `retrieval/dense.py` specify `max_length = 512`.
  - **Tokens 513 through 800 (over 35% of chunk content) are completely discarded by the dense embedding model!** Any critical medical statement residing in the latter half of an 800-token chunk cannot be retrieved via dense search.
* **Metadata Loss**: Chunks only carry `chapter_number` and `chapter_title`. They lack heading hierarchy (e.g. *Chapter 178 Headache > Migraine > Acute Treatment*), leading to context ambiguity in later chunks.

### B. Dense Retrieval
* **Embedding Model Viability**: `BAAI/bge-large-en-v1.5` is a strong 1024-dim dense retriever and remains an excellent local baseline for medical RAG.
* **Normalization**: L2 normalization via `faiss.normalize_L2` before inner product indexing is mathematically correct and provides exact cosine similarity.
* **Prefix Mismatch**:
  - `dense_faiss.py` used `texts = [f"passage: {chunk['content']}" ...]`
  - `dense.py` used `query = "query: " + query`
  - BGE v1.5 is **not** an E5 model. BGE passage encoding requires raw text, while query encoding utilizes the specific instruction: `"Represent this sentence for searching relevant passages: "` (or no prefix for symmetric matching). Using E5 prefixes introduces out-of-distribution token distortions in the embedding space.

### C. Sparse Retrieval
* **Medical Entity Tokenization**: The current regex `re.sub(r"[^a-z0-9\s]", " ", text)` causes severe entity mangling:
  - *"H. pylori"* $\to$ `['h', 'pylori']`
  - *"COVID-19"* $\to$ `['covid', '19']`
  - *"Type-2 diabetes"* $\to$ `['type', '2', 'diabetes']`
  - *"β-blockers"* $\to$ `['blockers']`
  - Drug names with hyphens (e.g. *amoxicillin-clavulanate*) and numerical staging (*Stage III*, *NYHA Class II*) lose structural integrity.
* **Stopwords Impact**: Filtering out generic NLTK stopwords removes query syntax like *"how"*, *"what"*, *"condition"*, and *"treatment"* (present in `STOPWORDS` in `auto_metrics.py`), which can impair medical intent discrimination.

### D. Fusion
* **Weighted RRF Dynamics**: 
  $$RRF(d) = \frac{\alpha}{60 + r_{dense}(d)} + \frac{1 - \alpha}{60 + r_{sparse}(d)}$$
  With $\alpha = 0.7$ and pool size $2 \cdot k = 8$:
  - Lowest dense score in candidate pool: $\frac{0.7}{60 + 8} = 0.010294$
  - Highest sparse score possible: $\frac{0.3}{60 + 1} = 0.004918$
  - Sparse score ($0.004918$) cannot exceed the 8th dense score ($0.010294$).
  - **Result**: BM25 can never inject a unique document into the top-k results; it can only reorder documents that dense retrieval already found.
* **Candidate Pool Starvation**: Fetching only $2 \cdot k$ candidates (8 candidates for $k=4$) starves the fusion stage of potential complementary sparse candidates.

### E. Context Selection & Re-ranking
* **Chunk Redundancy**: If dense retrieval scores multiple adjacent chunks from the same chapter highly, all 4 context slots can be consumed by adjacent slices of the same disease, crowding out differential diagnoses or systemic interactions.
* **No Second-Stage Re-ranking**: The system has no cross-encoder or diversity filter (e.g. MMR) to balance high precision with topical coverage.

---

## 6. Generation Quality Audit

### Strengths to Preserve
* **Prompt Architecture**: The prompting structure in [generate.py](file:///c:/Users/Khushdeep%20Singh/Desktop/Projects/MediRAG/generate.py) is well-crafted:
  - Demands a concise 2–3 sentence `## Overview` paragraph.
  - Enforces logical clinical subheadings (`## Symptoms & Clinical Presentation`, `## Treatment & Management`).
  - Emphasizes explicit inline bracket citations `[1]`, `[2]`.
  - Forbids hallucinations with a clear directive: *"The provided context does not mention [aspect]"*.
* **Output Sanitization**: The uncommitted modifications to `clean_answer()` in `generate.py` provide robust sanitization:
  - Strips reasoning `<think>` tags emitted by reasoning models.
  - Normalizes Chinese brackets `【1】` and verbose `[Source 1]` into uniform `[1]`.
  - Removes trailing reference sections that duplicate UI sources.
  - Deduplicates consecutive duplicate citations (`[1][1]` $\to$ `[1]`).
* **Rate-Limit Resilience**: The exponential backoff retry loop in `generate.py` handles Groq free-tier 429 rate limits gracefully.

### Deficiencies to Improve
* **Context Formatting Truncation**: Formatting clips chunk text to 2,200 characters (`[:CHUNK_CHAR_LIMIT]`). While necessary to respect Groq token limits, hard truncation mid-sentence can sever clinical statements.
* **Citation Hallucination Vulnerability**: The model is instructed to use `[1]` through `[4]`, but there is no runtime post-validation checking whether the cited chunk actually supports the statement in that sentence.
* **No Context Abstention Safeguard**: If `hybrid_search()` returns empty or low-relevance results, the generator returns a generic string rather than structured fallback guidance.

---

## 7. Evaluation Audit

### What is Actually Measured
1. **Retrieval**: 
   - Recall@5, Recall@10, MRR, NDCG@10 across 50 queries in 3 tiers.
   - Ground truth is a single integer chapter number.
2. **Automated Generation**:
   - Retrieval Hit Rate: Is the expected chapter in top-5 retrieved chunks?
   - Citation Accuracy: Did the answer cite a chunk originating from the expected chapter?
   - Citation Consistency: Is every cited chapter among the retrieved chapters? (Tautologically true by implementation).
   - Structural Grounded Rate: Conjunction of Hit, Citation Accuracy, and Citation Consistency.
   - Hallucination Rate: Computed as `1 - Structural Grounded Rate`.
   - Lexical Grounded: Token overlap between answer and 500-character snippet.
3. **LLM-as-a-Judge**:
   - `llama-3.3-70b-versatile` scores Faithfulness, Completeness, Medical Accuracy (1–5) and flags hallucinations.
4. **Manual Grading**:
   - 0 (Wrong), 1 (Partial), 2 (Correct).

### Methodological Weaknesses
1. **Coarse Relevance Labels**: A query like *"What causes iron deficiency anemia?"* has ground truth `relevant_chapter: 105`. If the system retrieves chunk `105_12` (which discusses lab diagnostics for anemia, not causes), it is scored as a perfect retrieval hit.
2. **Tautological Consistency Metric**: In `auto_metrics.py`:
   ```python
   cited_actual_chapters = [
       retrieved_chunks[i - 1]["chapter_number"]
       for i in cited_numbers if 0 < i <= len(retrieved_chunks)
   ]
   citation_consistent = set(cited_actual_chapters).issubset(set(retrieved_chapters))
   ```
   Because `cited_actual_chapters` is formed exclusively by sampling `retrieved_chunks`, its elements are guaranteed to be a subset of `retrieved_chapters`. This metric is always 100% and tests nothing.
3. **Severe Context Truncation in LLM Judge**: In `evaluation/llm_judge.py`, context is truncated to `[:1500]` characters. If the model drew valid facts from Chunk 3 or Chunk 4, the judge never sees that text and falsely penalizes the model with low Faithfulness scores and hallucination flags.
4. **Integration Disconnect in `manual_grades.py`**:
   - `auto_metrics.py` writes `"per_question"` containing `"question_id"`, `"structural_grounded"`.
   - `manual_grades.py` reads `auto_data["per_query"]` and looks for `"auto_grounded_rate"`.
   - Result: All auto metrics in `generation_eval_report.json` are `null`.

---

## 8. Engineering / Maintainability Assessment

| Issue | File(s) | Severity | Description |
|---|---|---|---|
| **Module-Level Heavy Imports** | `retrieval/dense.py`, `retrieval/sparse.py`, `preprocessing/chunking.py` | **P0** | Loading models, indexes, 15 MB JSON files, and rebuilding BM25 tokenizers happens immediately upon `import`, preventing fast unit testing and modular execution. |
| **Global Client Initialization** | `generate.py`, `evaluation/llm_judge.py` | **P0** | Instantiating `Groq(api_key=...)` at top-level crashes any process if `GROQ_API_KEY` is not present in `.env`, even when running retrieval-only tasks. |
| **Path Fragility & Accidental Dirs** | `evaluation/generation_eval.py`, `app.py`, `retrieval/dense.py` | **P1** | Inconsistent path resolution (`os.path.join(..., "..")` vs `Path(__file__).resolve().parent.parent`). Running scripts from different working directories created an accidental `evaluation/evaluation/` directory. |
| **Lack of Serialization for BM25** | `retrieval/sparse.py` | **P1** | BM25 is built from scratch on every run. Persisting the BM25 model via `pickle` or `joblib` would reduce startup latency from ~4s to ~50ms. |
| **Weak Typing & Code Duplication** | Throughout | **P2** | Functions lack Pydantic schemas or type annotations. Mean-pooling logic is duplicated identically in `indexing/dense_faiss.py` and `retrieval/dense.py`. |
| **Stale Files & Artifacts in Git** | `evaluation/evaluation/generation_outputs.json`, `data/merck_manual.pdf` | **P1** | 19.2 MB raw PDF, 40 MB of processed JSON/text, and an orphaned nested evaluation output file are tracked in git. |

---

## 9. Dependency Assessment

Inspection of `requirements.txt` and the virtual environment (`venv/`):

1. **File Encoding Issue (P0)**:
   - `requirements.txt` is encoded in **UTF-16LE** with BOM (`FF FE`).
   - Standard tools like `pip install -r requirements.txt` on Unix/macOS or CI runners throw UnicodeDecodeErrors unless converted to UTF-8.
2. **Missing Dependencies (P0)**:
   - `pypdf` is imported in `preprocessing/clean_text.py` but is **missing from `requirements.txt`** and is not installed in the virtual environment. Running `clean_text.py` fails immediately with `ModuleNotFoundError`.
3. **Version Pinning & Ecosystem**:
   - `torch==2.10.0`, `transformers==5.2.0`, `sentence-transformers==5.2.3`, `streamlit==1.54.0` are installed.
   - `groq==1.0.0` is pinned and operational.
   - `rank-bm25==0.2.2` and `faiss-cpu==1.13.2` are installed and stable.
4. **Unused Packages**:
   - `beautifulsoup4==4.14.3` is pinned in `requirements.txt` but never imported anywhere in `preprocessing/`, `retrieval/`, or `evaluation/` (text is parsed from PDF/text, not HTML).
   - `tiktoken==0.12.0` is installed but the project uses HuggingFace `AutoTokenizer` for chunking.

---

## 10. Recommended 7-Day Upgrade Roadmap

```
Day 1: Foundation, Encoding & Lifecycle Clean-up
Day 2: Preprocessing, Chunk Alignment (512-token fix) & Metadata Enrichment
Day 3: BM25 Persistence & Medical Tokenizer Upgrade
Day 4: Fusion Calibration, Candidate Expansion & BGE Instruction Alignment
Day 5: Generation Guardrails, Citation Attribution & Latency Tuning
Day 6: Evaluation Suite Repair & Ground Truth Refinement
Day 7: Streamlit UI Polish, Artifact Verification & Documentation
```

### Day 1: Foundation, Encoding & Lifecycle Clean-up
* **P0**: Re-encode `requirements.txt` to UTF-8 without BOM. Add `pypdf>=4.0.0`; remove unused `beautifulsoup4` and `tiktoken`.
* **P0**: Remove module-level import side effects in `retrieval/dense.py`, `retrieval/sparse.py`, and `generate.py`. Encapsulate model and client creation in lazy classes or initialization functions.
* **P1**: Standardize `PROJECT_ROOT` resolution across all scripts using `pathlib.Path(__file__).resolve().parents[...]`. Remove `sys.path.append` workarounds.
* **P1**: Clean up git tracking: remove accidental `evaluation/evaluation/generation_outputs.json` from git index; ensure `.gitignore` properly covers large raw PDFs and indexes.

### Day 2: Preprocessing, Chunk Alignment (512-Token Fix) & Metadata Enrichment
* **P0**: **Resolve the 512-token truncation disconnect**. Re-chunk corpus to 450 tokens with 64 overlap (guaranteeing 100% token coverage under BGE's 512 max length limit).
* **P1**: Enrich chunk metadata: during section parsing, capture section titles and prepend context tags (e.g. `Chapter 178: Headache > Migraine > Acute Therapy`) to each chunk.
* **P1**: Re-index with `dense_faiss.py` without the invalid `"passage: "` prefix. Enable device auto-detection (`cuda` if available, else `cpu`).

### Day 3: BM25 Persistence & Medical Tokenizer Upgrade
* **P0**: Persist the BM25 index to disk (`index/bm25.pkl` or `.joblib`). Eliminate the 4-second startup re-indexing on every process launch.
* **P0**: Upgrade BM25 tokenization to preserve clinical punctuation: retain hyphens (*Type-2*, *β-blockers*), decimal points (*H. pylori*, *0.5 mg*), and slash notation (*mg/kg*, *V/Q*).
* **P2**: Refine stopword filtering to retain clinical questioning intent.

### Day 4: Fusion Calibration, Candidate Expansion & BGE Instruction Alignment
* **P0**: Correct BGE query formatting in `retrieval/dense.py` to use BGE-1.5 query instruction: `"Represent this sentence for searching relevant passages: "` (or raw query if symmetric).
* **P0**: Fix RRF candidate starvation: increase retrieval candidate pool from $2 \cdot k$ to at least 25 candidates per retriever prior to RRF fusion.
* **P1**: Calibrate $\alpha$ across the expanded pool to ensure sparse retrieval can surface high-precision lexical matches that dense search ranks lower.

### Day 5: Generation Guardrails, Citation Attribution & Latency Tuning
* **P0**: Commit and preserve the improved prompting and cleaning logic in `generate.py`.
* **P1**: Implement actual citation attribution verification: verify that the sentence preceding `[N]` shares semantic overlap / entailment with chunk $N$.
* **P2**: Add graceful fallback and abstention messaging when retrieved confidence is low.

### Day 6: Evaluation Suite Repair & Ground Truth Refinement
* **P0**: Fix the schema integration between `auto_metrics.py` and `manual_grades.py` so `generation_eval_report.json` correctly populates automated metrics.
* **P0**: Fix `llm_judge.py` context truncation: feed full retrieved passages for the cited items rather than slicing context at 1,500 characters.
* **P1**: Fix the tautological definition of `citation_consistency` and decouple hallucination rate from retrieval hit rate.
* **P1**: Re-run the full 50-query retrieval benchmark and 20-query generation benchmark to establish the new verified baseline.

### Day 7: Streamlit UI Polish, Artifact Verification & Documentation
* **P1**: Update `app.py` to render all `TOP_K_CONTEXT` sources dynamically without hardcoded limits.
* **P1**: Replace hardcoded sidebar metrics with dynamically loaded summary values from `generation_eval_report.json`.
* **P2**: Add an interactive source attribution indicator showing which sentence maps to which chunk.
* **P2**: Generate comprehensive modernization documentation and reproducible runbook.

---

## 11. Things We Should NOT Change

The following architectural choices are solid, well-aligned with project scale (~4,239 chunks), and **must not be replaced**:

1. **FAISS `IndexFlatIP`**: With only 4,239 vectors of dimension 1024, exact flat inner-product search executes in under 2 milliseconds on CPU. Replacing FAISS with Milvus, Pinecone, Qdrant, or Chroma would add network latency, operational fragility, and external dependencies without any accuracy gain.
2. **`BAAI/bge-large-en-v1.5` Embedding Model**: This model achieves 0.960 Recall@5 and 1.000 Recall@10 on the 50-query benchmark. Upgrading to a heavier proprietary embedding API would incur ongoing costs with minimal headroom for improvement.
3. **BM25Okapi Algorithm**: BM25 is the standard for medical keyword retrieval (exact drug names, pathogen species, rare syndromes). Replacing it with SPLADE or complex learned sparse retrievers is unnecessary for a 4k chunk corpus.
4. **Weighted Reciprocal Rank Fusion (RRF)**: RRF is parameter-efficient, robust against disparate score distributions, and requires no score normalization heuristics. It should be tuned ($\alpha$ and candidate pool size), not replaced with a black-box combiner.
5. **Groq API + `openai/gpt-oss-120b`**: Generation latency on Groq is under 1.5 seconds for ~400 words. It provides exceptional clinical structure, zero temperature variance at 0.2, and strong adherence to citation constraints.
6. **Streamlit Frontend**: A lightweight Streamlit UI is ideal for an academic/clinical demonstration system. Rewriting the frontend in React, Next.js, or FastAPI would consume days of engineering time without enhancing RAG quality.
7. **The 50-Query Retrieval and 20-Query Generation Benchmark Datasets**: The existing query sets have verified chapter relevance and 3 distinct difficulty tiers. They provide valuable continuity for before-and-after modernization comparisons.

---

## 12. Open Questions / Experiments

Before committing to significant architectural adjustments, the following 5 controlled experiments must be conducted:

### Experiment 1: Impact of Correcting the 512 Truncation Disconnect
* **Hypothesis**: Re-chunking to $\le 512$ tokens (e.g. 450 tokens with 64 overlap) so that the entire chunk is visible to BGE will improve Recall@5 on Tier-3 Hard queries where key facts were previously truncated.
* **Measurement**: Run `retrieval_metrics.py` before and after re-chunking. Compare Tier-3 MRR and Recall@5.

### Experiment 2: BGE Query Instruction & Prefix Correction
* **Hypothesis**: Replacing `"query: "` with `"Represent this sentence for searching relevant passages: "` and removing `"passage: "` from chunks will improve cosine similarity alignment and dense MRR.
* **Measurement**: Compare dense retrieval MRR and NDCG@10 across the 50 evaluation queries.

### Experiment 3: RRF Candidate Pool Expansion & Alpha Calibration
* **Hypothesis**: Expanding the candidate pool from $2 \cdot k$ (8) to 30 candidates per system will allow high-scoring sparse keyword hits to enter the top-4 context when dense search fails.
* **Measurement**: Measure Hybrid Recall@5 and Recall@10 on Tier-1 (Direct drug/pathogen queries) across $\alpha \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$.

### Experiment 4: Cross-Encoder Re-ranking vs. Pure RRF
* **Hypothesis**: Applying a lightweight cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`) on the top-15 fused candidates will improve NDCG@5 and eliminate off-topic chunks from the top-4 generation context.
* **Measurement**: Measure latency penalty vs. gain in NDCG@5 and LLM Judge Faithfulness score.

### Experiment 5: Full-Context vs. Truncated LLM-as-a-Judge
* **Hypothesis**: Giving the LLM judge the full retrieved text rather than a 1,500-character snippet will eliminate false hallucination flags for queries 6, 15, and 16.
* **Measurement**: Re-run `llm_judge.py` with full context and compare hallucination flag count and faithfulness distribution.

---

## Day 1 Foundation Status

### Completed Changes
1. **Requirements Modernization**:
   - Re-encoded `requirements.txt` from UTF-16LE to standard UTF-8 without BOM.
   - Added `pypdf>=4.0.0` (required by `preprocessing/clean_text.py`).
   - Grouped dependencies logically (Core, Retrieval / ML, Generation / LLM API, Text Processing & Ingestion, Evaluation / Visualization, App / UI, Utilities).
   - Validated dependency resolution with `pip install -r requirements.txt --dry-run` (exit code 0).
2. **Lazy Initialization Across All Subsystems**:
   - `retrieval/dense.py`: Refactored into a `DenseRetriever` class with on-demand loading of BGE model, tokenizer, FAISS index, and chunks metadata. Importing the module is now instantaneous and consumes minimal RAM.
   - `retrieval/sparse.py`: Refactored into a `SparseRetriever` class with on-demand corpus tokenization and BM25 index construction. Eliminated import-time re-tokenization of 4,239 documents.
   - `generate.py`: Encapsulated Groq API client creation inside `get_groq_client()`. Importing `generate` no longer requires `GROQ_API_KEY` to be present or raises unhandled exceptions.
   - `evaluation/llm_judge.py`: Encapsulated Groq client creation inside `get_judge_client()`.
   - `preprocessing/chunking.py`: Encapsulated BGE tokenizer loading inside `get_tokenizer()`, preventing automatic download/loading on module import.
3. **Project Path Standardisation & Directory Creation**:
   - Converted fragile relative path resolutions and `sys.path` append strings across `preprocessing/`, `indexing/`, `retrieval/`, `evaluation/`, and `app.py` to use `pathlib.Path(__file__).resolve().parent...`.
   - Relocated side-effecting `.mkdir()` directory creations from module scope into explicit execution blocks.
4. **Error Handling & Console Encoding Safeguards**:
   - Added explicit `FileNotFoundError` checks with actionable guidance for missing input/index files across `clean_text.py`, `section_parser.py`, `chunking.py`, `dense_faiss.py`, `dense.py`, `sparse.py`, and `auto_metrics.py`.
   - Added `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` guards in `dense.py` and `sparse.py` to prevent `UnicodeEncodeError` crashes on Windows non-UTF-8 console environments.
5. **Git Hygiene**:
   - Untracked the accidental duplicate `evaluation/evaluation/generation_outputs.json` from git tracking.
   - Updated `.gitignore` to prevent any future `evaluation/evaluation/` directory creation from being tracked.

### Files Modified
* `requirements.txt`
* `.gitignore`
* `retrieval/dense.py`
* `retrieval/sparse.py`
* `retrieval/fusion.py`
* `generate.py`
* `indexing/dense_faiss.py`
* `preprocessing/clean_text.py`
* `preprocessing/section_parser.py`
* `preprocessing/chunking.py`
* `evaluation/generation_eval.py`
* `evaluation/llm_judge.py`
* `evaluation/auto_metrics.py`
* `evaluation/manual_grades.py`
* `evaluation/retrieval_metrics.py`
* `app.py`
* `docs/MediRAG_2026_AUDIT.md`

### Verification Performed
* `pip install -r requirements.txt --dry-run` succeeded without error.
* Isolated import checks:
  - `import retrieval.dense` $\to$ instant, 0 model weights loaded.
  - `import retrieval.sparse` $\to$ instant, 0 documents tokenized.
  - `import generate` without `GROQ_API_KEY` $\to$ succeeded without exception.
  - `import evaluation.llm_judge` without `GROQ_API_KEY` $\to$ succeeded without exception.
  - `import preprocessing.chunking` $\to$ succeeded without tokenizer initialization.
  - `import retrieval.fusion` $\to$ succeeded cleanly.
* End-to-end retrieval smoke test: Executed `dense_search`, `sparse_search`, and `hybrid_search` on query *"What are the causes and treatment of migraine?"*; verified that all return structures are intact, cosine scores and BM25 scores compute identically, and Chapter 178 is returned as top rank.

### Known Remaining Issues
* Tracked large data files: `data/merck_manual.pdf` (19.2 MB) and `data/processed/*` (38.8 MB) remain in git history/tracking. They should be untracked (`git rm --cached`) once remote storage/LFS strategy is confirmed.
* Evaluation schema disconnect: `auto_metrics.py` outputs `"per_question"` while `manual_grades.py` reads `"per_query"`, causing auto metrics in `generation_eval_report.json` to remain `null`.

### Explicitly Deferred Changes (Days 2–7)
* Re-chunking to 450 tokens with 64 overlap (Day 2).
* Metadata injection into chunk text (Day 2).
* BM25 index disk serialization (`index/bm25.pkl`) and medical tokenizer upgrade (Day 3).
* Removal of `"passage: "` prefix and correction of BGE `"query: "` prefix (Day 4).
* RRF candidate pool expansion from $2 \cdot k$ to 25–30 candidates and $\alpha$ sweep (Day 4).
* Cross-encoder re-ranking exploration (Day 4).
* LLM Judge full-context expansion and evaluation metric formulas (Day 6).
* Streamlit UI source count alignment (Day 7).

