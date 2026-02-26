# MediRAG 🏥

MediRAG is a modular **Hybrid Retrieval-Augmented Generation (RAG)** system built on top of the Merck Manual medical corpus. It implements a research-oriented retrieval pipeline combining dense and sparse retrieval strategies with hybrid fusion.

---

## 🚀 Architecture
```
User Query
    → Dense Retrieval (FAISS IndexFlatIP)
    → Sparse Retrieval (BM25)
    → Weighted RRF Fusion
    → Top-k Context Selection
```

---

## 📦 Project Structure
```
MediRAG/
│
├── preprocessing/
│   ├── clean_text.py
│   ├── section_parser.py
│   └── chunking.py
│
├── indexing/
│   └── dense_faiss.py
│
├── retrieval/
│   ├── dense.py
│   ├── sparse.py
│   └── fusion.py
│
├── evaluation/
├── embeddings/
├── index/
└── data/
```

---

## 🔍 Features

- **Exact FAISS retrieval** — IndexFlatIP with L2-normalized cosine similarity
- **Custom token-aware chunking** — 800 tokens with 150-token overlap
- **BM25 sparse retrieval** — keyword-based complementary search
- **Weighted RRF hybrid fusion** — Dense weighted higher than Sparse
- **Deterministic indexing** — clean FAISS ID mapping
- **CPU-compatible** — no GPU required

---

## 🧠 Model

[BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) — 1024-dimensional embeddings

---

## 📊 Current Status

| Component              | Status        |
|------------------------|---------------|
| Dense Retrieval        | ✅ Complete   |
| Sparse Retrieval       | ✅ Complete   |
| Hybrid Weighted Fusion | ✅ Complete   |
| Evaluation Framework   | 🔄 In Progress |

---

## 🏗️ Setup
```bash
pip install -r requirements.txt
```

### ▶ Run Hybrid Retrieval
```bash
python retrieval/fusion.py
```