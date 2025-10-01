# Legal AI RAG Pipeline Evaluation - Workflow

## 📋 Overview

This project evaluates different retrieval approaches for a RAG pipeline using **pre-processed** Sri Lankan legal acts.

**Key Point:** The dataset (`data/acts_with_metadata.tsv`) is already prepared with metadata fields. All data loading and preparation happens automatically in the source code.

## 🎯 What Gets Done Automatically

The following tasks are handled by the source code modules (in `src/`):

1. **Data Loading** - `src/data_loader.py` loads the TSV file
2. **Metadata Joining** - Combines `short_title`, `keywords`, `section_title`, `summary` into one field
3. **Query Design** - `src/queries.py` contains 10 pre-designed legal queries
4. **Retrieval Logic** - BM25, FAISS, and reranking implementations

## 📓 Notebook Workflow (4 Notebooks)

### **01_bm25_retrieval.ipynb** - Start Here

- **Purpose:** Evaluate BM25 sparse retrieval
- **What it does:**
  - Loads data using `load_data()` and `prepare_data()`
  - Retrieves documents for 10 queries on **content field**
  - Retrieves documents for 10 queries on **metadata field**
  - Compares performance and saves results

### **02_faiss_index_builder.ipynb** - GPU Required

- **Purpose:** Build FAISS indices for dense retrieval
- **What it does:**
  - Builds indices for **3 embedding models:**
    1. Legal-BERT (`nlpaueb/legal-bert-base-uncased`)
    2. GTE-Large (`thenlper/gte-large`)
    3. BGE-Large (`BAAI/bge-large-en-v1.5`)
  - Creates **6 indices total:** 3 models × 2 fields (content + metadata)
  - Saves indices to `indices/` directory for CPU use
- **Time:** 30-60 minutes depending on GPU
- **Note:** Only needs to run once

### **03_faiss_retrieval.ipynb** - CPU OK

- **Purpose:** Evaluate FAISS retrieval using pre-built indices
- **What it does:**
  - Loads all 6 FAISS indices from disk
  - Retrieves documents for 10 queries with each model/field combination
  - Compares all 3 models on both content and metadata
  - Analyzes overlap and diversity
  - Saves results

### **04_reranker.ipynb** - Final Step

- **Purpose:** Apply cross-encoder reranking to improve results
- **What it does:**
  - Loads previous results (BM25 and FAISS)
  - Applies reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Compares before/after rankings
  - Measures reranking impact
  - Creates final comparison tables

## 🚀 Quick Start

```powershell
# 1. Setup environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook

# 3. Run notebooks in order:
# - 01_bm25_retrieval.ipynb (15 min)
# - 02_faiss_index_builder.ipynb (30-60 min, GPU)
# - 03_faiss_retrieval.ipynb (20 min, CPU OK)
# - 04_reranker.ipynb (15 min)
```

## 📊 Evaluation Matrix

| Notebook | Retriever | Fields           | Models | Total Configs |
| -------- | --------- | ---------------- | ------ | ------------- |
| 01       | BM25      | 2                | -      | 2             |
| 02       | FAISS     | 2                | 3      | 6 (builds)    |
| 03       | FAISS     | 2                | 3      | 6 (evaluates) |
| 04       | Reranker  | Applied to above | 1      | Multiple      |

**Grand Total:** 12+ retrieval configurations across 10 legal queries

## 🔧 Key Source Modules

- **`src/config.py`** - All configuration and paths
- **`src/data_loader.py`** - Data loading and metadata joining
- **`src/queries.py`** - 10 pre-designed legal queries
- **`src/bm25_retriever.py`** - BM25 implementation
- **`src/faiss_retriever.py`** - FAISS with 3 models
- **`src/reranker.py`** - Cross-encoder reranking
- **`src/evaluation.py`** - Metrics and comparison utilities

## 📂 Output Structure

```
results/
├── bm25_content_results.json
├── bm25_metadata_results.json
├── faiss_legal-bert_content_results.json
├── faiss_legal-bert_metadata_results.json
├── faiss_gte-large_content_results.json
├── faiss_gte-large_metadata_results.json
├── faiss_bge-large_content_results.json
├── faiss_bge-large_metadata_results.json
├── reranked_bm25_content.json
├── reranked_faiss_legalbert_content.json
└── final_comparison_with_reranking.csv

indices/
├── content_legal-bert/
├── content_gte-large/
├── content_bge-large/
├── metadata_legal-bert/
├── metadata_gte-large/
└── metadata_bge-large/
```

## ✅ Task Completion Checklist

- [x] Dataset pre-processed with metadata fields
- [x] Metadata joining implemented (4 fields → 1)
- [x] 10 legal queries designed
- [x] Separate retrieval for content and metadata fields
- [x] BM25 retriever implementation
- [x] FAISS with 3 embedding models (Legal-BERT + 2 SOTA)
- [x] Separate GPU notebook for FAISS index building
- [x] CPU-compatible FAISS retrieval notebook
- [x] Cross-encoder reranker for result improvement
- [x] Complete evaluation and comparison utilities

## 🎓 Understanding the Results

After running all notebooks, you'll have:

1. **BM25 baseline** - Fast, lexical matching
2. **Legal-BERT results** - Domain-specific embeddings
3. **GTE-Large results** - State-of-the-art general embeddings
4. **BGE-Large results** - Top retrieval model
5. **Reranked results** - Improved quality with cross-encoder

Compare them to determine the best approach for your legal RAG pipeline!

## 🔍 Next Steps

1. Analyze which model performs best for your queries
2. Determine if content or metadata retrieval is more effective
3. Assess whether reranking provides enough value for the added latency
4. Consider hybrid approaches (BM25 + FAISS fusion)
5. Fine-tune for your specific legal domain if needed
