# BGE-M3 Model Integration

## Overview

BGE-M3 has been successfully integrated as the **4th embedding model** in the Legal AI RAG evaluation pipeline. This document summarizes the integration and unique capabilities of BGE-M3.

## What is BGE-M3?

**BGE-M3** (BAAI General Embedding Model 3) is a state-of-the-art embedding model distinguished by:

- **M³** = Multi-Functionality, Multi-Linguality, Multi-Granularity
- Developed by Beijing Academy of Artificial Intelligence (BAAI)
- Published February 2024 ([arXiv:2402.03216](https://arxiv.org/abs/2402.03216))
- Top performer on multiple retrieval benchmarks

## Key Features

### 1. Multi-Functionality 🔄

BGE-M3 supports **three retrieval modes** in a single model:

| Mode                 | Description                        | Use Case                          |
| -------------------- | ---------------------------------- | --------------------------------- |
| **Dense Retrieval**  | Single embedding vector (1024-dim) | Semantic similarity matching      |
| **Sparse Retrieval** | Lexical weights (like BM25)        | Keyword/exact matching            |
| **ColBERT**          | Multi-vector representation        | Fine-grained token-level matching |

For this project, we use **dense retrieval mode** to compare with other models fairly.

### 2. Multi-Linguality 🌍

- Supports **100+ languages**
- Trained on multilingual datasets (MIRACL, MKQA, MLDR)
- Excellent for cross-lingual retrieval

While our Sri Lankan legal dataset is primarily English, BGE-M3 can handle:

- Mixed-language legal documents
- Foreign language citations
- Code-switched text

### 3. Multi-Granularity 📏

The most impactful feature for legal documents:

| Model      | Max Tokens | Legal Document Coverage |
| ---------- | ---------- | ----------------------- |
| Legal-BERT | 512        | ~2 paragraphs           |
| GTE-Large  | 512        | ~2 paragraphs           |
| BGE-Large  | 512        | ~2 paragraphs           |
| **BGE-M3** | **8192**   | **~3-4 pages**          |

**8192 tokens = 16× longer context** than other models!

## Technical Specifications

```python
Model: BAAI/bge-m3
Architecture: XLM-RoBERTa-large
Parameters: ~568 million
Embedding Dimension: 1024
Max Sequence Length: 8192 tokens
Model Size: ~2.2 GB
VRAM Required: ~6-8 GB (batch size 32)
Library: FlagEmbedding
```

## Integration Changes

### 1. Code Changes

#### `src/config.py`

Added BGE-M3 configuration:

```python
"bge-m3": {
    "name": "BAAI/bge-m3",
    "description": "BGE-M3: Multi-Functionality, Multi-Linguality model",
    "max_length": 8192,
    "library": "flagembedding",
}
```

#### `src/faiss_retriever.py`

- Added FlagEmbedding library import
- Modified `__init__()` to support both sentence-transformers and FlagEmbedding
- Updated `encode_documents()` to handle BGE-M3's API:
  ```python
  embeddings = self.model.encode(
      documents,
      batch_size=batch_size,
      max_length=8192,
      return_dense=True,
      return_sparse=False,    # Use dense mode only
      return_colbert_vecs=False
  )['dense_vecs']
  ```

#### `requirements.txt`

Added dependency:

```
FlagEmbedding>=1.2.0
```

### 2. Documentation Updates

#### `README.md`

- Added BGE-M3 to model list with 🆕 badge
- Updated evaluation flow diagram
- Updated evaluation matrix: 12+ → 16+ configurations
- Updated project structure to include BGE-M3 indices

#### `COMPUTE_REQUIREMENTS.md`

- Added BGE-M3 specifications
- Updated minimum GPU requirement: 12GB → 16GB VRAM
- Updated storage requirements: ~16GB → ~22GB
- Updated timing estimates for 4 models
- Noted BGE-M3's higher VRAM usage due to 8K context

### 3. New Indices

BGE-M3 will generate 2 additional FAISS indices:

```
indices/
├── content_bge_m3/      (~2.8 GB)
│   └── index.faiss      (1024-dim × 573K vectors)
└── metadata_bge_m3/     (~2.8 GB)
    └── index.faiss      (1024-dim × 573K vectors)
```

## Performance Expectations

### Advantages of BGE-M3

1. **Long Document Understanding**

   - Can process entire legal sections (8K tokens)
   - Better context understanding for complex legal language
   - Reduced information loss from chunking

2. **State-of-the-Art Performance**

   - Top scores on MTEB retrieval benchmarks
   - Outperforms OpenAI models on multilingual tasks
   - Self-knowledge distillation training

3. **Hybrid Capability** (Future Extension)
   - Can combine dense + sparse retrieval in future work
   - Potential for weighted hybrid scoring
   - ColBERT mode for fine-grained matching

### Potential Challenges

1. **Computational Cost**

   - Slower inference due to 8K context
   - Higher VRAM requirements
   - Larger model size

2. **Overkill for Short Texts**
   - Most dataset chunks are <512 tokens
   - May not show advantage on short chunks
   - Better suited for full-document retrieval

## Usage in Notebooks

### Notebook 02: Building FAISS Indices

BGE-M3 will be processed alongside other models:

```python
from src.faiss_retriever import FAISSRetriever

# BGE-M3 will be loaded with FlagEmbedding
retriever = FAISSRetriever(
    model_name='bge-m3',
    documents=content_docs
)

# Encoding happens with 8192 max tokens
retriever.build_index(documents=content_docs, use_gpu=True)
retriever.save_index(INDICES_DIR / "content_bge_m3")
```

### Notebook 03: FAISS Retrieval

Testing BGE-M3 alongside other models:

```python
# Load BGE-M3 index
retriever_bge_m3 = FAISSRetriever(
    model_name='bge-m3',
    index_path=INDICES_DIR / "content_bge_m3"
)

# Retrieve with long query (can use up to 8K tokens)
results = retriever_bge_m3.retrieve(query, k=10)
```

### Notebook 04: Reranking

BGE-M3 results will be included in reranking comparisons:

```python
# Rerank BGE-M3 results
reranked = reranker.rerank(
    query=query,
    documents=results_bge_m3,
    top_k=10
)
```

## Evaluation Matrix (Updated)

| Configuration | Retriever              | Field        | Reranked? |
| ------------- | ---------------------- | ------------ | --------- |
| 1             | BM25                   | Content      | ❌        |
| 2             | BM25                   | Metadata     | ❌        |
| 3             | Legal-BERT             | Content      | ❌        |
| 4             | Legal-BERT             | Metadata     | ❌        |
| 5             | GTE-Large              | Content      | ❌        |
| 6             | GTE-Large              | Metadata     | ❌        |
| 7             | BGE-Large              | Content      | ❌        |
| 8             | BGE-Large              | Metadata     | ❌        |
| **9**         | **BGE-M3**             | **Content**  | ❌        |
| **10**        | **BGE-M3**             | **Metadata** | ❌        |
| 11            | BM25 + Legal-BERT      | Content      | ✅        |
| 12            | BM25 + GTE-Large       | Content      | ✅        |
| 13            | BM25 + BGE-Large       | Content      | ✅        |
| **14**        | **BM25 + BGE-M3**      | **Content**  | ✅        |
| ...           | _Various combinations_ | Both         | ✅        |

**Total: 16+ configurations** (up from 12+)

## Expected Research Insights

### Comparative Analysis

We can now compare:

1. **Domain-Specific vs General**

   - Legal-BERT (domain) vs BGE-M3 (general)
   - Does legal training beat general SOTA?

2. **Context Length Impact**

   - BGE-M3 (8K) vs others (512)
   - How much does long context help?

3. **Model Size vs Performance**

   - BGE-M3 (568M) vs others (110-335M)
   - Diminishing returns analysis

4. **Multilingual Advantage**
   - BGE-M3's multilingual training
   - Benefits for legal terminology?

### Hypothesis

**H1:** BGE-M3 will outperform other models on:

- Complex legal queries requiring broad context
- Multi-sentence queries
- Queries involving cross-references

**H2:** BGE-M3 may not show significant advantage on:

- Simple keyword queries
- Short document chunks (<512 tokens)
- High-frequency legal terms (BM25 territory)

**H3:** BGE-M3 + Reranking will achieve best overall performance

- Combines broad context understanding
- With cross-encoder precision

## Installation Note

Before running notebooks, ensure FlagEmbedding is installed:

```powershell
pip install FlagEmbedding>=1.2.0
```

Or reinstall all dependencies:

```powershell
pip install -r requirements.txt
```

## References

1. **Paper:** [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216)
2. **Model Card:** [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
3. **GitHub:** [https://github.com/FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
4. **Library Docs:** [https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3)

## Changelog

- **2025-10-01:** Integrated BGE-M3 as 4th embedding model
- Updated all documentation and code
- Revised compute requirements (16GB VRAM minimum)
- Total configurations: 12+ → 16+
- Storage requirements: ~16GB → ~22GB

---

**Summary:** BGE-M3 adds cutting-edge long-context retrieval capabilities to the evaluation pipeline, enabling comparison of 8K vs 512 token context windows on legal document retrieval tasks.
