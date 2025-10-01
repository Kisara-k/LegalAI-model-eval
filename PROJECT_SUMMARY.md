# Project Summary: Legal AI RAG Pipeline Evaluation

## Overview

This project provides a comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) pipelines applied to legal documents, specifically Sri Lankan legal acts. The system compares multiple retrieval approaches to help determine the most effective method for legal document retrieval.

## What Has Been Created

### 1. Project Structure

```
LegalAI-model-eval/
├── data/                           # Dataset location
│   └── acts_with_metadata.tsv     # Legal acts with metadata
├── notebooks/                      # Interactive Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_bm25_retrieval.ipynb
│   ├── 03_faiss_index_builder.ipynb
│   ├── 04_faiss_retrieval.ipynb
│   └── 05_reranker.ipynb
├── src/                           # Reusable Python modules
│   ├── config.py                  # Configuration and constants
│   ├── data_loader.py             # Data loading utilities
│   ├── queries.py                 # 10 legal queries
│   ├── bm25_retriever.py          # BM25 implementation
│   ├── faiss_retriever.py         # FAISS implementation
│   ├── reranker.py                # Cross-encoder reranking
│   ├── evaluation.py              # Evaluation metrics
│   └── __init__.py
├── indices/                       # Saved FAISS indices (created by notebook 03)
├── results/                       # Evaluation results (created by notebooks)
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation
└── USAGE_GUIDE.md                # Detailed usage instructions
```

### 2. Complete Code Modules

#### `src/config.py`

- Project paths and directory structure
- Embedding model configurations (Legal-BERT, GTE-Large, BGE-Large)
- Retrieval parameters (TOP_K, batch sizes)
- Metadata field definitions

#### `src/data_loader.py`

- Load legal acts dataset from TSV
- Create joined metadata field from: title, keywords, section, summary
- Data preparation and preprocessing
- Sample display utilities

#### `src/queries.py`

- 10 diverse legal queries covering:
  - Constitutional law (presidential elections, nominations)
  - Tax law (obligations, assessments, collection)
  - Electoral law (voting, penalties, bribery)
  - Administrative law (Defence Academy)
  - Customs law (import duties)
  - Procedural law (appeals)

#### `src/bm25_retriever.py`

- BM25 sparse retrieval implementation
- Configurable k1 and b parameters
- Batch retrieval for multiple queries
- Performance evaluation and result saving

#### `src/faiss_retriever.py`

- FAISS dense retrieval implementation
- Support for 3 embedding models:
  - **Legal-BERT**: Domain-specific legal model
  - **GTE-Large**: State-of-the-art general embedding
  - **BGE-Large**: Top MTEB leaderboard model
- GPU-accelerated index building
- CPU-based index loading and retrieval
- Batch processing and evaluation

#### `src/reranker.py`

- Cross-encoder reranking using `ms-marco-MiniLM-L-6-v2`
- Improves initial retrieval results
- Batch reranking support
- Performance measurement

#### `src/evaluation.py`

- Retrieval performance comparison
- Overlap analysis between methods
- Diversity metrics
- Result visualization utilities

### 3. Jupyter Notebooks

#### Notebook 01: Data Preparation

- Loads the pre-processed legal acts dataset (manually created, ready to use)
- No exploration needed - dataset is complete
- Statistical analysis and visualization
- Prepares documents for retrieval

#### Notebook 02: BM25 Retrieval

- Implements traditional sparse retrieval
- Evaluates on content and metadata fields
- Compares performance across fields
- Saves baseline results

#### Notebook 02: FAISS Index Builder (GPU)

- **GPU-required notebook**
- Builds FAISS indices for all 3 models
- Creates 6 indices total (3 models × 2 fields)
- Saves indices to disk for later use
- Takes 30-60 minutes on GPU

#### Notebook 03: FAISS Retrieval Evaluation (CPU)

- Loads pre-built FAISS indices
- Evaluates all 3 embedding models
- Compares model performance
- Analyzes retrieval diversity
- Can run on CPU

#### Notebook 04: Reranker Evaluation

- Applies cross-encoder reranking
- Compares before/after rankings
- Analyzes reranking impact
- Creates final performance comparison

### 4. Documentation

#### README.md

- Project overview
- Dataset description
- Retrieval approaches explanation
- Installation instructions
- Basic usage

#### USAGE_GUIDE.md

- Detailed step-by-step instructions
- Code examples for each module
- Customization guide
- Troubleshooting section
- Advanced usage patterns

## Key Features

### Comprehensive Evaluation

- **2 Retrieval Methods**: BM25 (sparse) and FAISS (dense)
- **3 Embedding Models**: Legal-BERT, GTE-Large, BGE-Large
- **2 Fields**: Content and metadata
- **10 Legal Queries**: Diverse query types
- **Reranking**: Cross-encoder for improved results

### Modular Design

- Reusable Python modules
- Clear separation of concerns
- Easy to extend with new models or methods
- Well-documented code

### Production-Ready

- GPU/CPU separation (build on GPU, deploy on CPU)
- Saved indices for fast loading
- Batch processing support
- Performance metrics and logging

### Research-Focused

- Multiple baselines for comparison
- Detailed evaluation metrics
- Result visualization
- Reproducible experiments

## Retrieval Methods Evaluated

### 1. BM25 (Sparse Retrieval)

- **Algorithm**: Okapi BM25
- **Pros**: Fast, interpretable, no training required
- **Cons**: Vocabulary mismatch, no semantic understanding
- **Use Case**: Baseline, keyword-based queries

### 2. FAISS with Legal-BERT

- **Model**: `nlpaueb/legal-bert-base-uncased`
- **Pros**: Domain-specific, understands legal terminology
- **Cons**: Limited to legal domain
- **Use Case**: Legal-specific queries

### 3. FAISS with GTE-Large

- **Model**: `thenlper/gte-large`
- **Pros**: SOTA general embedding, cross-domain performance
- **Cons**: Larger model, slower encoding
- **Use Case**: General legal information retrieval

### 4. FAISS with BGE-Large

- **Model**: `BAAI/bge-large-en-v1.5`
- **Pros**: Top retrieval performance, excellent ranking
- **Cons**: Larger model size
- **Use Case**: High-quality retrieval tasks

### 5. Reranking

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Pros**: Improves any retrieval method, accurate scoring
- **Cons**: Adds latency, can't add new documents
- **Use Case**: Two-stage retrieval for production

## Legal Queries Designed

1. **Presidential election procedures when office becomes vacant**
2. **Tax obligations and payment requirements for businesses**
3. **Penalty provisions for bribery and undue influence**
4. **Appeal process to Board of Review and Court of Appeal**
5. **Voting procedures and ballot requirements**
6. **Tax assessment and collection powers of assessors**
7. **Defence Academy establishment and governance**
8. **Import duty and customs procedures**
9. **Presidential candidate nomination process**
10. **Commissioner-General powers in tax administration**

## How to Use This Project

### Quick Start (3 Steps)

1. **Install Dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

2. **Run Data Preparation**

   - Open `notebooks/01_data_preparation.ipynb`
   - Run all cells

3. **Choose Your Path**
   - **BM25 Only**: Run notebook 02
   - **Full Evaluation**: Run notebooks 02-05 in sequence
   - **Just Reranking**: Run notebooks 02, then 05

### For GPU Index Building

1. Open notebook 03 on GPU machine (or Google Colab)
2. Run all cells to build indices
3. Download indices to local machine
4. Continue with notebook 04 on CPU

### For Analysis

- All notebooks generate visualizations
- Results saved to `results/` directory
- Comparison tables exported as CSV
- Individual results in JSON format

## Evaluation Metrics

### Performance Metrics

- **Retrieval Time**: Average time per query
- **Top-K Accuracy**: Quality of top results
- **Diversity**: Unique documents and acts retrieved

### Comparison Metrics

- **Overlap**: Agreement between methods
- **Before/After Reranking**: Impact analysis
- **Field Comparison**: Content vs metadata

## Expected Outcomes

After running the complete evaluation, you'll have:

1. **Quantitative Results**

   - Retrieval times for all methods
   - Performance comparisons
   - Statistical analysis

2. **Qualitative Insights**

   - Which model works best for which queries
   - Content vs metadata trade-offs
   - Reranking impact

3. **Saved Artifacts**
   - FAISS indices (reusable)
   - JSON results (all experiments)
   - CSV comparisons (spreadsheet-ready)
   - Visualizations (presentation-ready)

## Customization Options

### Easy Customizations

- Add new queries to `src/queries.py`
- Change TOP_K or batch sizes in `src/config.py`
- Adjust BM25 parameters (k1, b)
- Add more metadata fields

### Advanced Customizations

- Add new embedding models
- Implement hybrid retrieval (BM25 + FAISS)
- Create custom evaluation metrics
- Add new reranking models

## Technical Requirements

### Minimum Requirements

- Python 3.8+
- 8GB RAM
- 10GB disk space (for indices)

### Recommended for Full Evaluation

- Python 3.10+
- 16GB RAM
- CUDA-capable GPU (for index building)
- 20GB disk space

### Cloud Alternatives

- Google Colab (free GPU)
- Kaggle Notebooks (free GPU)
- AWS/Azure/GCP instances

## Research Applications

This framework can be used for:

1. **RAG Pipeline Development**

   - Compare retrieval methods
   - Optimize for your use case
   - A/B testing different approaches

2. **Legal AI Research**

   - Domain adaptation studies
   - Legal retrieval benchmarking
   - Multi-lingual extensions

3. **Production Deployment**
   - Choose best method for your requirements
   - Benchmark before deployment
   - Performance monitoring baseline

## Limitations and Considerations

### Current Limitations

- English language only
- Sri Lankan legal acts (domain-specific)
- No relevance judgments (manual evaluation needed)
- Limited query diversity (10 queries)

### Future Enhancements

- Add manual relevance annotations
- Expand query set
- Multi-language support
- Hybrid retrieval methods
- Fine-tuning experiments

## Next Steps After Evaluation

1. **Analyze Results**

   - Review all comparison tables
   - Identify best-performing method
   - Consider trade-offs (speed vs quality)

2. **Choose Deployment Method**

   - Single-stage: BM25 or FAISS only
   - Two-stage: Retrieval + Reranking
   - Hybrid: Combine multiple methods

3. **Optimize**

   - Fine-tune parameters
   - Adjust for your latency requirements
   - Consider caching strategies

4. **Integrate**
   - Build API around chosen method
   - Add to your RAG pipeline
   - Monitor production performance

## Conclusion

This project provides a complete, production-ready framework for evaluating RAG retrieval methods on legal documents. All code is modular, well-documented, and ready to use. The notebooks guide you through the entire process, from data loading to final evaluation.

**Everything you need is included:**

- ✅ Complete source code
- ✅ 5 detailed notebooks
- ✅ 10 legal queries
- ✅ Multiple retrieval methods
- ✅ Comprehensive documentation
- ✅ Ready to run

**Start evaluating immediately** by following the USAGE_GUIDE.md!

---

_Project created for Legal AI RAG pipeline evaluation._  
_All testing done on Jupyter notebooks as requested._
