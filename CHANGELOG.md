# Changelog

All notable changes to the Legal RAG Pipeline Evaluation project will be documented in this file.

## [1.0.0] - 2025-01-XX

### Initial Release

#### Added

- **Complete RAG Evaluation Framework** for legal documents
  - BM25 sparse retrieval implementation
  - FAISS dense retrieval with 3 embedding models
  - Cross-encoder reranking capability
- **Embedding Models**

  - Legal-BERT (nlpaueb/legal-bert-base-uncased) - domain-specific
  - GTE-Large (thenlper/gte-large) - state-of-the-art general
  - BGE-Large (BAAI/bge-large-en-v1.5) - top MTEB performance

- **Source Code Modules** (`src/`)

  - `config.py` - Central configuration
  - `queries.py` - 10 diverse legal queries
  - `data_loader.py` - TSV loading and metadata handling
  - `bm25_retriever.py` - Sparse retrieval
  - `faiss_retriever.py` - Dense retrieval
  - `reranker.py` - Cross-encoder reranking
  - `evaluation.py` - Metrics and comparison utilities

- **Jupyter Notebooks** (`notebooks/`)

  - `01_data_preparation.ipynb` - Dataset exploration
  - `02_bm25_retrieval.ipynb` - BM25 baseline evaluation
  - `03_faiss_index_builder.ipynb` - GPU-based index building
  - `04_faiss_retrieval.ipynb` - CPU-based retrieval testing
  - `05_reranker.ipynb` - Reranking evaluation

- **Documentation**

  - `README.md` - Project overview and quick start
  - `USAGE_GUIDE.md` - Detailed step-by-step instructions
  - `PROJECT_SUMMARY.md` - Comprehensive project summary
  - `QUICK_REFERENCE.md` - Quick reference card
  - `requirements.txt` - Python dependencies

- **Features**
  - Dual-field retrieval (content vs. metadata)
  - GPU/CPU workflow separation
  - Batch processing support
  - Result persistence (JSON)
  - Comprehensive evaluation metrics
  - Interactive result exploration
  - Visualization utilities

#### Technical Details

- Python 3.8+ compatible
- Modular architecture for easy extension
- Configurable hyperparameters
- Memory-efficient batch processing
- Automatic GPU detection and fallback

#### Dataset Support

- Pre-processed TSV format with chunked legal acts (ready to use)
- Metadata field joining (title | keywords | section | summary)
- 573K+ document chunks from Sri Lankan legal acts
- No data exploration or preparation required

---

## [Unreleased]

### Potential Future Enhancements

- [ ] Support for additional embedding models
- [ ] Advanced evaluation metrics (NDCG, MRR)
- [ ] Query expansion techniques
- [ ] Hybrid retrieval (BM25 + FAISS fusion)
- [ ] Web interface for interactive querying
- [ ] Support for other legal document formats
- [ ] Multi-language support
- [ ] Distributed indexing for larger datasets
- [ ] Fine-tuning pipeline for custom models
- [ ] Automated hyperparameter optimization

---

## Version History

- **1.0.0** (Initial Release) - Complete RAG evaluation framework with BM25, FAISS, and reranking

---

## Maintenance Notes

### Dependencies

- Core: PyTorch, Transformers, Sentence-Transformers
- Retrieval: FAISS (CPU/GPU), rank-bm25
- Data: Pandas, NumPy
- Evaluation: scikit-learn, Matplotlib, Seaborn

### Testing

- All functionality tested via Jupyter notebooks
- Manual validation of retrieval results
- Visual inspection of evaluation metrics

### Known Limitations

- GPU required for efficient index building (Notebook 03)
- Large memory requirement for BGE-Large model
- BM25 performance depends on tokenization quality
- Reranker adds latency (cross-encoder inference)

---

**Notes:**

- Keep this changelog updated with each significant change
- Follow [Semantic Versioning](https://semver.org/)
- Document breaking changes clearly
- Include migration guides for major versions
