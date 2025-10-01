# Legal AI RAG Pipeline Evaluation

This project evaluates different approaches to building a Retrieval-Augmented Generation (RAG) pipeline for legal documents using the Sri Lankan legal acts dataset.

## Documentation

- **[README.md](README.md)** - This file, project overview
- **[WORKFLOW.md](WORKFLOW.md)** - Complete workflow explanation
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage instructions
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference card
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Comprehensive project summary
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## Project Structure

```
LegalAI-model-eval/
├── data/
│   └── acts_with_metadata.tsv          # Pre-processed dataset (ready to use)
├── notebooks/
│   ├── 01_bm25_retrieval.ipynb         # BM25 retrieval evaluation
│   ├── 02_faiss_index_builder.ipynb    # FAISS index creation (GPU)
│   ├── 03_faiss_retrieval.ipynb        # FAISS retrieval evaluation (CPU)
│   └── 04_reranker.ipynb               # Reranking evaluation
├── src/
│   ├── config.py                       # Configuration and constants
│   ├── data_loader.py                  # Data loading utilities
│   ├── queries.py                      # Legal query definitions
│   ├── bm25_retriever.py               # BM25 retriever implementation
│   ├── faiss_retriever.py              # FAISS retriever implementation
│   ├── reranker.py                     # Reranker implementation
│   └── evaluation.py                   # Evaluation metrics
├── indices/                            # Saved FAISS indices
│   ├── content_legalbert/
│   ├── content_gte_large/
│   ├── content_bge_large/
│   ├── metadata_legalbert/
│   ├── metadata_gte_large/
│   └── metadata_bge_large/
├── results/                            # Evaluation results
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Quick Start

1. **Install dependencies:**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Launch Jupyter:**

   ```powershell
   jupyter notebook
   ```

3. **Run notebooks in order:**
   - `01_bm25_retrieval.ipynb` - BM25 baseline (~15 min)
   - `02_faiss_index_builder.ipynb` - Build indices on GPU (~30-60 min)
   - `03_faiss_retrieval.ipynb` - FAISS evaluation on CPU (~20 min)
   - `04_reranker.ipynb` - Reranking analysis (~15 min)

**See [WORKFLOW.md](WORKFLOW.md) for detailed workflow explanation.**

## Dataset

**Note:** The dataset has been manually pre-processed and is ready for use.

The dataset (`data/acts_with_metadata.tsv`) contains chunked legal acts from Sri Lankan legislation with:

- **content**: Full text content of legal act chunks
- **short_title**: Short title of the act
- **keywords**: Extracted keywords
- **section_title**: Title of the section
- **summary**: Summary of the content

## Retrieval Approaches

### 1. BM25 Retrieval

Traditional sparse retrieval using BM25 algorithm implemented with `rank-bm25` library.

### 2. FAISS Retrieval

Dense retrieval using FAISS (Facebook AI Similarity Search) with three embedding models:

1. **Legal-BERT** (`nlpaueb/legal-bert-base-uncased`)
   - Domain-specific model trained on legal corpora
2. **GTE-Large** (`thenlper/gte-large`)
   - State-of-the-art general-purpose embedding model
   - Excellent performance on MTEB benchmarks
3. **BGE-Large** (`BAAI/bge-large-en-v1.5`)
   - Top-performing model for retrieval tasks
   - Strong performance on legal and technical documents

### 3. Reranking

Cross-encoder reranking using `ms-marco-MiniLM-L-6-v2` to improve initial retrieval results.

## Evaluation Methodology

For each retriever and field combination (content/metadata), we evaluate using 10 legal queries:

1. **Presidential election procedures**
2. **Tax obligations for businesses**
3. **Penalty provisions for offences**
4. **Court appeal procedures**
5. **Voting procedures and requirements**
6. **Assessment and collection of taxes**
7. **Defence academy establishment**
8. **Import duties and customs**
9. **Nomination process requirements**
10. **Executive powers and functions**

### Evaluation Metrics

- Retrieval time
- Top-K accuracy
- Qualitative assessment of retrieved documents

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preparation

Run `01_data_preparation.ipynb` to:

- Load and explore the dataset
- Create the joined metadata field
- Prepare data for retrieval

### Step 2: BM25 Retrieval

Run `02_bm25_retrieval.ipynb` to:

- Build BM25 indices for content and metadata
- Retrieve documents for all queries
- Evaluate BM25 performance

### Step 3: Build FAISS Indices (GPU Required)

Run `03_faiss_index_builder.ipynb` on a GPU-enabled machine to:

- Load the three embedding models
- Generate embeddings for content and metadata
- Build and save FAISS indices

### Step 4: FAISS Retrieval (CPU)

Run `04_faiss_retrieval.ipynb` to:

- Load pre-built FAISS indices
- Retrieve documents using all three models
- Compare performance across models

### Step 5: Reranking

Run `05_reranker.ipynb` to:

- Apply cross-encoder reranking
- Compare with base retrieval results
- Evaluate improvement from reranking

## Key Features

- **Separate Content and Metadata Retrieval**: Evaluates retrieval on both raw content and structured metadata
- **Multiple Embedding Models**: Compares domain-specific (Legal-BERT) with state-of-the-art general models
- **GPU/CPU Separation**: Index building on GPU, retrieval on CPU for practical deployment
- **Comprehensive Evaluation**: 10 diverse legal queries covering different aspects
- **Reranking Pipeline**: Demonstrates the benefit of two-stage retrieval

## Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU index building)
- Transformers
- FAISS (CPU and GPU versions)
- pandas, numpy
- rank-bm25
- sentence-transformers

## Results

Results will be saved in the `results/` directory with:

- Retrieval times for each method
- Retrieved document lists for each query
- Comparison tables and visualizations

## Notes

- FAISS index building requires GPU with sufficient VRAM (16GB+ recommended)
- Indices are saved and can be reused without rebuilding
- All testing/retrieval can be done on CPU after indices are built

## License

This project is for educational and research purposes.
