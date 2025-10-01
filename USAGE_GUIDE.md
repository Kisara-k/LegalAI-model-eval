# Usage Guide: Legal AI RAG Pipeline Evaluation

This guide explains how to use the complete RAG evaluation system.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Workflow](#detailed-workflow)
4. [Using the Notebooks](#using-the-notebooks)
5. [Using the Source Code](#using-the-source-code)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster index building

### Step 1: Clone/Download the Project

```bash
cd "d:\Core\_Code D\LegalAI-model-eval"
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# For GPU support (if you have CUDA-capable GPU), also install:
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Step 4: Verify Installation

```powershell
# Test imports
python -c "import pandas, torch, transformers, sentence_transformers, faiss; print('✓ All imports successful')"
```

## Quick Start

### Option 1: Run All Notebooks in Sequence

1. Open Jupyter:

   ```powershell
   jupyter notebook
   ```

2. Navigate to `notebooks/` and run in order:
   - `01_bm25_retrieval.ipynb` - Start here (dataset is pre-loaded by code)
   - `02_faiss_index_builder.ipynb` (GPU recommended)
   - `03_faiss_retrieval.ipynb` (can run on CPU)
   - `04_reranker.ipynb`

### Option 2: Use Python Scripts

Create a simple script to run evaluations:

```python
import sys
sys.path.append('.')

from src.data_loader import load_data, prepare_data, get_documents_by_field
from src.queries import get_all_queries
from src.bm25_retriever import evaluate_bm25
from src.config import RESULTS_DIR

# Load data
df = load_data()
df = prepare_data(df)

# Get documents and queries
documents_content = get_documents_by_field(df, 'content')
queries = get_all_queries()

# Evaluate BM25
results = evaluate_bm25(documents_content, queries, field_name="content")

print(f"Results saved to {RESULTS_DIR}")
```

## Detailed Workflow

**Note:** The dataset is pre-processed. All notebooks automatically load and prepare the data using `src/data_loader.py` functions.

### 1. BM25 Retrieval (Notebook 01)

**Purpose**: Evaluate traditional sparse retrieval.

**Steps**:

1. Build BM25 indices for content and metadata
2. Retrieve documents for all 10 queries
3. Compare performance between fields
4. Save results

**Key Outputs**:

- `results/bm25_content_results.json`
- `results/bm25_metadata_results.json`
- Retrieval time comparisons

**Example**:

```python
from src.bm25_retriever import BM25Retriever

retriever = BM25Retriever(documents_content)
indices, scores, time = retriever.retrieve("What are presidential election procedures?")
print(f"Top result: {df.iloc[indices[0]]['short_title']}")
```

### 2. FAISS Index Building (Notebook 02)

**Purpose**: Build dense retrieval indices using embedding models.

**⚠️ Important**:

- **Requires GPU** for efficient processing
- Takes 30-60 minutes depending on dataset size
- Run this on GPU-enabled environment (Google Colab, Kaggle, cloud instance)

**Steps**:

1. Loads 3 embedding models:
   - Legal-BERT: `nlpaueb/legal-bert-base-uncased`
   - GTE-Large: `thenlper/gte-large`
   - BGE-Large: `BAAI/bge-large-en-v1.5`
2. Generates embeddings for all documents
3. Builds FAISS indices
4. Saves indices to `indices/` directory

**Key Outputs**:

- 6 FAISS indices (3 models × 2 fields)
- Index metadata files

**Example**:

```python
from src.faiss_retriever import build_all_indices

build_all_indices(
    documents_content=content_docs,
    documents_metadata=metadata_docs,
    use_gpu=True,  # Set to False if no GPU
    batch_size=32
)
```

**Google Colab Alternative**:

```python
# In Colab notebook
!pip install -r requirements.txt

# Mount Google Drive to save indices
from google.colab import drive
drive.mount('/content/drive')

# Run index building
# Save to Drive so you can download and use locally
```

### 3. FAISS Retrieval Evaluation (Notebook 03)

**Purpose**: Evaluate dense retrieval on CPU.

**Can run on CPU** - loads pre-built indices from disk.

**Steps**:

1. Loads saved FAISS indices
2. Evaluates all 3 models on both fields
3. Compares model performance
4. Analyzes retrieval diversity

**Key Outputs**:

- `results/faiss_*_results.json` (6 files)
- Model comparison tables
- Performance visualizations

**Example**:

```python
from src.faiss_retriever import FAISSRetriever
from src.config import INDICES_DIR

# Load pre-built index
retriever = FAISSRetriever(
    "legal-bert",
    index_path=INDICES_DIR / "content_legal-bert"
)

# Retrieve
indices, scores, time = retriever.retrieve("What are tax obligations for businesses?")
```

### 4. Reranking (Notebook 04)

**Purpose**: Apply cross-encoder reranking to improve results.

**Steps**:

1. Loads previous retrieval results
2. Applies cross-encoder reranking
3. Compares before/after rankings
4. Analyzes reranking impact

**Key Outputs**:

- `results/reranked_*_results.json`
- Before/after comparisons
- Performance impact analysis

**Example**:

```python
from src.reranker import Reranker

reranker = Reranker()

# Rerank BM25 results
indices_reranked, scores_reranked, time = reranker.rerank(
    query="What are presidential election procedures?",
    documents=[documents_content[i] for i in bm25_indices],
    top_k=10
)
```

## Using the Source Code

### Data Loading

```python
from src.data_loader import load_data, prepare_data

# Load raw data
df = load_data()

# Add metadata field
df = prepare_data(df)

# Get document lists
from src.data_loader import get_documents_by_field
content_docs = get_documents_by_field(df, 'content')
metadata_docs = get_documents_by_field(df, 'metadata')
```

### Queries

```python
from src.queries import get_all_queries, get_query_by_id

# Get all queries
queries = get_all_queries()

# Get specific query
query = get_query_by_id(1)
print(query['query'])
print(query['category'])
```

### BM25 Retrieval

```python
from src.bm25_retriever import BM25Retriever, evaluate_bm25

# Create retriever
retriever = BM25Retriever(content_docs)

# Single query
indices, scores, time = retriever.retrieve("presidential elections", top_k=5)

# Batch evaluation
results = evaluate_bm25(content_docs, queries, top_k=10, field_name="content")
```

### FAISS Retrieval

```python
from src.faiss_retriever import FAISSRetriever, evaluate_faiss

# Build index
retriever = FAISSRetriever("legal-bert")
retriever.build_index(content_docs, use_gpu=True)
retriever.save_index(INDICES_DIR / "my_index")

# Load existing index
retriever = FAISSRetriever("legal-bert", index_path=INDICES_DIR / "my_index")

# Retrieve
indices, scores, time = retriever.retrieve("tax obligations", top_k=10)

# Batch evaluation
results = evaluate_faiss("legal-bert", queries, "content", top_k=10)
```

### Reranking

```python
from src.reranker import Reranker, evaluate_reranking

# Initialize reranker
reranker = Reranker()

# Rerank results
indices_reranked, scores_reranked, time = reranker.rerank(
    query="What are tax obligations?",
    documents=[content_docs[i] for i in initial_indices],
    top_k=10
)

# Batch reranking
results = evaluate_reranking(
    documents=content_docs,
    queries=queries,
    initial_results=bm25_results['results'],
    retrieval_method="BM25",
    field_name="content"
)
```

## Customization

### Adding New Queries

Edit `src/queries.py`:

```python
LEGAL_QUERIES = [
    # ... existing queries ...
    {
        "id": 11,
        "query": "Your new query here",
        "category": "Category Name",
        "keywords": ["keyword1", "keyword2"],
    },
]
```

### Adding New Embedding Models

Edit `src/config.py`:

```python
EMBEDDING_MODELS = {
    # ... existing models ...
    "my-model": {
        "name": "huggingface/model-name",
        "description": "Model description",
        "max_length": 512,
    },
}
```

Then rebuild indices in notebook 03.

### Changing Retrieval Parameters

Edit `src/config.py`:

```python
TOP_K = 20  # Retrieve top-20 instead of top-10
BM25_K1 = 1.2  # Adjust BM25 parameters
EMBEDDING_BATCH_SIZE = 64  # Increase batch size if you have more VRAM
```

### Custom Metadata Fields

Edit `src/data_loader.py` in the `create_metadata_field` function:

```python
def create_metadata_field(row: pd.Series) -> str:
    parts = []

    # Add your custom fields
    if pd.notna(row['custom_field']):
        parts.append(f"Custom: {row['custom_field']}")

    # ... rest of the function
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'rank_bm25'`

**Solution**:

```powershell
pip install rank-bm25
```

#### 2. CUDA Out of Memory (during index building)

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:

- Reduce batch size:
  ```python
  build_all_indices(..., batch_size=16)  # Instead of 32
  ```
- Use CPU (slower):
  ```python
  build_all_indices(..., use_gpu=False)
  ```
- Process models one at a time
- Use Google Colab with high-RAM runtime

#### 3. Index Not Found

**Problem**: `FileNotFoundError: Index file not found`

**Solution**:

- Make sure you ran notebook 03 first
- Check that `indices/` directory contains the indices
- Verify index names match exactly

#### 4. Slow Performance on CPU

**Problem**: Index building is very slow

**Solutions**:

- Use GPU for index building (notebook 03)
- Build indices on cloud (Colab, Kaggle)
- Download pre-built indices if available
- Reduce dataset size for testing

#### 5. Jupyter Kernel Crashes

**Problem**: Kernel dies during embedding generation

**Solutions**:

- Reduce batch size
- Process fewer documents
- Increase RAM/VRAM
- Process one model at a time

### Getting Help

1. Check error messages carefully
2. Verify all dependencies are installed
3. Ensure data file exists and is readable
4. Check GPU availability if using FAISS-GPU
5. Review notebook outputs for warnings

### Performance Tips

1. **Index Building**: Use GPU, will be 10-100x faster
2. **Retrieval**: CPU is fine, indices load quickly
3. **Batch Processing**: Larger batches = faster (if memory allows)
4. **Caching**: Indices are saved, no need to rebuild
5. **Parallel Processing**: Can run different notebooks simultaneously

## Advanced Usage

### Hybrid Retrieval

Combine BM25 and FAISS:

```python
# Get results from both
bm25_indices, bm25_scores, _ = bm25_retriever.retrieve(query)
faiss_indices, faiss_scores, _ = faiss_retriever.retrieve(query)

# Combine (simple concatenation)
combined_indices = list(bm25_indices) + list(faiss_indices)

# Remove duplicates and rerank
combined_docs = [documents[i] for i in combined_indices]
final_indices, final_scores, _ = reranker.rerank(query, combined_docs)
```

### Custom Evaluation Metrics

```python
from src.evaluation import calculate_overlap

# Calculate overlap between methods
overlap = calculate_overlap(bm25_indices, faiss_indices)
print(f"Overlap: {overlap:.1%}")

# Analyze diversity
diversity = analyze_retrieval_diversity(results, df)
print(diversity)
```

### Export Results

```python
import json
import pandas as pd

# Export to Excel
results_df = pd.DataFrame(results['results'])
results_df.to_excel('results/retrieval_results.xlsx', index=False)

# Export to JSON (pretty)
with open('results/pretty_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Next Steps

After running the complete evaluation:

1. **Analyze Results**: Compare different approaches
2. **Choose Best Method**: Based on your requirements (speed vs quality)
3. **Optimize**: Fine-tune parameters for your use case
4. **Deploy**: Integrate chosen method into your application
5. **Monitor**: Track performance in production

## Support

For issues or questions:

- Review this guide thoroughly
- Check notebook outputs and error messages
- Consult source code documentation
- Review the main README.md

Happy evaluating! 🚀
