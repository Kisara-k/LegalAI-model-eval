"""
Configuration file for the Legal AI RAG evaluation project.
Contains all constants, paths, and hyperparameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = PROJECT_ROOT / "indices"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
DATA_FILE = DATA_DIR / "acts_with_metadata.tsv"

# Ensure directories exist
INDICES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Embedding models configuration
EMBEDDING_MODELS = {
    "legal-bert": {
        "name": "nlpaueb/legal-bert-base-uncased",
        "description": "Domain-specific BERT model trained on legal corpora",
        "max_length": 512,
        "library": "sentence-transformers",
    },
    "gte-large": {
        "name": "thenlper/gte-large",
        "description": "State-of-the-art general-purpose embedding model",
        "max_length": 512,
        "library": "sentence-transformers",
    },
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "description": "Top-performing model for retrieval tasks",
        "max_length": 512,
        "library": "sentence-transformers",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "description": "BGE-M3: Multi-Functionality, Multi-Linguality model with 8192 token support",
        "max_length": 8192,
        "library": "flagembedding",
    },
}

# Reranker model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval parameters
TOP_K = 10  # Number of documents to retrieve
BM25_K1 = 1.5  # BM25 k1 parameter
BM25_B = 0.75  # BM25 b parameter

# FAISS parameters
FAISS_NLIST = 100  # Number of clusters for IVF index
FAISS_NPROBE = 10  # Number of clusters to search

# Batch sizes
# Larger batches improve throughput when encoding with GPU
# For 16GB VRAM: 128-256 works well
# For 24GB+ VRAM: 256-512 works well
# For 8-12GB VRAM: Use 64-128
EMBEDDING_BATCH_SIZE = 128  # Optimized for T4/V100 GPUs
RERANKING_BATCH_SIZE = 64   # For cross-encoder reranking

# Device configuration
# Will be set at runtime based on availability
DEVICE = None  # 'cuda' or 'cpu'

# Metadata fields to join
METADATA_FIELDS = ["short_title", "keywords", "section_title", "summary"]
METADATA_SEPARATOR = " | "

# Random seed for reproducibility
RANDOM_SEED = 42

# Column names in the dataset
COL_KEY = "key"
COL_SHORT_TITLE = "short_title"
COL_CHUNK_ID = "chunk_id"
COL_CONTENT = "content"
COL_LENGTH = "length"
COL_KEYWORDS = "keywords"
COL_SECTION_TITLE = "section_title"
COL_SUMMARY = "summary"

# Result file names
RESULTS_BM25 = RESULTS_DIR / "bm25_results.json"
RESULTS_FAISS = RESULTS_DIR / "faiss_results.json"
RESULTS_RERANKER = RESULTS_DIR / "reranker_results.json"
RESULTS_COMPARISON = RESULTS_DIR / "comparison.csv"
