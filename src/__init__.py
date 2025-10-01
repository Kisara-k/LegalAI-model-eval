"""
Initialization file for the src package.
"""

from .config import *
from .data_loader import *
from .queries import *
from .bm25_retriever import *
from .faiss_retriever import *
from .reranker import *
from .evaluation import *

__all__ = [
    # Config
    'PROJECT_ROOT',
    'DATA_DIR',
    'INDICES_DIR',
    'RESULTS_DIR',
    'EMBEDDING_MODELS',
    'TOP_K',
    
    # Data loader
    'load_data',
    'prepare_data',
    'create_metadata_field',
    'get_documents_by_field',
    
    # Queries
    'LEGAL_QUERIES',
    'get_all_queries',
    'get_query_texts',
    
    # Retrievers
    'BM25Retriever',
    'FAISSRetriever',
    'evaluate_bm25',
    'evaluate_faiss',
    'build_all_indices',
    
    # Reranker
    'Reranker',
    'evaluate_reranking',
    
    # Evaluation
    'compare_retrieval_methods',
    'create_comparison_table',
    'print_query_results',
]
