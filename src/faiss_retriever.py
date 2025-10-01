"""
FAISS retriever implementation with multiple embedding models.
"""

import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss

from .config import (
    TOP_K,
    EMBEDDING_MODELS,
    EMBEDDING_BATCH_SIZE,
    INDICES_DIR,
    DEVICE,
)

# Import BGE-M3 model
try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_M3_AVAILABLE = True
except ImportError:
    BGE_M3_AVAILABLE = False
    print("Warning: FlagEmbedding not available. BGE-M3 model will not work.")


def get_device():
    """Get the appropriate device for computation."""
    if DEVICE is not None:
        return DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


class FAISSRetriever:
    """FAISS-based dense retriever using embedding models."""
    
    def __init__(
        self, 
        model_name: str,
        index_path: Optional[Path] = None,
        documents: Optional[List[str]] = None,
    ):
        """
        Initialize FAISS retriever.
        
        Args:
            model_name: Name of the embedding model (key from EMBEDDING_MODELS).
            index_path: Path to saved FAISS index. If None, will build new index.
            documents: List of documents (required if building new index).
        """
        self.model_key = model_name
        self.model_config = EMBEDDING_MODELS[model_name]
        self.device = get_device()
        self.library = self.model_config.get("library", "sentence-transformers")
        
        print(f"Loading model: {self.model_config['name']}")
        print(f"Library: {self.library}")
        print(f"Device: {self.device}")
        
        # Load embedding model based on library
        if self.library == "flagembedding":
            if not BGE_M3_AVAILABLE:
                raise ImportError("FlagEmbedding library is required for BGE-M3. Install with: pip install FlagEmbedding")
            self.model = BGEM3FlagModel(
                self.model_config["name"],
                use_fp16=True  # Use FP16 for faster inference
            )
        else:
            # Default to sentence-transformers
            self.model = SentenceTransformer(
                self.model_config["name"],
                device=self.device
            )
        
        self.index = None
        self.documents = documents
        
        if index_path is not None:
            self.load_index(index_path)
    
    def encode_documents(
        self, 
        documents: List[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode documents into embeddings.
        
        Args:
            documents: List of document strings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            
        Returns:
            Array of embeddings with shape (num_docs, embedding_dim).
        """
        print(f"Encoding {len(documents)} documents...")
        
        if self.library == "flagembedding":
            # BGE-M3 model encoding
            max_length = self.model_config.get("max_length", 8192)
            embeddings = self.model.encode(
                documents,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )['dense_vecs']
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            # SentenceTransformer encoding
            embeddings = self.model.encode(
                documents,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def build_index(
        self, 
        documents: List[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        use_gpu: bool = False
    ):
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of document strings.
            batch_size: Batch size for encoding.
            use_gpu: Whether to use GPU for FAISS index.
        """
        self.documents = documents
        
        # Encode documents
        embeddings = self.encode_documents(documents, batch_size)
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        
        # Move to GPU if requested
        if use_gpu and torch.cuda.is_available():
            print("Moving index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        # If index was on GPU, move back to CPU for saving
        if use_gpu and torch.cuda.is_available():
            print("Moving index back to CPU for storage...")
            index = faiss.index_gpu_to_cpu(index)
        
        self.index = index
        print(f"FAISS index built with {index.ntotal} vectors")
    
    def save_index(self, save_path: Path):
        """
        Save FAISS index to disk.
        
        Args:
            save_path: Directory path to save the index.
        """
        save_path.mkdir(parents=True, exist_ok=True)
        
        index_file = save_path / "index.faiss"
        
        print(f"Saving index to {index_file}...")
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata = {
            "model_key": self.model_key,
            "model_name": self.model_config["name"],
            "num_documents": self.index.ntotal if self.index else 0,
            "dimension": self.index.d if self.index else 0,
        }
        
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Index saved successfully")
    
    def load_index(self, load_path: Path):
        """
        Load FAISS index from disk.
        
        Args:
            load_path: Directory path containing the index.
        """
        index_file = load_path / "index.faiss"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        print(f"Loading index from {index_file}...")
        self.index = faiss.read_index(str(index_file))
        print(f"Index loaded with {self.index.ntotal} vectors")
        
        # Load metadata
        metadata_file = load_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                print(f"Model: {metadata['model_name']}")
                print(f"Dimension: {metadata['dimension']}")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = TOP_K
    ) -> Tuple[List[int], List[float], float]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string.
            top_k: Number of documents to retrieve.
            
        Returns:
            Tuple of (document indices, scores, retrieval time).
        """
        if self.index is None:
            raise ValueError("No index available. Build or load an index first.")
        
        start_time = time.time()
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), 
            top_k
        )
        
        retrieval_time = time.time() - start_time
        
        # Convert to lists
        indices = indices[0].tolist()
        scores = scores[0].tolist()
        
        return indices, scores, retrieval_time
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings.
            top_k: Number of documents to retrieve per query.
            
        Returns:
            List of retrieval results for each query.
        """
        results = []
        
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            indices, scores, ret_time = self.retrieve(query, top_k)
            
            result = {
                "query": query,
                "indices": indices,
                "scores": scores,
                "retrieval_time": ret_time,
            }
            results.append(result)
        
        return results


def build_all_indices(
    documents_content: List[str],
    documents_metadata: List[str],
    use_gpu: bool = True,
    batch_size: int = EMBEDDING_BATCH_SIZE,
):
    """
    Build FAISS indices for all embedding models and both fields.
    
    Args:
        documents_content: List of content documents.
        documents_metadata: List of metadata documents.
        use_gpu: Whether to use GPU for building indices.
        batch_size: Batch size for encoding.
    """
    print("\n" + "="*80)
    print("BUILDING FAISS INDICES FOR ALL MODELS")
    print("="*80 + "\n")
    
    for model_key in EMBEDDING_MODELS.keys():
        print(f"\n{'='*80}")
        print(f"Model: {model_key}")
        print(f"{'='*80}\n")
        
        # Build content index
        print("\n--- Building Content Index ---")
        content_retriever = FAISSRetriever(model_key)
        content_retriever.build_index(documents_content, batch_size, use_gpu)
        
        content_path = INDICES_DIR / f"content_{model_key}"
        content_retriever.save_index(content_path)
        
        # Build metadata index
        print("\n--- Building Metadata Index ---")
        metadata_retriever = FAISSRetriever(model_key)
        metadata_retriever.build_index(documents_metadata, batch_size, use_gpu)
        
        metadata_path = INDICES_DIR / f"metadata_{model_key}"
        metadata_retriever.save_index(metadata_path)
        
        print(f"\nCompleted {model_key}\n")
    
    print("\n" + "="*80)
    print("ALL INDICES BUILT SUCCESSFULLY")
    print("="*80 + "\n")


def evaluate_faiss(
    model_key: str,
    queries: List[Dict[str, Any]],
    field_name: str,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Evaluate FAISS retrieval on a set of queries.
    
    Args:
        model_key: Key of the embedding model.
        queries: List of query dictionaries.
        field_name: Name of the field ('content' or 'metadata').
        top_k: Number of documents to retrieve.
        
    Returns:
        Dictionary with evaluation results.
    """
    print(f"\n{'='*80}")
    print(f"FAISS EVALUATION - {model_key.upper()} - {field_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load index
    index_path = INDICES_DIR / f"{field_name}_{model_key}"
    retriever = FAISSRetriever(model_key, index_path=index_path)
    
    # Extract query texts
    query_texts = [q["query"] for q in queries]
    
    # Retrieve documents
    results = retriever.batch_retrieve(query_texts, top_k)
    
    # Calculate statistics
    avg_time = np.mean([r["retrieval_time"] for r in results])
    
    # Format results
    formatted_results = []
    for query_info, result in zip(queries, results):
        formatted_results.append({
            "query_id": query_info["id"],
            "query": query_info["query"],
            "category": query_info["category"],
            "retrieved_indices": result["indices"],
            "scores": result["scores"],
            "retrieval_time": result["retrieval_time"],
        })
    
    evaluation = {
        "field": field_name,
        "method": "FAISS",
        "model": model_key,
        "model_name": EMBEDDING_MODELS[model_key]["name"],
        "top_k": top_k,
        "num_queries": len(queries),
        "avg_retrieval_time": avg_time,
        "results": formatted_results,
    }
    
    print(f"\nResults:")
    print(f"  Average retrieval time: {avg_time:.4f} seconds")
    print(f"  Total queries: {len(queries)}")
    print(f"{'='*80}\n")
    
    return evaluation
