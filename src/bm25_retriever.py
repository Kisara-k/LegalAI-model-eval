"""
BM25 retriever implementation for legal document retrieval.
"""

import time
import json
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

from .config import TOP_K, BM25_K1, BM25_B


class BM25Retriever:
    """BM25-based retriever for document retrieval."""
    
    def __init__(self, documents: List[str], k1: float = BM25_K1, b: float = BM25_B):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of document strings.
            k1: BM25 k1 parameter (term frequency saturation).
            b: BM25 b parameter (length normalization).
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        print(f"Tokenizing {len(documents)} documents for BM25...")
        # Simple whitespace tokenization
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)
        print("BM25 index built successfully")
    
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
        start_time = time.time()
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices].tolist()
        
        retrieval_time = time.time() - start_time
        
        return top_indices.tolist(), top_scores, retrieval_time
    
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
    
    def get_documents(self, indices: List[int]) -> List[str]:
        """
        Get documents by their indices.
        
        Args:
            indices: List of document indices.
            
        Returns:
            List of document strings.
        """
        return [self.documents[i] for i in indices]


def evaluate_bm25(
    documents: List[str],
    queries: List[Dict[str, Any]],
    top_k: int = TOP_K,
    field_name: str = "content"
) -> Dict[str, Any]:
    """
    Evaluate BM25 retrieval on a set of queries.
    
    Args:
        documents: List of document strings.
        queries: List of query dictionaries.
        top_k: Number of documents to retrieve.
        field_name: Name of the field being evaluated.
        
    Returns:
        Dictionary with evaluation results.
    """
    print(f"\n{'='*80}")
    print(f"BM25 EVALUATION - {field_name.upper()}")
    print(f"{'='*80}\n")
    
    # Initialize retriever
    retriever = BM25Retriever(documents)
    
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
        "method": "BM25",
        "top_k": top_k,
        "num_queries": len(queries),
        "num_documents": len(documents),
        "avg_retrieval_time": avg_time,
        "results": formatted_results,
    }
    
    print(f"\nResults:")
    print(f"  Average retrieval time: {avg_time:.4f} seconds")
    print(f"  Total queries: {len(queries)}")
    print(f"{'='*80}\n")
    
    return evaluation


def save_results(results: Dict[str, Any], output_file: str):
    """
    Save retrieval results to JSON file.
    
    Args:
        results: Results dictionary.
        output_file: Output file path.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
