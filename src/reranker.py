"""
Reranker implementation using cross-encoder models.
"""

import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL, RERANKING_BATCH_SIZE, TOP_K


class Reranker:
    """Cross-encoder based reranker for improving retrieval results."""
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        """
        Initialize reranker.
        
        Args:
            model_name: Name of the cross-encoder model.
        """
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print("Reranker loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = TOP_K,
        batch_size: int = RERANKING_BATCH_SIZE
    ) -> Tuple[List[int], List[float], float]:
        """
        Rerank documents for a query.
        
        Args:
            query: Query string.
            documents: List of document strings to rerank.
            top_k: Number of documents to return after reranking.
            batch_size: Batch size for scoring.
            
        Returns:
            Tuple of (reranked indices, scores, reranking time).
        """
        start_time = time.time()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score pairs
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices].tolist()
        
        reranking_time = time.time() - start_time
        
        return top_indices.tolist(), top_scores, reranking_time
    
    def rerank_results(
        self,
        query: str,
        documents: List[str],
        initial_indices: List[int],
        initial_scores: List[float],
        top_k: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Rerank initial retrieval results.
        
        Args:
            query: Query string.
            documents: Full list of documents.
            initial_indices: Indices from initial retrieval.
            initial_scores: Scores from initial retrieval.
            top_k: Number of documents to return after reranking.
            
        Returns:
            Dictionary with reranking results.
        """
        # Get documents for initial indices
        retrieved_docs = [documents[i] for i in initial_indices]
        
        # Rerank
        reranked_local_indices, reranked_scores, rerank_time = self.rerank(
            query, retrieved_docs, top_k
        )
        
        # Map back to original indices
        reranked_indices = [initial_indices[i] for i in reranked_local_indices]
        
        return {
            "reranked_indices": reranked_indices,
            "reranked_scores": reranked_scores,
            "initial_indices": initial_indices,
            "initial_scores": initial_scores,
            "reranking_time": rerank_time,
        }
    
    def batch_rerank_results(
        self,
        queries: List[str],
        documents: List[str],
        initial_results: List[Dict[str, Any]],
        top_k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Rerank results for multiple queries.
        
        Args:
            queries: List of query strings.
            documents: Full list of documents.
            initial_results: Initial retrieval results for each query.
            top_k: Number of documents to return after reranking.
            
        Returns:
            List of reranking results for each query.
        """
        results = []
        
        for i, (query, initial_result) in enumerate(zip(queries, initial_results)):
            print(f"Reranking query {i+1}/{len(queries)}: {query[:50]}...")
            
            rerank_result = self.rerank_results(
                query=query,
                documents=documents,
                initial_indices=initial_result["indices"],
                initial_scores=initial_result["scores"],
                top_k=top_k
            )
            
            result = {
                "query": query,
                **rerank_result
            }
            results.append(result)
        
        return results


def evaluate_reranking(
    documents: List[str],
    queries: List[Dict[str, Any]],
    initial_results: List[Dict[str, Any]],
    retrieval_method: str,
    field_name: str,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Evaluate reranking on initial retrieval results.
    
    Args:
        documents: List of all documents.
        queries: List of query dictionaries.
        initial_results: Initial retrieval results.
        retrieval_method: Name of the initial retrieval method.
        field_name: Name of the field being evaluated.
        top_k: Number of documents to return after reranking.
        
    Returns:
        Dictionary with evaluation results.
    """
    print(f"\n{'='*80}")
    print(f"RERANKING EVALUATION - {retrieval_method.upper()} - {field_name.upper()}")
    print(f"{'='*80}\n")
    
    # Initialize reranker
    reranker = Reranker()
    
    # Extract query texts
    query_texts = [q["query"] for q in queries]
    
    # Prepare initial results format
    initial_results_formatted = []
    for result in initial_results:
        if "retrieved_indices" in result:
            # Format from evaluation results
            initial_results_formatted.append({
                "indices": result["retrieved_indices"],
                "scores": result["scores"],
            })
        else:
            # Already in correct format
            initial_results_formatted.append(result)
    
    # Rerank
    rerank_results = reranker.batch_rerank_results(
        query_texts, documents, initial_results_formatted, top_k
    )
    
    # Calculate statistics
    avg_rerank_time = np.mean([r["reranking_time"] for r in rerank_results])
    
    # Format results
    formatted_results = []
    for query_info, result in zip(queries, rerank_results):
        formatted_results.append({
            "query_id": query_info["id"],
            "query": query_info["query"],
            "category": query_info["category"],
            "initial_indices": result["initial_indices"],
            "initial_scores": result["initial_scores"],
            "reranked_indices": result["reranked_indices"],
            "reranked_scores": result["reranked_scores"],
            "reranking_time": result["reranking_time"],
        })
    
    evaluation = {
        "field": field_name,
        "initial_method": retrieval_method,
        "reranker": RERANKER_MODEL,
        "top_k": top_k,
        "num_queries": len(queries),
        "avg_reranking_time": avg_rerank_time,
        "results": formatted_results,
    }
    
    print(f"\nResults:")
    print(f"  Average reranking time: {avg_rerank_time:.4f} seconds")
    print(f"  Total queries: {len(queries)}")
    print(f"{'='*80}\n")
    
    return evaluation


def save_reranking_results(results: Dict[str, Any], output_file: str):
    """
    Save reranking results to JSON file.
    
    Args:
        results: Results dictionary.
        output_file: Output file path.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Reranking results saved to {output_file}")
