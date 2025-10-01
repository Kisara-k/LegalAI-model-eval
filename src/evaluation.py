"""
Evaluation utilities and metrics for RAG pipeline evaluation.
"""

import pandas as pd
import json
from typing import List, Dict, Any
from pathlib import Path


def calculate_overlap(list1: List[int], list2: List[int]) -> float:
    """
    Calculate the overlap between two lists of indices.
    
    Args:
        list1: First list of indices.
        list2: Second list of indices.
        
    Returns:
        Overlap ratio (0-1).
    """
    set1 = set(list1)
    set2 = set(list2)
    
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def compare_retrieval_methods(
    results_list: List[Dict[str, Any]],
    query_id: int
) -> pd.DataFrame:
    """
    Compare different retrieval methods for a specific query.
    
    Args:
        results_list: List of result dictionaries from different methods.
        query_id: ID of the query to compare.
        
    Returns:
        DataFrame with comparison.
    """
    comparison = []
    
    for results in results_list:
        method_name = f"{results.get('method', 'Unknown')}"
        if 'model' in results:
            method_name += f" ({results['model']})"
        
        # Find result for this query
        query_result = None
        for r in results.get('results', []):
            if r.get('query_id') == query_id:
                query_result = r
                break
        
        if query_result:
            comparison.append({
                'Method': method_name,
                'Field': results.get('field', 'Unknown'),
                'Retrieval Time (s)': query_result.get('retrieval_time', 0),
                'Top-1 Index': query_result.get('retrieved_indices', [None])[0],
                'Top-1 Score': query_result.get('scores', [None])[0],
                'Num Retrieved': len(query_result.get('retrieved_indices', [])),
            })
    
    return pd.DataFrame(comparison)


def create_comparison_table(
    results_dict: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a comprehensive comparison table of all methods.
    
    Args:
        results_dict: Dictionary mapping method names to result dictionaries.
        
    Returns:
        DataFrame with comparison.
    """
    rows = []
    
    for method_name, results in results_dict.items():
        row = {
            'Method': method_name,
            'Field': results.get('field', 'Unknown'),
            'Avg Retrieval Time (s)': results.get('avg_retrieval_time', 0),
            'Num Queries': results.get('num_queries', 0),
            'Num Documents': results.get('num_documents', 0) if 'num_documents' in results else 'N/A',
        }
        
        if 'model' in results:
            row['Model'] = results['model']
        
        if 'initial_method' in results:
            row['Initial Method'] = results['initial_method']
            row['Avg Reranking Time (s)'] = results.get('avg_reranking_time', 0)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_query_results(
    df: pd.DataFrame,
    query: str,
    indices: List[int],
    scores: List[float],
    top_n: int = 3,
    field: str = "content"
):
    """
    Print retrieved results for a query.
    
    Args:
        df: DataFrame with documents.
        query: Query string.
        indices: Retrieved document indices.
        scores: Retrieval scores.
        top_n: Number of results to display.
        field: Field name to display.
    """
    print(f"\nQuery: {query}")
    print(f"{'='*80}\n")
    
    for i, (idx, score) in enumerate(zip(indices[:top_n], scores[:top_n])):
        if idx >= len(df):
            continue
            
        row = df.iloc[idx]
        print(f"Rank {i+1} (Score: {score:.4f}):")
        print(f"  Key: {row.get('key', 'N/A')}")
        print(f"  Title: {row.get('short_title', 'N/A')}")
        
        content = str(row.get(field, ''))
        print(f"  {field.capitalize()} (first 200 chars): {content[:200]}...")
        print()


def save_comparison_table(df: pd.DataFrame, output_file: Path):
    """
    Save comparison table to CSV.
    
    Args:
        df: DataFrame to save.
        output_file: Output file path.
    """
    df.to_csv(output_file, index=False)
    print(f"Comparison table saved to {output_file}")


def analyze_retrieval_diversity(
    results: Dict[str, Any],
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze the diversity of retrieved documents.
    
    Args:
        results: Retrieval results dictionary.
        df: DataFrame with documents.
        
    Returns:
        Dictionary with diversity metrics.
    """
    all_indices = set()
    unique_acts = set()
    
    for result in results.get('results', []):
        indices = result.get('retrieved_indices', [])
        all_indices.update(indices)
        
        for idx in indices:
            if idx < len(df):
                act_key = df.iloc[idx].get('key', '')
                unique_acts.add(act_key)
    
    return {
        'total_unique_documents': len(all_indices),
        'total_unique_acts': len(unique_acts),
        'num_queries': results.get('num_queries', 0),
        'avg_unique_per_query': len(all_indices) / results.get('num_queries', 1),
    }


def print_diversity_analysis(diversity: Dict[str, Any]):
    """Print diversity analysis results."""
    print(f"\nDiversity Analysis:")
    print(f"  Total unique documents retrieved: {diversity['total_unique_documents']}")
    print(f"  Total unique acts retrieved: {diversity['total_unique_acts']}")
    print(f"  Number of queries: {diversity['num_queries']}")
    print(f"  Avg unique docs per query: {diversity['avg_unique_per_query']:.2f}")
