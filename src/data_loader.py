"""
Data loading and preprocessing utilities for the legal acts dataset.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import ast

from .config import (
    DATA_FILE,
    COL_KEY,
    COL_SHORT_TITLE,
    COL_CHUNK_ID,
    COL_CONTENT,
    COL_KEYWORDS,
    COL_SECTION_TITLE,
    COL_SUMMARY,
    METADATA_FIELDS,
    METADATA_SEPARATOR,
)


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the legal acts dataset from TSV file.
    
    Args:
        file_path: Path to the TSV file. If None, uses default from config.
        
    Returns:
        DataFrame with the loaded data.
    """
    if file_path is None:
        file_path = DATA_FILE
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
    print(f"Loaded {len(df)} records")
    
    return df


def parse_keywords(keywords_str: str) -> List[str]:
    """
    Parse keywords from string representation of list.
    
    Args:
        keywords_str: String representation of keywords list.
        
    Returns:
        List of keyword strings.
    """
    if pd.isna(keywords_str):
        return []
    
    try:
        # Try to parse as Python literal
        keywords = ast.literal_eval(keywords_str)
        if isinstance(keywords, list):
            return keywords
    except (ValueError, SyntaxError):
        pass
    
    # Fallback: treat as comma-separated string
    return [k.strip() for k in str(keywords_str).split(",")]


def create_metadata_field(row: pd.Series) -> str:
    """
    Create a structured metadata field by joining multiple metadata columns.
    
    Args:
        row: A row from the DataFrame.
        
    Returns:
        Joined metadata string.
    """
    parts = []
    
    # Short title
    if pd.notna(row[COL_SHORT_TITLE]):
        parts.append(f"Title: {row[COL_SHORT_TITLE]}")
    
    # Section title
    if pd.notna(row[COL_SECTION_TITLE]):
        parts.append(f"Section: {row[COL_SECTION_TITLE]}")
    
    # Keywords
    if pd.notna(row[COL_KEYWORDS]):
        keywords = parse_keywords(row[COL_KEYWORDS])
        if keywords:
            keywords_str = ", ".join(keywords)
            parts.append(f"Keywords: {keywords_str}")
    
    # Summary
    if pd.notna(row[COL_SUMMARY]):
        parts.append(f"Summary: {row[COL_SUMMARY]}")
    
    return METADATA_SEPARATOR.join(parts)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset by creating the joined metadata field.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with added 'metadata' column.
    """
    print("Creating metadata field...")
    df["metadata"] = df.apply(create_metadata_field, axis=1)
    print("Metadata field created")
    
    return df


def get_documents_by_field(df: pd.DataFrame, field: str) -> List[str]:
    """
    Get all documents for a specific field.
    
    Args:
        df: DataFrame with the data.
        field: Field name ('content' or 'metadata').
        
    Returns:
        List of document strings.
    """
    if field not in df.columns:
        raise ValueError(f"Field '{field}' not found in DataFrame")
    
    return df[field].fillna("").tolist()


def get_document_metadata(df: pd.DataFrame, indices: List[int]) -> List[Dict[str, Any]]:
    """
    Get metadata for documents at specified indices.
    
    Args:
        df: DataFrame with the data.
        indices: List of document indices.
        
    Returns:
        List of metadata dictionaries.
    """
    metadata_list = []
    
    for idx in indices:
        if idx < 0 or idx >= len(df):
            continue
            
        row = df.iloc[idx]
        metadata = {
            "key": row[COL_KEY],
            "short_title": row[COL_SHORT_TITLE],
            "chunk_id": row[COL_CHUNK_ID],
            "section_title": row[COL_SECTION_TITLE],
            "keywords": parse_keywords(row[COL_KEYWORDS]),
        }
        metadata_list.append(metadata)
    
    return metadata_list


def display_sample(df: pd.DataFrame, n: int = 3):
    """
    Display sample records from the dataset.
    
    Args:
        df: DataFrame to sample from.
        n: Number of samples to display.
    """
    print(f"\n{'='*80}")
    print(f"DATASET SAMPLE ({n} records)")
    print(f"{'='*80}\n")
    
    for idx in range(min(n, len(df))):
        row = df.iloc[idx]
        print(f"Record {idx + 1}:")
        print(f"  Key: {row[COL_KEY]}")
        print(f"  Title: {row[COL_SHORT_TITLE]}")
        print(f"  Section: {row[COL_SECTION_TITLE]}")
        print(f"  Content (first 200 chars): {str(row[COL_CONTENT])[:200]}...")
        if "metadata" in df.columns:
            print(f"  Metadata (first 200 chars): {str(row['metadata'])[:200]}...")
        print()
    
    print(f"{'='*80}\n")


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """
    Save processed data to a file.
    
    Args:
        df: DataFrame to save.
        output_path: Path to save the data.
    """
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print("Data saved successfully")
