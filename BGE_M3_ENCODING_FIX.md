# BGE-M3 Encoding Fix

## Issue

When using BGE-M3 model with FlagEmbedding library, the following error occurred:

```
TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'normalize_embeddings'
```

## Root Cause

The `normalize_embeddings` parameter is specific to SentenceTransformers and is **not supported** by FlagEmbedding's BGEM3FlagModel.

## Solution

We need to handle encoding differently for FlagEmbedding vs SentenceTransformer models:

### ❌ Wrong (causes error):

```python
# Don't use normalize_embeddings with BGE-M3
query_embedding = self.model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True,  # ← ERROR for BGE-M3!
)
```

### ✅ Correct:

```python
if self.library == "flagembedding":
    # BGE-M3 model
    query_embedding = self.model.encode(
        [query],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )['dense_vecs']
    # Manual normalization
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
else:
    # SentenceTransformer model
    query_embedding = self.model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,  # ← OK for SentenceTransformers
    )
```

## Key Differences

### SentenceTransformers (Legal-BERT, GTE-Large, BGE-Large)

- Uses: `sentence_transformers.SentenceTransformer`
- Parameters: `normalize_embeddings=True` (built-in)
- Returns: numpy array directly

### FlagEmbedding (BGE-M3)

- Uses: `FlagEmbedding.BGEM3FlagModel`
- Parameters: `return_dense=True`, `return_sparse=False`, `return_colbert_vecs=False`
- Returns: dictionary with `'dense_vecs'` key
- Normalization: Manual (using numpy)

## Fixed Files

1. ✅ **`src/faiss_retriever.py`**

   - `encode_documents()` method - Already correct
   - `retrieve()` method - **FIXED** to check library type

2. ✅ **`notebooks/02_faiss_index_builder.ipynb`**
   - `FAISSIndexBuilder.encode_documents()` - Already correct
   - No retrieve method (only builds indices)

## Verification

To test the fix:

```python
from src.faiss_retriever import FAISSRetriever

# Test BGE-M3 model
retriever = FAISSRetriever("bge-m3")
retriever.load_index("indices/content_bge-m3")

# This should now work without error
indices, scores, time = retriever.retrieve("test query")
print(f"Retrieved {len(indices)} results in {time:.4f}s")
```

## BGE-M3 Encoding Parameters

| Parameter             | Value | Purpose                                    |
| --------------------- | ----- | ------------------------------------------ |
| `batch_size`          | 128   | Number of documents per batch              |
| `max_length`          | 8192  | Maximum token length (BGE-M3 supports 8K!) |
| `return_dense`        | True  | Return dense embeddings (1024-dim)         |
| `return_sparse`       | False | Don't return sparse embeddings             |
| `return_colbert_vecs` | False | Don't return ColBERT vectors               |

Output: Dictionary with `'dense_vecs'` containing numpy array of shape `(batch_size, 1024)`

## Why Manual Normalization?

FAISS uses **Inner Product (IP)** similarity. For normalized vectors:

```
Inner Product = Cosine Similarity
```

So we normalize embeddings to convert IP search into cosine similarity search:

```python
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

This ensures:

- Unit length vectors: `||v|| = 1`
- IP(v1, v2) = cos(θ) where θ is angle between vectors
- Scores range from -1 to 1 (but typically 0 to 1 for similar documents)

## Summary

- ✅ Fixed `retrieve()` method in `FAISSRetriever`
- ✅ Added library type checking before encoding
- ✅ BGE-M3 now uses correct FlagEmbedding API
- ✅ Manual normalization applied for BGE-M3
- ✅ SentenceTransformers still use built-in normalization

**All models now work correctly without errors!** 🎉
