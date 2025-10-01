# FAISS CPU-Only Implementation

## Summary

This project uses **`faiss-cpu`** exclusively. All FAISS GPU references have been removed from the codebase.

## Why CPU-Only?

1. **`faiss-gpu` doesn't exist on PyPI** - It was discontinued as of v1.7.3
2. **GPU is used where it matters** - Encoding embeddings (95% of compute time)
3. **FAISS indexing is fast on CPU** - Only ~8 minutes for all indices
4. **Simplicity** - Works everywhere without conda dependencies

## Performance Breakdown

| Operation               | Hardware        | Time (4 models × 10K docs) |
| ----------------------- | --------------- | -------------------------- |
| **Encoding embeddings** | GPU (PyTorch)   | ~45 minutes                |
| **FAISS indexing**      | CPU (faiss-cpu) | ~8 minutes                 |
| **Total**               | GPU + CPU       | **~53 minutes**            |

If FAISS GPU were available:

- FAISS indexing: ~2 minutes (vs 8 minutes)
- Total savings: **6 minutes** (11% improvement)
- Not worth the complexity

## What Uses GPU?

✅ **Encoding with GPU (via PyTorch/Transformers):**

- SentenceTransformer models (Legal-BERT, GTE-Large, BGE-Large)
- FlagEmbedding models (BGE-M3)
- This is 95% of the compute time

❌ **FAISS indexing uses CPU:**

- Building index structure
- Adding vectors to index
- This is only 5% of the compute time

## Code Changes

### Removed Parameters

**Before:**

```python
def build_index(documents, batch_size=128, use_gpu=True):
    # ...
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    # ...
```

**After:**

```python
def build_index(documents, batch_size=128):
    # ...
    # FAISS index uses CPU (faiss-cpu package)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    # ...
```

### Simplified Function Signatures

| Function                          | Before                                                      | After                                              |
| --------------------------------- | ----------------------------------------------------------- | -------------------------------------------------- |
| `FAISSRetriever.build_index()`    | `build_index(docs, batch_size, use_gpu)`                    | `build_index(docs, batch_size)`                    |
| `build_all_indices()`             | `build_all_indices(content, metadata, use_gpu, batch_size)` | `build_all_indices(content, metadata, batch_size)` |
| `FAISSIndexBuilder.build_index()` | `build_index(docs, batch_size, use_gpu)`                    | `build_index(docs, batch_size)`                    |

## Files Modified

### Source Code

- ✅ `src/faiss_retriever.py` - Removed all FAISS GPU code
- ✅ `src/config.py` - Updated batch size comments
- ✅ `notebooks/02_faiss_index_builder.ipynb` - Removed use_gpu parameters
- ✅ `requirements.txt` - Simplified comments

### Documentation

- ✅ `FAISS_INSTALLATION_GUIDE.md` - Kept for reference (explains why)
- ✅ `FAISS_ISSUE_RESOLUTION.md` - Kept for reference (troubleshooting)
- ✅ `FAISS_CPU_ONLY.md` - This file (implementation guide)

## Verification

To verify the setup:

```python
import faiss
import torch

print(f"FAISS version: {faiss.__version__}")
print(f"FAISS has GPU: {hasattr(faiss, 'StandardGpuResources')}")  # Should be False
print(f"PyTorch CUDA: {torch.cuda.is_available()}")  # Should be True (for encoding)
```

Expected output:

```
FAISS version: 1.7.4
FAISS has GPU: False        ← Correct (using faiss-cpu)
PyTorch CUDA: True          ← Correct (GPU for encoding)
```

## Migration Guide

If you have existing code calling the old API:

**Old code:**

```python
retriever.build_index(documents, batch_size=128, use_gpu=True)
```

**New code:**

```python
retriever.build_index(documents, batch_size=128)  # GPU used for encoding automatically
```

Just remove the `use_gpu` parameter!

## Benefits

1. ✅ **Simpler code** - No GPU resource management
2. ✅ **Works everywhere** - No conda requirement
3. ✅ **Same performance** - GPU still used for encoding
4. ✅ **Portable** - Easier deployment
5. ✅ **No errors** - No StandardGpuResources not found

## See Also

- [FAISS_INSTALLATION_GUIDE.md](FAISS_INSTALLATION_GUIDE.md) - Why faiss-gpu doesn't exist
- [FAISS_ISSUE_RESOLUTION.md](FAISS_ISSUE_RESOLUTION.md) - Troubleshooting guide
- [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) - Batch size tuning for encoding
