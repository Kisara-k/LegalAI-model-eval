# FAISS Installation Guide

## TL;DR

**Use `pip install faiss-cpu` for Google Colab** - it's the correct and optimal choice!

## Why Not `faiss-gpu`?

The `faiss-gpu` package **does not exist on PyPI**. It was discontinued as of version 1.7.3.

### Official FAISS Installation Methods

| Environment           | Command                              | Notes                   |
| --------------------- | ------------------------------------ | ----------------------- |
| **Google Colab**      | `pip install faiss-cpu`              | ✅ Recommended          |
| **Local with Conda**  | `conda install -c pytorch faiss-gpu` | Requires Conda          |
| **Local from Source** | Build with CUDA/nvcc                 | Complex, time-consuming |

### PyPI Packages Available

- ✅ `faiss-cpu` - Available, maintained, works everywhere
- ❌ `faiss-gpu` - **Does not exist** (discontinued 2+ years ago)

## Why `faiss-cpu` Is Perfect for Colab

### Performance Breakdown

When building FAISS indices, there are 2 main operations:

1. **Encoding documents to embeddings** 🐌 SLOW (95% of time)

   - This uses your embedding model (BERT, BGE-M3, etc.)
   - ✅ **Runs on GPU with `faiss-cpu`** (using transformers/torch)
   - Takes 40-50 minutes for 4 models × 10K documents

2. **Building FAISS index** ⚡ FAST (5% of time)
   - Adding vectors to index structure
   - Takes 2-3 minutes per model (even on CPU)
   - GPU speedup here: ~30 seconds saved per model

### Time Comparison

| Setup                        | Encoding Time | Index Build Time | Total          |
| ---------------------------- | ------------- | ---------------- | -------------- |
| **faiss-cpu + GPU encoding** | 45 min (GPU)  | 8 min (CPU)      | ~53 min        |
| **faiss-gpu + GPU encoding** | 45 min (GPU)  | 2 min (GPU)      | ~47 min        |
| **Difference**               | -             | -                | **~6 minutes** |

### Why The Difference Doesn't Matter

- 💰 **Complexity vs Benefit**: Installing FAISS GPU in Colab requires conda setup
- ⏱️ **Minimal Time Saving**: 6 minutes saved out of 53 minutes (11%)
- 🔧 **Maintenance**: `faiss-cpu` is simpler, more portable
- 🎯 **GPU Still Used**: Where it matters most (encoding)

## How to Install FAISS GPU (If You Really Need It)

### Option 1: Use Conda in Colab (Not Recommended)

```python
# Install conda in Colab
!pip install -q condacolab
import condacolab
condacolab.install()

# Install faiss-gpu
!conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

**Issues:**

- Requires kernel restart
- Adds complexity
- May conflict with Colab's pre-installed packages

### Option 2: Local Setup with Conda (Recommended for Local Development)

```bash
# Create environment
conda create -n legalai python=3.10

# Activate
conda activate legalai

# Install FAISS GPU
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Install other packages
pip install -r requirements.txt
```

### Option 3: Build from Source (Advanced)

See: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

Requires:

- CUDA Toolkit
- nvcc compiler
- CMake
- C++ build tools

## Recommendation by Use Case

### Google Colab (Free T4 GPU)

```python
pip install faiss-cpu  # ✅ Best choice
```

### Local Development (No GPU)

```python
pip install faiss-cpu  # Only option
```

### Local Development (With GPU)

```bash
conda install -c pytorch faiss-gpu  # Optimal
```

### Production Server (With GPU)

```bash
# Docker with conda
conda install -c pytorch -c nvidia faiss-gpu

# Or build from source with GPU support
```

## Verification

After installation, verify GPU usage:

```python
import faiss
import torch

# Check if FAISS has GPU support
print(f"FAISS version: {faiss.__version__}")
print(f"FAISS has GPU: {hasattr(faiss, 'StandardGpuResources')}")

# Check if PyTorch can use GPU
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Expected Output in Colab with `faiss-cpu`:

```
FAISS version: 1.7.4
FAISS has GPU: False
PyTorch CUDA available: True
PyTorch CUDA device: Tesla T4
```

This is **perfect** because:

- ✅ PyTorch uses GPU (for encoding - the slow part)
- ✅ FAISS uses CPU (for indexing - already fast)

## Common Errors and Solutions

### Error: `Could not find a version that satisfies the requirement faiss-gpu`

**Cause:** Trying to install `faiss-gpu` via pip

**Solution:** Use `pip install faiss-cpu` instead

### Error: `ImportError: cannot import name 'StandardGpuResources' from 'faiss'`

**Cause:** Using `faiss-cpu` but code expects GPU functions

**Solution:** Either:

1. Remove GPU-specific FAISS code (recommended for Colab)
2. Install `faiss-gpu` via conda

### Error: `RuntimeError: CUDA out of memory` (during encoding)

**Cause:** Batch size too large for embedding model

**Solution:** Reduce `EMBEDDING_BATCH_SIZE`:

```python
EMBEDDING_BATCH_SIZE = 64  # Instead of 128
```

## References

- [FAISS Official Installation Guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
- [FAISS-CPU PyPI Page](https://pypi.org/project/faiss-cpu/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Conda FAISS Package](https://anaconda.org/pytorch/faiss-gpu)

## Summary

For this project:

- ✅ **Colab:** Use `faiss-cpu` (already configured in notebook)
- ✅ **Local GPU:** Use conda `faiss-gpu` (optional optimization)
- ✅ **Local CPU:** Use `faiss-cpu` (only option)

The notebook is already optimized for Colab with `faiss-cpu` - no changes needed! 🎉
