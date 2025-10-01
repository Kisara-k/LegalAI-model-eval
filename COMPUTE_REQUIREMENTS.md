# Compute Requirements for FAISS Index Building

## Overview

Building FAISS indices for the Legal AI RAG pipeline involves encoding 573K+ document chunks with three embedding models. Here's what you need:

## Model Specifications

### 1. Legal-BERT (`nlpaueb/legal-bert-base-uncased`)

- **Model Size:** ~440 MB
- **Parameters:** ~110M
- **Architecture:** BERT-base
- **Embedding Dimension:** 768
- **Max Sequence Length:** 512 tokens
- **VRAM Required:** ~2-3 GB (with batch processing)

### 2. GTE-Large (`thenlper/gte-large`)

- **Model Size:** ~670 MB
- **Parameters:** ~335M
- **Architecture:** Transformer-based (larger)
- **Embedding Dimension:** 1024
- **Max Sequence Length:** 512 tokens
- **VRAM Required:** ~4-6 GB (with batch processing)

### 3. BGE-Large (`BAAI/bge-large-en-v1.5`)

- **Model Size:** ~1.3 GB
- **Parameters:** ~335M
- **Architecture:** Large transformer
- **Embedding Dimension:** 1024
- **Max Sequence Length:** 512 tokens
- **VRAM Required:** ~4-6 GB (with batch processing)

### 4. BGE-M3 (`BAAI/bge-m3`) 🆕

- **Model Size:** ~2.2 GB
- **Parameters:** ~568M
- **Architecture:** XLM-RoBERTa-large (multilingual)
- **Embedding Dimension:** 1024
- **Max Sequence Length:** **8192 tokens** (16x longer than others!)
- **VRAM Required:** ~6-8 GB (with batch processing)
- **Special Features:**
  - Multi-lingual (100+ languages)
  - Hybrid retrieval (dense + sparse + ColBERT)
  - Ideal for long legal documents
  - Superior performance on long-context tasks

## Total Compute Requirements

### GPU Requirements (Recommended)

#### **Minimum Configuration:**

- **GPU:** NVIDIA GPU with 16 GB VRAM (due to BGE-M3)
  - Examples: RTX 4060 Ti (16GB), Tesla T4 (16GB), RTX 3080 Ti
- **RAM:** 16 GB system RAM
- **Storage:** 50 GB free space
- **Time:** ~80-120 minutes for all indices

#### **Recommended Configuration:**

- **GPU:** NVIDIA GPU with 20-24 GB VRAM
  - Examples: RTX 3090, RTX 4090, A4000, A5000, V100
- **RAM:** 32 GB system RAM
- **Storage:** 50 GB free space (SSD preferred)
- **Time:** ~40-60 minutes for all indices

#### **Optimal Configuration:**

- **GPU:** NVIDIA GPU with 24+ GB VRAM
  - Examples: RTX 4090, A5000, A6000, V100 (32GB)
- **RAM:** 64 GB system RAM
- **Storage:** 100 GB NVMe SSD
- **Time:** ~25-35 minutes for all indices

### CPU-Only Alternative (Not Recommended)

If you don't have a GPU, index building is still possible but **much slower**:

- **CPU:** Modern multi-core processor (8+ cores recommended)
- **RAM:** 32 GB minimum, 64 GB recommended
- **Storage:** 50 GB free space
- **Time:** ~8-12 hours for all indices (potentially overnight)

## Memory Breakdown

### During Index Building (Peak Memory)

For **573,000 documents**:

| Component                    | Memory Required       |
| ---------------------------- | --------------------- |
| **Dataset in RAM**           | ~2-3 GB               |
| **Legal-BERT model**         | ~3 GB VRAM            |
| **GTE-Large model**          | ~6 GB VRAM            |
| **BGE-Large model**          | ~6 GB VRAM            |
| **BGE-M3 model**             | ~8 GB VRAM            |
| **Embedding storage (temp)** | ~4-5 GB RAM per model |
| **FAISS index (temp)**       | ~2-3 GB RAM per model |
| **Working memory**           | ~2-4 GB               |

**Peak VRAM:** ~8 GB (BGE-M3 model, one model at a time)  
**Peak RAM:** ~20 GB (dataset + embeddings + index)

### Stored Index Sizes (Approximate)

After building, indices saved to disk:

```
indices/
├── content_legal-bert/      (~2.1 GB)
│   └── index.faiss          (768-dim × 573K vectors)
├── content_gte-large/       (~2.8 GB)
│   └── index.faiss          (1024-dim × 573K vectors)
├── content_bge-large/       (~2.8 GB)
│   └── index.faiss          (1024-dim × 573K vectors)
├── content_bge-m3/          (~2.8 GB)
│   └── index.faiss          (1024-dim × 573K vectors)
├── metadata_legal-bert/     (~2.1 GB)
│   └── index.faiss          (768-dim × 573K vectors)
├── metadata_gte-large/      (~2.8 GB)
│   └── index.faiss          (1024-dim × 573K vectors)
├── metadata_bge-large/      (~2.8 GB)
│   └── index.faiss          (1024-dim × 573K vectors)
└── metadata_bge-m3/         (~2.8 GB)
    └── index.faiss          (1024-dim × 573K vectors)

Total: ~21-22 GB (8 indices)
```

## Batch Size Impact

The notebook uses configurable batch sizes (default: 32):

| Batch Size | VRAM Usage | Speed    | Recommendation   |
| ---------- | ---------- | -------- | ---------------- |
| 8          | ~4-5 GB    | Slowest  | 8-12GB VRAM      |
| 16         | ~6-8 GB    | Moderate | 16GB VRAM cards  |
| 32         | ~8-10 GB   | Fast     | 16-20GB VRAM     |
| 64         | ~12-14 GB  | Fastest  | 24GB+ VRAM cards |

**Note:** BGE-M3 requires more VRAM due to its 8192 token context window.

You can adjust in `src/config.py`:

```python
EMBEDDING_BATCH_SIZE = 32  # Reduce if out of memory
```

## Cloud GPU Options

If you don't have a local GPU, consider these cloud options:

### Google Colab

- **Free Tier:** T4 GPU (16 GB VRAM) - **Recommended for this project**
- **Colab Pro:** Better GPUs (V100, A100)
- **Cost:** Free or $10/month
- **Setup time:** ~5 minutes
- **Notes:** Upload dataset, run notebook 02, download indices

### Kaggle Notebooks

- **Free:** P100 GPU (16 GB VRAM)
- **Weekly quota:** 30 hours/week
- **Cost:** Free
- **Notes:** Similar to Colab, good alternative

### AWS EC2

- **Instance:** g4dn.xlarge (T4, 16 GB VRAM)
- **Cost:** ~$0.50/hour
- **Total cost:** ~$0.50 for index building
- **Notes:** More setup required

### Vast.ai / RunPod

- **GPUs:** Various options (RTX 3090, 4090, A6000)
- **Cost:** ~$0.20-$0.80/hour
- **Total cost:** ~$0.20-$0.40 for index building
- **Notes:** Best price/performance for one-time use

## Optimizations to Reduce Requirements

### 1. Reduce Batch Size

```python
# In src/config.py or notebook
EMBEDDING_BATCH_SIZE = 16  # or 8 for lower VRAM
```

### 2. Build Indices One at a Time

Instead of running all 4 models, build one model at a time:

```python
# Build only Legal-BERT first
build_all_indices(..., models=['legal-bert'])

# Then BGE-M3
build_all_indices(..., models=['bge-m3'])
```

### 3. Use Mixed Precision

```python
# In notebook 02, add:
model.half()  # Convert to FP16
```

Reduces VRAM by ~40-50%

### 4. Sample Dataset for Testing

```python
# Test with smaller subset first
df_sample = df.sample(n=50000)  # ~10% of data
```

## Recommendations by Scenario

### Scenario 1: You Have a Gaming PC

**GPU:** RTX 3060 (12GB) or RTX 4060 Ti (16GB)

- ✅ Can build all indices (may need to reduce batch size for BGE-M3)
- Set `EMBEDDING_BATCH_SIZE = 16` for 12GB, or `32` for 16GB
- Time: ~60-80 minutes
- **Action:** Run notebook 02 locally

### Scenario 2: You Have a Laptop (No GPU)

- ❌ Index building will be very slow
- **Action:** Use Google Colab (free T4 GPU, 16GB)
- Upload dataset to Colab, run notebook 02
- Download indices (~22 GB)
- Continue with notebooks 03-04 on laptop

### Scenario 3: You Have a Workstation

**GPU:** RTX 4090, A5000, etc. (24GB+)

- ✅✅ Perfect for this project
- Set `EMBEDDING_BATCH_SIZE = 64`
- Time: ~25-35 minutes
- **Action:** Run everything locally

### Scenario 4: Cloud-Only

- **Action:** Use Kaggle (free) or Colab Pro
- Build indices in cloud
- Download and use locally for retrieval

## Performance Estimates

Based on 573,000 documents with 4 models (Legal-BERT, GTE-Large, BGE-Large, BGE-M3):

| GPU Model      | VRAM  | Batch Size | Time per Model | Total Time (4 models) |
| -------------- | ----- | ---------- | -------------- | --------------------- |
| T4             | 16 GB | 24         | ~14 min        | ~56 min               |
| RTX 3060       | 12 GB | 16         | ~18 min        | ~72 min               |
| RTX 4060 Ti    | 16 GB | 32         | ~14 min        | ~56 min               |
| RTX 3090       | 24 GB | 48         | ~10 min        | ~40 min               |
| RTX 4090       | 24 GB | 64         | ~7 min         | ~28 min               |
| V100           | 16 GB | 32         | ~12 min        | ~48 min               |
| A100           | 40 GB | 96         | ~5 min         | ~20 min               |
| CPU (16 cores) | -     | 8          | ~150 min       | ~600 min (10 hours)   |

_Note: Times include loading models, encoding, and index building. BGE-M3 takes longer due to 8K context window._

## Bottom Line

### ✅ **Minimum to Get Started:**

- **GPU:** 16 GB VRAM (RTX 4060 Ti, T4, RTX 3080 Ti)
- **RAM:** 16 GB
- **Storage:** 60 GB
- **Alternative:** Google Colab (free T4 with 16GB)

### 🎯 **Recommended:**

- **GPU:** 20-24 GB VRAM (RTX 3090, 4090, V100)
- **RAM:** 32 GB
- **Storage:** 100 GB SSD

### 💰 **Most Cost-Effective:**

- Use **Google Colab free tier** (T4 GPU, 16GB VRAM)
- Build indices once (~50-60 min)
- Download indices (~22 GB)
- Run all other notebooks locally on CPU

---

**TL;DR:** BGE-M3's 8K context window requires 16GB+ VRAM. Use Google Colab free tier (T4) if you don't have a local GPU. Once indices are built, everything else runs fine on CPU!
