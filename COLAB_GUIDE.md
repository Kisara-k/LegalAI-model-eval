# Running FAISS Index Builder on Google Colab

This guide explains how to use the standalone `02_faiss_index_builder.ipynb` notebook on Google Colab.

## Why Use Google Colab?

- **Free GPU access** (NVIDIA T4 with 16GB VRAM)
- No local GPU required
- Build indices in ~50-60 minutes
- Download and use locally

## Quick Start Guide

### Step 1: Open in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `02_faiss_index_builder.ipynb`

**Or use this direct link:**

```
https://colab.research.google.com/github/Kisara-k/LegalAI-model-eval/blob/main/notebooks/02_faiss_index_builder.ipynb
```

### Step 2: Enable GPU

1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Choose **T4 GPU** (free tier)
4. Click **Save**

### Step 3: Upload Dataset

1. Click the **folder icon** 📁 on the left sidebar
2. Click the **upload** button
3. Upload your `acts_with_metadata.tsv` file
4. Wait for upload to complete (~50-100 MB)

**Alternative:** If file is in Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
data_file = '/content/drive/MyDrive/path/to/acts_with_metadata.tsv'
```

### Step 4: Run All Cells

1. Click **Runtime → Run all** (or press `Ctrl+F9`)
2. Wait for completion (~50-60 minutes)
3. Monitor progress in output

### Step 5: Download Indices

After completion:

1. The notebook automatically creates `indices.zip`
2. Download will start automatically in Colab
3. Extract `indices.zip` to your local project's `indices/` folder

## What the Notebook Does

### ✅ Completely Standalone

- **No imports from `src/`** - Everything is self-contained
- **All dependencies installed** - First cell installs all packages
- **All helper functions included** - FAISSIndexBuilder class defined inline

### 📦 Builds 8 FAISS Indices

| Model      | Content Index | Metadata Index | Total Size |
| ---------- | ------------- | -------------- | ---------- |
| Legal-BERT | ✅            | ✅             | ~4.2 GB    |
| GTE-Large  | ✅            | ✅             | ~5.6 GB    |
| BGE-Large  | ✅            | ✅             | ~5.6 GB    |
| BGE-M3     | ✅            | ✅             | ~5.6 GB    |

**Total:** ~22 GB of indices

### ⚡ Optimized for T4 GPU

- Batch size: 128 (uses ~70-80% of 16GB VRAM)
- Mixed precision (FP16) for BGE-M3
- GPU memory monitoring included

## Important Note About FAISS

**FAISS GPU package is not available via pip!**

- PyPI only has `faiss-cpu` (GPU version discontinued since v1.7.3)
- For GPU FAISS, you need `conda install -c pytorch faiss-gpu`
- Google Colab doesn't have conda by default

**However, this doesn't matter because:**

- ✅ GPU is used for **encoding documents** (the slow part - 95% of time)
- ✅ FAISS index operations are **already fast on CPU**
- ✅ Building indices on CPU FAISS takes only ~2-3 minutes per model
- ✅ Total time impact: ~5 minutes across all models (negligible)

**Bottom line:** Using `faiss-cpu` with GPU for embeddings is the optimal setup for Colab!

## Notebook Structure

```
Cell 1:  Install Dependencies (pip install faiss-cpu, etc.)
Cell 2:  Import Libraries
Cell 3:  Configuration (models, batch size, paths)
Cell 4:  Helper Functions (FAISSIndexBuilder class)
Cell 5:  Check GPU Availability
Cell 6:  Upload & Load Dataset
Cell 7:  Prepare Data Fields (metadata joining)
Cell 8:  Review Models Configuration
Cell 9:  Build All Indices (main processing)
Cell 10: Monitor GPU Usage (optional, run in parallel)
Cell 11: Verify Saved Indices
Cell 12: Download Indices (auto-zip)
Cell 13: Test Index Loading
Cell 14: Summary
```

## Troubleshooting

### Issue: "No GPU Available"

**Solution:**

1. Runtime → Change runtime type → GPU
2. If GPU quota exhausted, wait or use Kaggle instead

### Issue: "Out of Memory (OOM)"

**Solution:** Reduce batch size in Cell 3:

```python
EMBEDDING_BATCH_SIZE = 64  # Instead of 128
```

### Issue: "Upload Failed"

**Solution:** Use Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
data_file = '/content/drive/MyDrive/acts_with_metadata.tsv'
```

### Issue: "Download Not Working"

**Solution:** Download manually:

1. Click folder icon 📁
2. Right-click `indices.zip`
3. Click **Download**

### Issue: "Session Disconnected"

**Solution:**

- Colab has 12-hour limit
- Keep tab active
- Use Colab Pro for longer sessions

## Performance Expectations

### On Google Colab T4 GPU (Free Tier)

| Metric              | Value                 |
| ------------------- | --------------------- |
| **GPU**             | NVIDIA T4 (16GB VRAM) |
| **Batch Size**      | 128                   |
| **GPU Utilization** | 70-85%                |
| **Time per Model**  | ~12-14 min            |
| **Total Time**      | ~50-60 min            |
| **Output Size**     | ~22 GB                |

### Processing Speed

- **Legal-BERT:** ~12 min (smaller model)
- **GTE-Large:** ~13 min
- **BGE-Large:** ~13 min
- **BGE-M3:** ~14-16 min (8K context window)

## Tips & Tricks

### 1. Monitor GPU in Real-Time

Run this in a new cell while building:

```python
import time
while True:
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    time.sleep(5)
```

### 2. Save Checkpoints

Modify the build loop to save after each model:

```python
# After each model builds, download that index
!zip -r {model_key}_indices.zip indices/{model_key}_*
```

### 3. Use Smaller Dataset for Testing

Test with subset first:

```python
# After loading df
df = df.head(10000)  # Only 10K documents for testing
```

### 4. Increase Speed (if you have Colab Pro)

```python
EMBEDDING_BATCH_SIZE = 256  # Use more GPU
# Faster: ~30-40 minutes total
```

## Alternative: Kaggle Notebooks

If Colab GPU is unavailable, use Kaggle:

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Create new notebook
3. Enable GPU (P100, 16GB free)
4. Upload dataset
5. Copy-paste cells from this notebook

**Advantages:**

- P100 GPU (slightly faster than T4)
- 30 hours/week quota
- Better for long sessions

## Post-Processing (After Download)

### Extract Indices Locally

```powershell
# On your local machine
Expand-Archive -Path indices.zip -DestinationPath .

# Or use 7-Zip, WinRAR, etc.
```

### Verify Extraction

```powershell
tree indices /F

# Should show:
# indices/
# ├── content_legal-bert/
# ├── content_gte-large/
# ├── content_bge-large/
# ├── content_bge-m3/
# ├── metadata_legal-bert/
# ├── metadata_gte-large/
# ├── metadata_bge-large/
# └── metadata_bge-m3/
```

### Use in Local Notebooks

Now you can run notebooks 03 and 04 on CPU:

```python
from src.faiss_retriever import FAISSRetriever

# Load pre-built index
retriever = FAISSRetriever(
    'legal-bert',
    index_path='indices/content_legal-bert'
)

# No GPU needed for retrieval!
results = retriever.retrieve(query, k=10)
```

## Cost Comparison

| Option                  | GPU       | Time    | Cost             |
| ----------------------- | --------- | ------- | ---------------- |
| **Google Colab (Free)** | T4 16GB   | ~60 min | $0 ✅            |
| **Colab Pro**           | T4/V100   | ~40 min | $10/month        |
| **Kaggle (Free)**       | P100 16GB | ~50 min | $0 ✅            |
| **AWS EC2 g4dn.xlarge** | T4 16GB   | ~60 min | ~$0.50           |
| **Local RTX 4090**      | 24GB      | ~28 min | $1600 (one-time) |

**Recommendation:** Use **Google Colab free tier** for one-time index building!

## Summary Checklist

- [ ] Open notebook in Colab
- [ ] Enable GPU (T4)
- [ ] Upload `acts_with_metadata.tsv`
- [ ] Run all cells
- [ ] Wait ~60 minutes
- [ ] Download `indices.zip`
- [ ] Extract to local `indices/` folder
- [ ] Continue with notebooks 03-04 on CPU

---

**Questions?** Check the main README.md or GPU_OPTIMIZATION.md for more details.

**Total Time Investment:** ~10 min setup + 60 min processing = **70 minutes** for all 8 indices!
