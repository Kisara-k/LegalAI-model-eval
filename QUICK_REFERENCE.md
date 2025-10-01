# Quick Reference Card

## 🚀 Quick Start Commands

```powershell
# 1. Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook

# 3. Run notebooks in order
# notebooks/01_bm25_retrieval.ipynb (start here)
# notebooks/02_faiss_index_builder.ipynb (GPU)
# notebooks/03_faiss_retrieval.ipynb (CPU OK)
# notebooks/04_reranker.ipynb
```

## 📁 Project Structure at a Glance

```
LegalAI-model-eval/
├── data/acts_with_metadata.tsv    # Pre-processed dataset (ready to use)
├── notebooks/                      # 4 Jupyter notebooks
├── src/                           # Python modules (handles data loading)
├── indices/                       # FAISS indices (built by notebook 02)
├── results/                       # Output results
└── requirements.txt               # Dependencies
```

## 📊 What Gets Evaluated

| Method   | Field              | Models                           | Queries |
| -------- | ------------------ | -------------------------------- | ------- |
| BM25     | Content + Metadata | -                                | 10      |
| FAISS    | Content + Metadata | Legal-BERT, GTE-Large, BGE-Large | 10      |
| Reranker | Applied to above   | ms-marco-MiniLM                  | 10      |

**Total Configurations**: 6 retrievers × 2 fields = 12+ experiments

## 🔧 Key Modules

```python
# Data
from src.data_loader import load_data, prepare_data
from src.queries import get_all_queries

# Retrieval
from src.bm25_retriever import BM25Retriever, evaluate_bm25
from src.faiss_retriever import FAISSRetriever, evaluate_faiss, build_all_indices

# Reranking
from src.reranker import Reranker, evaluate_reranking

# Evaluation
from src.evaluation import create_comparison_table, print_query_results
```

## 📝 10 Legal Queries

1. Presidential election procedures
2. Tax obligations for businesses
3. Penalty provisions for offences
4. Court appeal procedures
5. Voting procedures
6. Tax assessment and collection
7. Defence academy establishment
8. Import duties and customs
9. Nomination process
10. Commissioner-General powers

## 🤖 3 Embedding Models

| Model          | Type            | Best For             |
| -------------- | --------------- | -------------------- |
| **Legal-BERT** | Domain-specific | Legal terminology    |
| **GTE-Large**  | General SOTA    | Cross-domain queries |
| **BGE-Large**  | Top retrieval   | Ranking quality      |

## ⚡ Typical Workflow

### Option A: Full Evaluation

```
1. Data Prep (notebook 01)     →  5 min
2. BM25 (notebook 02)           →  2 min
3. Build Indices (notebook 03)  →  30-60 min (GPU)
4. FAISS Retrieval (notebook 04) →  5 min (CPU)
5. Reranking (notebook 05)      →  5 min
```

### Option B: BM25 Only

```
1. Data Prep (notebook 01)  →  5 min
2. BM25 (notebook 02)       →  2 min
```

### Option C: FAISS Only

```
1. Data Prep (notebook 01)      →  5 min
2. Build Indices (notebook 03)  →  30-60 min (GPU)
3. FAISS Retrieval (notebook 04) →  5 min (CPU)
```

## 💾 Output Files

```
results/
├── bm25_content_results.json
├── bm25_metadata_results.json
├── faiss_legal-bert_content_results.json
├── faiss_gte-large_content_results.json
├── faiss_bge-large_content_results.json
├── (same for metadata)
├── reranked_*.json
├── *_comparison.csv
└── final_comparison_with_reranking.csv
```

## 🎯 Common Tasks

### Load Data

```python
df = load_data()
df = prepare_data(df)
content_docs = df['content'].tolist()
```

### BM25 Retrieval

```python
retriever = BM25Retriever(content_docs)
indices, scores, time = retriever.retrieve("tax obligations", top_k=10)
```

### FAISS Retrieval

```python
# Load pre-built index
from src.config import INDICES_DIR
retriever = FAISSRetriever("legal-bert", index_path=INDICES_DIR / "content_legal-bert")
indices, scores, time = retriever.retrieve("presidential elections", top_k=10)
```

### Reranking

```python
reranker = Reranker()
reranked_indices, scores, time = reranker.rerank(
    query="voting procedures",
    documents=[content_docs[i] for i in initial_indices]
)
```

## 🐛 Quick Troubleshooting

| Problem               | Solution                              |
| --------------------- | ------------------------------------- |
| `ModuleNotFoundError` | `pip install -r requirements.txt`     |
| CUDA out of memory    | Reduce batch_size or use CPU          |
| Index not found       | Run notebook 03 first                 |
| Slow on CPU           | Normal for index building; use GPU    |
| Kernel crash          | Reduce batch_size, process fewer docs |

## ⚙️ Key Configuration

Edit `src/config.py`:

```python
TOP_K = 10                    # Docs to retrieve
EMBEDDING_BATCH_SIZE = 32     # Adjust for your GPU
BM25_K1 = 1.5                 # BM25 parameter
BM25_B = 0.75                 # BM25 parameter
```

## 📈 Evaluation Metrics

- **Retrieval Time**: Speed of retrieval
- **Top-K Scores**: Relevance scores
- **Overlap**: Agreement between methods
- **Diversity**: Unique documents retrieved
- **Reranking Impact**: Before/after comparison

## 🎓 Best Practices

1. ✅ Run notebook 01 first (always)
2. ✅ Use GPU for notebook 03 (index building)
3. ✅ Save results frequently
4. ✅ Run one model at a time if memory limited
5. ✅ Compare multiple methods before choosing

## 📚 Documentation

- **README.md**: Project overview
- **USAGE_GUIDE.md**: Detailed instructions
- **PROJECT_SUMMARY.md**: Complete summary
- **This file**: Quick reference

## 🔗 Useful Links

- Notebook 01: Data preparation
- Notebook 02: BM25 baseline
- Notebook 03: Index building (GPU)
- Notebook 04: FAISS evaluation (CPU)
- Notebook 05: Reranking

## 📞 Getting Help

1. Check error message
2. Review USAGE_GUIDE.md
3. Check notebook outputs
4. Verify data file exists
5. Confirm dependencies installed

## ✨ Features

- ✅ 5 comprehensive notebooks
- ✅ Modular, reusable code
- ✅ Multiple retrieval methods
- ✅ 10 legal queries
- ✅ GPU/CPU support
- ✅ Saved indices
- ✅ Complete documentation
- ✅ Ready to extend

---

**Need more details?** → See USAGE_GUIDE.md  
**Want overview?** → See README.md  
**Need full context?** → See PROJECT_SUMMARY.md

**Ready to start?** → Run `jupyter notebook` and open `notebooks/01_data_preparation.ipynb`

🎉 Happy evaluating!
