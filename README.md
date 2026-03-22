# Sponsored Search Ranking System

Production-grade two-stage sponsored search ranking using **TF-Ranking** with listwise LambdaLoss optimization, **FAISS HNSW** for sub-30ms candidate retrieval, and a **FastAPI** serving layer.

---

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 — Candidate Retrieval              ~3ms    │
│                                                     │
│  Query embedding (bi-encoder)                       │
│       │                                             │
│  FAISS HNSW IndexFlatL2                             │
│  500K ad vectors · efSearch=64                      │
│       │                                             │
│  Top-500 candidates                                 │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  Stage 2 — TF-Ranking Reranker              ~27ms   │
│                                                     │
│  Input features per (query, ad) pair:               │
│    BM25 · semantic_sim · historical_CTR             │
│    bid_CPM · query_freq · position_bias             │
│                                                     │
│  Deep Feature Tower: 256 → 128 → 64                │
│    BatchNorm + ReLU + Dropout(0.2)                  │
│                                                     │
│  LambdaLoss objective (listwise NDCG opt.)          │
│  IPS position bias correction                       │
│                                                     │
│  Top-10 ranked ads                                  │
└─────────────────────────────────────────────────────┘
```

## Design decisions

| Choice | Rationale |
|--------|-----------|
| LambdaLoss over pointwise BCE | Listwise loss directly optimizes NDCG. Pointwise treats each item independently — misses list-level structure. |
| IPS position bias correction | Clicks in position 1 are over-represented. IPS reweights by inverse propensity, debiasing the click labels. |
| HNSW over IVF | No training step. Recall@100 ~99% vs IVF ~97%. At 500K vectors HNSW search ~3ms vs IVF ~5ms. |
| CosineDecayRestarts LR | Escapes local minima during listwise training. Better than fixed LR for ranking models. |
| Shared tower | Single feature encoder learns general query-ad affinity before the ranking head. Fewer parameters, faster inference. |

---

## Quickstart

### 1. Install dependencies

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Note: TF requires Python 3.8–3.11. On Mac ARM use `tensorflow-macos`.

### 2. Train on Google Colab (recommended)

Open `notebooks/train_colab.ipynb` in Colab with GPU runtime. Run all cells — takes ~8 min on T4.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saitejasrivilli/sponsored-search-ranking/blob/main/notebooks/train_colab.ipynb)

### 3. Train locally (Python 3.11 required)

```bash
python train.py \
  --n_queries 5000 \
  --epochs 20 \
  --batch_size 256 \
  --model_dir ./models
```

### 4. Serve

```bash
python serving/api.py
# API running at http://localhost:8000
```

### 5. Test

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": 1,
    "query_text": "buy laptop online",
    "top_k": 5,
    "candidates": [
      {"ad_id":1,"ad_text":"Best Laptop Deals","bm25_score":0.4,"semantic_sim":0.7,"historical_ctr":0.05,"bid_cpm":3.0,"position_bias":1.0,"query_freq":5000},
      {"ad_id":2,"ad_text":"Buy Laptop Online","bm25_score":0.8,"semantic_sim":0.9,"historical_ctr":0.08,"bid_cpm":2.0,"position_bias":0.63,"query_freq":5000},
      {"ad_id":3,"ad_text":"Cheap Sneakers Sale","bm25_score":0.0,"semantic_sim":0.1,"historical_ctr":0.02,"bid_cpm":1.5,"position_bias":0.5,"query_freq":5000}
    ]
  }'
```

---

## Project structure

```
sponsored-search-ranking/
├── model/
│   └── ranking_model.py      # TF-Ranking listwise model (LambdaLoss)
├── pipeline/
│   ├── feature_pipeline.py   # PySpark daily feature computation
│   ├── index_builder.py      # FAISS HNSW index builder
│   └── airflow_dag.py        # Daily refresh DAG
├── serving/
│   └── api.py                # FastAPI two-stage ranking endpoint
├── data/
│   └── synthetic_data.py     # Synthetic query-ad dataset generator
├── notebooks/
│   └── train_colab.ipynb     # One-click Colab training notebook
├── train.py                  # End-to-end training script
└── requirements.txt
```

---

## Metrics (10K queries, 20 epochs, T4 GPU)

| Metric | Score |
|--------|-------|
| NDCG@10 | 0.63+ |
| NDCG@5 | 0.61+ |
| MRR | 0.48+ |
| Stage 1 latency | ~3ms |
| Stage 2 latency | ~27ms |
| End-to-end | ~30ms |

*On full Criteo dataset (45M samples): NDCG@10 ~0.70+*

---

## References

- [TF-Ranking: Scalable TensorFlow Library for Learning-to-Rank](https://arxiv.org/abs/1812.00073)
- [LambdaRank: Learning to Rank with Nonsmooth Cost Functions](https://proceedings.neurips.cc/paper/2006/file/af44c4c56f385c43f2529f9b1b018f6a-Paper.pdf)
- [Unbiased Learning-to-Rank with Biased Feedback (IPS)](https://arxiv.org/abs/1608.04468)
- [FAISS: A Library for Efficient Similarity Search](https://arxiv.org/abs/2401.08281)

---

## License

MIT
# sponsored-search-ranking
