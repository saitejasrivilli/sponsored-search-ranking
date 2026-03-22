"""
FastAPI serving endpoint for sponsored search ranking.
Two-stage pipeline: FAISS retrieval → TF-Ranking rerank.

Endpoints:
  POST /search   — full two-stage ranking pipeline
  POST /rerank   — rerank a provided candidate list
  GET  /health   — health check
  GET  /metrics  — latency and throughput stats
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import time
import os
import asyncio
from collections import deque

app = FastAPI(
    title="Sponsored Search Ranking API",
    description="Two-stage sponsored search: FAISS retrieval + TF-Ranking rerank",
    version="1.0.0",
)

# ── Request / Response schemas ────────────────────────────────────────────────

class AdCandidate(BaseModel):
    ad_id:          int
    ad_text:        str
    bm25_score:     float = 0.0
    semantic_sim:   float = 0.0
    historical_ctr: float = 0.05
    bid_cpm:        float = 1.0
    position_bias:  float = 1.0
    query_freq:     int   = 100

class SearchRequest(BaseModel):
    query_id:    int
    query_text:  str
    candidates:  List[AdCandidate]
    top_k:       int = Field(default=10, ge=1, le=100)
    floor_score: float = Field(default=0.0, ge=0.0)

class RankedAd(BaseModel):
    ad_id:       int
    ad_text:     str
    score:       float
    rank:        int
    bid_cpm:     float

class SearchResponse(BaseModel):
    query_id:        int
    query_text:      str
    results:         List[RankedAd]
    stage1_latency_ms: float
    stage2_latency_ms: float
    total_latency_ms:  float
    candidates_in:   int
    candidates_out:  int

class RerankRequest(BaseModel):
    query_id:   int
    query_text: str
    candidates: List[AdCandidate]
    top_k:      int = 10

# ── Model loader ──────────────────────────────────────────────────────────────

class ModelStore:
    """Lazy-loads TF model and FAISS index."""

    def __init__(self):
        self.ranker     = None
        self.index      = None
        self._loaded    = False
        self._load_lock = asyncio.Lock()

    async def load(self):
        async with self._load_lock:
            if self._loaded:
                return
            model_dir = os.getenv("MODEL_DIR", "models/")
            index_path = os.path.join(model_dir, "ad_index.bin")

            try:
                import tensorflow as tf
                from model.ranking_model import SponsoredSearchRanker
                self.ranker = SponsoredSearchRanker()
                # Build model with dummy input
                dummy = tf.zeros((1, 10, 6))
                self.ranker(dummy)
                ckpt = os.path.join(model_dir, "ranker_weights")
                if os.path.exists(ckpt + ".index"):
                    self.ranker.load_weights(ckpt)
                    print(f"Loaded ranker weights from {ckpt}")
                else:
                    print("[warn] No weights found — using random weights for demo")
            except Exception as e:
                print(f"[warn] TF model load failed: {e} — using mock scorer")
                self.ranker = None

            try:
                from pipeline.index_builder import AdSearchIndex
                self.index = AdSearchIndex()
                if os.path.exists(index_path):
                    self.index.load(index_path)
                    print(f"Loaded FAISS index: {self.index.index.ntotal:,} ads")
                else:
                    print("[warn] No FAISS index found — stage 1 will use candidate list directly")
                    self.index = None
            except Exception as e:
                print(f"[warn] FAISS load failed: {e}")
                self.index = None

            self._loaded = True

    def score_candidates(self, candidates: List[AdCandidate]) -> List[float]:
        """Score candidates — uses TF model if available, else heuristic."""
        if self.ranker is not None:
            try:
                import tensorflow as tf
                from model.ranking_model import FEATURE_NAMES
                features = np.array([[
                    c.bm25_score, c.semantic_sim, c.historical_ctr,
                    c.bid_cpm, np.log1p(c.query_freq), c.position_bias
                ] for c in candidates], dtype=np.float32)

                features_3d = features[np.newaxis, :, :]  # (1, n, 6)
                scores = self.ranker(tf.constant(features_3d), training=False)
                return scores.numpy()[0].tolist()
            except Exception as e:
                print(f"[warn] Model inference failed: {e} — using heuristic")

        # Heuristic fallback: weighted feature combination
        scores = []
        for c in candidates:
            score = (
                c.bm25_score      * 0.3 +
                c.semantic_sim    * 0.4 +
                c.historical_ctr  * 0.2 +
                (c.bid_cpm / 10)  * 0.1
            )
            scores.append(float(score))
        return scores


store = ModelStore()

# ── Latency tracker ───────────────────────────────────────────────────────────

class LatencyTracker:
    def __init__(self, window: int = 1000):
        self.total_ms = deque(maxlen=window)
        self.s1_ms    = deque(maxlen=window)
        self.s2_ms    = deque(maxlen=window)
        self.count    = 0

    def record(self, total, s1, s2):
        self.total_ms.append(total)
        self.s1_ms.append(s1)
        self.s2_ms.append(s2)
        self.count += 1

    def stats(self):
        def p(arr, pct):
            if not arr: return 0
            s = sorted(arr)
            return round(s[int(len(s)*pct/100)], 2)
        return {
            "requests":    self.count,
            "total_p50":   p(self.total_ms, 50),
            "total_p95":   p(self.total_ms, 95),
            "total_p99":   p(self.total_ms, 99),
            "stage1_p50":  p(self.s1_ms, 50),
            "stage2_p50":  p(self.s2_ms, 50),
        }

tracker = LatencyTracker()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await store.load()


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_loaded": store.ranker is not None,
        "index_loaded": store.index is not None,
    }


@app.get("/metrics")
def metrics():
    return tracker.stats()


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Two-stage sponsored search ranking."""
    t0 = time.perf_counter()

    # Stage 1: candidate retrieval
    s1_start = time.perf_counter()
    candidates = req.candidates  # in prod: FAISS ANN retrieval
    s1_ms = (time.perf_counter() - s1_start) * 1000

    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")

    # Apply floor score filter
    candidates = [c for c in candidates if c.bm25_score >= req.floor_score]

    # Stage 2: TF-Ranking rerank
    s2_start = time.perf_counter()
    scores = store.score_candidates(candidates)
    s2_ms = (time.perf_counter() - s2_start) * 1000

    # Sort by score descending
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )[:req.top_k]

    total_ms = (time.perf_counter() - t0) * 1000
    tracker.record(total_ms, s1_ms, s2_ms)

    return SearchResponse(
        query_id=req.query_id,
        query_text=req.query_text,
        results=[
            RankedAd(
                ad_id=c.ad_id,
                ad_text=c.ad_text,
                score=round(s, 4),
                rank=i + 1,
                bid_cpm=c.bid_cpm,
            )
            for i, (c, s) in enumerate(ranked)
        ],
        stage1_latency_ms=round(s1_ms, 2),
        stage2_latency_ms=round(s2_ms, 2),
        total_latency_ms=round(total_ms, 2),
        candidates_in=len(req.candidates),
        candidates_out=len(ranked),
    )


@app.post("/rerank")
async def rerank(req: RerankRequest):
    """Rerank a provided list of candidates."""
    scores = store.score_candidates(req.candidates)
    ranked = sorted(
        zip(req.candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )[:req.top_k]

    return {
        "query_id":  req.query_id,
        "results": [
            {"ad_id": c.ad_id, "score": round(s, 4), "rank": i+1}
            for i, (c, s) in enumerate(ranked)
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
