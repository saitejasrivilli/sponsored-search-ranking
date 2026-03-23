"""
Synthetic query-ad dataset generator for sponsored search ranking.
Produces realistic query-ad pairs with click labels, BM25 scores,
semantic similarity, and user context features.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import hashlib


QUERY_TEMPLATES = [
    "buy {} online", "best {} deals", "{} reviews",
    "cheap {}", "{} near me", "top {} 2024",
    "{} comparison", "discount {}", "{} price",
]

PRODUCTS = [
    "laptop", "phone", "headphones", "camera", "tablet",
    "sneakers", "watch", "jacket", "backpack", "sunglasses",
    "coffee maker", "blender", "air fryer", "tv", "monitor",
]

AD_TEMPLATES = [
    "Shop {product} — Free Shipping",
    "Best {product} Deals — Up to 50% Off",
    "{product} Reviews & Ratings",
    "Buy {product} Online — Lowest Price",
    "Premium {product} Collection",
    "{product} Sale — Limited Time Offer",
]


def _bm25_sim(query: str, ad_text: str) -> float:
    """Approximate BM25 similarity via token overlap."""
    q_tokens = set(query.lower().split())
    a_tokens = set(ad_text.lower().split())
    overlap = len(q_tokens & a_tokens)
    return overlap / (len(q_tokens) + 1e-8)


def _semantic_sim(query: str, ad_text: str, seed: int) -> float:
    """Mock semantic similarity — in prod this is a bi-encoder."""
    rng = np.random.default_rng(seed)
    base = _bm25_sim(query, ad_text)
    return float(np.clip(base + rng.normal(0, 0.1), 0, 1))


def generate_dataset(
    n_queries: int = 5000,
    ads_per_query: int = 10,
    click_noise: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic query-ad relevance dataset.

    Schema:
        query_id        int
        query_text      str
        ad_id           int
        ad_text         str
        label           float  click label [0, 1]
        bm25_score      float  lexical similarity
        semantic_sim    float  dense similarity (mock bi-encoder)
        historical_ctr  float  ad historical CTR
        bid_cpm         float  advertiser bid
        position_bias   float  IPS weight (inverse propensity)
        query_freq      int    query frequency in logs

    Returns:
        DataFrame with n_queries × ads_per_query rows
    """
    rng = np.random.default_rng(seed)
    rows = []

    for qid in range(n_queries):
        product = rng.choice(PRODUCTS)
        template = rng.choice(QUERY_TEMPLATES)
        query = template.format(product)
        query_freq = int(rng.integers(10, 10000))

        for aid in range(ads_per_query):
            # Mix relevant and irrelevant ads
            if aid < 4:
                ad_product = product  # relevant
            else:
                ad_product = rng.choice([p for p in PRODUCTS if p != product])

            ad_template = rng.choice(AD_TEMPLATES)
            ad_text = ad_template.format(product=ad_product)

            bm25 = _bm25_sim(query, ad_text)
            sem = _semantic_sim(query, ad_text, seed=qid * 1000 + aid)
            hist_ctr = float(np.clip(rng.beta(2, 20), 0.001, 0.3))
            bid_cpm = float(rng.uniform(0.5, 8.0))

            # Click probability — higher for relevant ads
            relevance = (bm25 * 0.4 + sem * 0.4 + hist_ctr * 0.2)
            click_prob = float(np.clip(relevance + rng.normal(0, click_noise), 0, 1))
            label = float(rng.random() < click_prob)

            # Position bias — earlier positions have higher propensity
            position = aid + 1
            propensity = 1.0 / np.log2(position + 1)

            rows.append({
                "query_id":      qid,
                "query_text":    query,
                "ad_id":         qid * ads_per_query + aid,
                "ad_text":       ad_text,
                "label":         label,
                "bm25_score":    round(bm25, 4),
                "semantic_sim":  round(sem, 4),
                "historical_ctr":round(hist_ctr, 4),
                "bid_cpm":       round(bid_cpm, 2),
                "position_bias": round(propensity, 4),
                "query_freq":    query_freq,
            })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} rows ({n_queries} queries × {ads_per_query} ads)")
    print(f"Click rate: {df['label'].mean():.3f}")
    print(f"Columns: {list(df.columns)}")
    return df


def train_val_test_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by query_id to prevent leakage."""
    query_ids = df["query_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(query_ids)

    n = len(query_ids)
    n_val  = int(n * val_frac)
    n_test = int(n * test_frac)

    test_ids = set(query_ids[:n_test])
    val_ids  = set(query_ids[n_test:n_test + n_val])

    test_df = df[df["query_id"].isin(test_ids)].copy()
    val_df  = df[df["query_id"].isin(val_ids)].copy()
    train_df = df[~df["query_id"].isin(test_ids | val_ids)].copy()

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    df = generate_dataset(n_queries=5000, ads_per_query=10)
    train_df, val_df, test_df = train_val_test_split(df)
    df.to_csv("data/synthetic_search_data.csv", index=False)
    print("Saved to data/synthetic_search_data.csv")
