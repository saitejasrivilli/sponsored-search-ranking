"""
Sponsored Search Ranking Model using TF-Ranking.

Architecture:
  - Input: query-ad feature vectors (BM25, semantic sim, historical CTR,
    bid CPM, query frequency, position bias)
  - Model: deep listwise ranker with LambdaLoss objective
  - Loss: LambdaLoss (listwise) — optimizes NDCG directly
  - Output: relevance score per (query, ad) pair

Design choices:
  LambdaLoss over pointwise BCE:
    Listwise loss directly optimizes ranking metrics (NDCG).
    Pointwise BCE treats each item independently — misses list-level structure.
  IPS debiasing:
    Position bias correction via Inverse Propensity Scoring.
    Clicked items in position 1 are over-represented — IPS reweights.
  Shared feature tower + task head:
    Shared layers learn general query-ad affinity.
    Separate head keeps ranking objective clean.
"""

import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np
from typing import Dict, Tuple


FEATURE_NAMES = [
    "bm25_score",
    "semantic_sim",
    "historical_ctr",
    "bid_cpm",
    "query_freq_log",
    "position_bias",
]
NUM_FEATURES = len(FEATURE_NAMES)
LIST_SIZE = 10  # ads per query


def build_feature_tower(
    input_dim: int,
    hidden_units: Tuple[int, ...] = (256, 128, 64),
    dropout_rate: float = 0.2,
    name: str = "feature_tower",
) -> tf.keras.Model:
    """
    Shared feature tower — encodes query-ad feature vectors.
    Uses BatchNorm + ReLU + Dropout for stability.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units, name=f"{name}_dense_{i}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_{i}")(x)
        x = tf.keras.layers.ReLU(name=f"{name}_relu_{i}")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_drop_{i}")(x)

    score = tf.keras.layers.Dense(1, name=f"{name}_score")(x)
    return tf.keras.Model(inputs=inputs, outputs=score, name=name)


class SponsoredSearchRanker(tf.keras.Model):
    """
    Listwise ranking model for sponsored search.

    Input shape:  (batch, list_size, num_features)
    Output shape: (batch, list_size)  — relevance scores

    Training objective: LambdaLoss (approximates NDCG gradient)
    Evaluation metrics: NDCG@1, NDCG@5, NDCG@10, MRR
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        list_size: int = LIST_SIZE,
        hidden_units: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.list_size = list_size
        self.temperature = temperature

        # Feature tower applied to each (query, ad) pair independently
        self.tower = build_feature_tower(
            input_dim=num_features,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
        )

        # LambdaLoss — listwise, directly optimizes NDCG
        self.loss_fn = tfr.keras.losses.LambdaWeightedRankingLoss(
            lambda_weight=tfr.keras.losses.LambdaWeight(
                name="lambda_weight",
            )
        )

        # Ranking metrics
        self.ndcg_1  = tfr.keras.metrics.NDCGMetric(name="ndcg_1",  topn=1)
        self.ndcg_5  = tfr.keras.metrics.NDCGMetric(name="ndcg_5",  topn=5)
        self.ndcg_10 = tfr.keras.metrics.NDCGMetric(name="ndcg_10", topn=10)
        self.mrr     = tfr.keras.metrics.MRRMetric(name="mrr")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            inputs: (batch, list_size, num_features)
            training: bool

        Returns:
            scores: (batch, list_size)
        """
        batch_size = tf.shape(inputs)[0]

        # Reshape to (batch * list_size, num_features) for tower
        flat = tf.reshape(inputs, [-1, self.num_features])
        scores_flat = self.tower(flat, training=training)  # (batch*list, 1)

        # Reshape back to (batch, list_size)
        scores = tf.reshape(scores_flat, [batch_size, -1])
        return scores / self.temperature

    def train_step(self, data):
        features, labels, sample_weights = data

        with tf.GradientTape() as tape:
            scores = self(features, training=True)
            loss = self.loss_fn(labels, scores, sample_weight=sample_weights)
            loss += sum(self.losses)  # regularization

        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.ndcg_1.update_state(labels, scores)
        self.ndcg_5.update_state(labels, scores)
        self.ndcg_10.update_state(labels, scores)
        self.mrr.update_state(labels, scores)

        return {
            "loss":    loss,
            "ndcg@1":  self.ndcg_1.result(),
            "ndcg@5":  self.ndcg_5.result(),
            "ndcg@10": self.ndcg_10.result(),
            "mrr":     self.mrr.result(),
        }

    def test_step(self, data):
        features, labels, sample_weights = data
        scores = self(features, training=False)
        loss = self.loss_fn(labels, scores, sample_weight=sample_weights)

        self.ndcg_1.update_state(labels, scores)
        self.ndcg_5.update_state(labels, scores)
        self.ndcg_10.update_state(labels, scores)
        self.mrr.update_state(labels, scores)

        return {
            "loss":    loss,
            "ndcg@1":  self.ndcg_1.result(),
            "ndcg@5":  self.ndcg_5.result(),
            "ndcg@10": self.ndcg_10.result(),
            "mrr":     self.mrr.result(),
        }

    @property
    def metrics(self):
        return [self.ndcg_1, self.ndcg_5, self.ndcg_10, self.mrr]


def build_dataset(
    df,
    list_size: int = LIST_SIZE,
    batch_size: int = 256,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Convert a DataFrame to a tf.data.Dataset for listwise training.

    Each example is one query with list_size ads.
    Features shape: (list_size, num_features)
    Labels shape:   (list_size,)
    Weights shape:  (list_size,)  — IPS position bias correction
    """
    import numpy as np

    query_ids = df["query_id"].unique()
    features_list, labels_list, weights_list = [], [], []

    for qid in query_ids:
        qdf = df[df["query_id"] == qid].head(list_size)

        # Pad if fewer than list_size ads
        pad_len = list_size - len(qdf)

        feat = qdf[FEATURE_NAMES].values.astype(np.float32)
        lbl  = qdf["label"].values.astype(np.float32)
        wt   = qdf["position_bias"].values.astype(np.float32)

        if pad_len > 0:
            feat = np.vstack([feat, np.zeros((pad_len, NUM_FEATURES), dtype=np.float32)])
            lbl  = np.concatenate([lbl,  np.zeros(pad_len, dtype=np.float32)])
            wt   = np.concatenate([wt,   np.zeros(pad_len, dtype=np.float32)])

        # Log-normalize query frequency
        feat[:, FEATURE_NAMES.index("query_freq_log")] = np.log1p(
            qdf["query_freq"].values[:len(qdf)]
        ) if pad_len == 0 else np.concatenate([
            np.log1p(qdf["query_freq"].values),
            np.zeros(pad_len)
        ])

        features_list.append(feat)
        labels_list.append(lbl)
        weights_list.append(wt)

    features_arr = np.stack(features_list)  # (N, list_size, num_features)
    labels_arr   = np.stack(labels_list)    # (N, list_size)
    weights_arr  = np.stack(weights_list)   # (N, list_size)

    ds = tf.data.Dataset.from_tensor_slices(
        (features_arr, labels_arr, weights_arr)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(query_ids))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
