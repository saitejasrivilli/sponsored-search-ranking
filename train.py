"""
End-to-end training script for sponsored search ranking.
Run locally with Python 3.11 + TF, or on Google Colab via train_colab.ipynb.

Usage:
    python train.py --n_queries 5000 --epochs 20 --batch_size 256
"""

import argparse
import os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Train sponsored search ranker")
    p.add_argument("--n_queries",   type=int,   default=5000)
    p.add_argument("--ads_per_q",   type=int,   default=10)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden",      type=int,   nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--model_dir",   type=str,   default="models/")
    p.add_argument("--data_path",   type=str,   default="data/synthetic_search_data.csv")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("SPONSORED SEARCH RANKER — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Generate data ──────────────────────────────────────
    print("\n[1/5] Generating synthetic dataset...")
    from data.synthetic_data import generate_dataset, train_val_test_split
    df = generate_dataset(n_queries=args.n_queries, ads_per_query=args.ads_per_q)
    train_df, val_df, test_df = train_val_test_split(df)
    df.to_csv(args.data_path, index=False)
    print(f"Saved to {args.data_path}")

    # ── Step 2: Build tf.data datasets ────────────────────────────
    print("\n[2/5] Building TF datasets...")
    import tensorflow as tf
    from model.ranking_model import SponsoredSearchRanker, build_dataset, FEATURE_NAMES

    train_ds = build_dataset(train_df, batch_size=args.batch_size, shuffle=True)
    val_ds   = build_dataset(val_df,   batch_size=args.batch_size, shuffle=False)
    test_ds  = build_dataset(test_df,  batch_size=args.batch_size, shuffle=False)
    print(f"Train batches: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Step 3: Build model ────────────────────────────────────────
    print("\n[3/5] Building model...")
    model = SponsoredSearchRanker(
        hidden_units=tuple(args.hidden),
        dropout_rate=args.dropout,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=args.lr,
            first_decay_steps=len(train_ds) * 5,
        )
    )
    model.compile(optimizer=optimizer)

    # Warm up
    dummy = tf.zeros((1, 10, len(FEATURE_NAMES)))
    model(dummy)
    print(f"Parameters: {model.count_params():,}")

    # ── Step 4: Train ──────────────────────────────────────────────
    print(f"\n[4/5] Training for {args.epochs} epochs...")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, "ranker_best.weights.h5"),
            monitor="val_ndcg@10",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ndcg@10",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.model_dir, "training_log.csv")
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 5: Evaluate ───────────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    results = model.evaluate(test_ds, verbose=1)
    metric_names = ["loss", "ndcg@1", "ndcg@5", "ndcg@10", "mrr"]
    print("\nTest results:")
    for name, val in zip(metric_names, results):
        print(f"  {name}: {val:.4f}")

    # Save final weights
    model.save_weights(os.path.join(args.model_dir, "ranker_weights"))
    print(f"\nWeights saved to {args.model_dir}/ranker_weights")

    # Build FAISS index from ad embeddings
    print("\nBuilding FAISS index...")
    from pipeline.index_builder import AdSearchIndex
    import pandas as pd

    # Use test set ad features as index
    from model.ranking_model import FEATURE_NAMES, NUM_FEATURES
    ad_features = test_df[FEATURE_NAMES].values.astype(np.float32)
    ad_ids = test_df["ad_id"].values.tolist()

    # Expand to (n, 1, num_features) then flatten for tower
    flat_features = tf.constant(ad_features)
    embeddings = model.tower(flat_features, training=False).numpy()

    index = AdSearchIndex(dimension=embeddings.shape[1])
    index.add(embeddings, ad_ids)
    index_path = os.path.join(args.model_dir, "ad_index.bin")
    index.save(index_path)

    bench = index.benchmark(n_queries=50, k=50)
    print(f"\nFAISS benchmark: {bench}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best NDCG@10: {max(history.history.get('val_ndcg@10', [0])):.4f}")
    print(f"Models saved to: {args.model_dir}")
    print("\nTo serve: python serving/api.py")


if __name__ == "__main__":
    main()
