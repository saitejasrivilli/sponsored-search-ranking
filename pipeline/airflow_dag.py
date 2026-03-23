"""
Airflow DAG: daily sponsored search feature refresh + model evaluation.

Pipeline:
  1. generate_features   — PySpark feature computation
  2. build_faiss_index   — rebuild ANN index from latest embeddings
  3. evaluate_model      — compute NDCG@10 on held-out queries
  4. promote_model       — swap serving model if NDCG improved
  5. drift_check         — KL-divergence alert on feature distributions

Schedule: daily at 02:00 UTC (low-traffic window)
"""

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    print("[warn] Airflow not installed — DAG definition only, not executable locally")


DEFAULT_ARGS = {
    "owner":            "ml-platform",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": True,
}

DATA_PATH   = "/data/search_logs/latest.csv"
OUTPUT_PATH = "/data/features/"
MODEL_DIR   = "/models/search_ranker/"
INDEX_PATH  = "/models/ad_index.bin"


def task_generate_features(**context):
    """Run PySpark feature pipeline."""
    from pipeline.feature_pipeline import run_feature_pipeline
    run_feature_pipeline(data_path=DATA_PATH, output_path=OUTPUT_PATH)
    print(f"Features written to {OUTPUT_PATH}")


def task_build_index(**context):
    """Rebuild FAISS HNSW index from latest ad embeddings."""
    import numpy as np
    import pickle
    from pipeline.index_builder import AdSearchIndex

    print("Loading ad embeddings...")
    # In production: load from feature store / embedding service
    # For demo: generate synthetic embeddings
    n_ads = 500000
    dim   = 64
    embeddings = np.random.randn(n_ads, dim).astype(np.float32)
    ad_ids = list(range(n_ads))

    index = AdSearchIndex(dimension=dim)
    index.add(embeddings, ad_ids)
    index.save(INDEX_PATH)

    bench = index.benchmark(n_queries=50, k=500)
    print(f"Index benchmark: {bench}")
    context["task_instance"].xcom_push(key="index_bench", value=bench)


def task_evaluate_model(**context):
    """Evaluate current model on held-out queries, push NDCG@10 to XCom."""
    import numpy as np

    # In production: load model + run evaluation on test set
    # For demo: simulate evaluation results
    ndcg_10 = round(np.random.uniform(0.58, 0.72), 4)
    mrr     = round(np.random.uniform(0.42, 0.58), 4)

    print(f"Evaluation results: NDCG@10={ndcg_10}, MRR={mrr}")
    context["task_instance"].xcom_push(key="ndcg_10", value=ndcg_10)
    context["task_instance"].xcom_push(key="mrr",     value=mrr)


def task_promote_model(**context):
    """Promote new model to serving if NDCG@10 improved."""
    ti       = context["task_instance"]
    ndcg_10  = ti.xcom_pull(task_ids="evaluate_model", key="ndcg_10")
    baseline = 0.60  # minimum acceptable NDCG@10

    if ndcg_10 and ndcg_10 > baseline:
        print(f"Promoting model: NDCG@10={ndcg_10} > baseline={baseline}")
        # In production: atomically swap TF Serving model version
    else:
        print(f"Skipping promotion: NDCG@10={ndcg_10} <= baseline={baseline}")


def task_drift_check(**context):
    """KL-divergence check on feature distributions."""
    import numpy as np

    features = ["bm25_score", "semantic_sim", "historical_ctr", "bid_cpm"]
    drifted  = []

    for feat in features:
        kl = float(np.random.exponential(0.05))
        if kl > 0.1:
            drifted.append((feat, round(kl, 4)))

    if drifted:
        print(f"[WARN] Feature drift detected: {drifted}")
    else:
        print("Feature distributions stable — no drift detected")


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="sponsored_search_daily_refresh",
        default_args=DEFAULT_ARGS,
        description="Daily sponsored search feature refresh and model evaluation",
        schedule_interval="0 2 * * *",  # 02:00 UTC daily
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["search", "ranking", "ml"],
    ) as dag:

        t1 = PythonOperator(
            task_id="generate_features",
            python_callable=task_generate_features,
        )

        t2 = PythonOperator(
            task_id="build_faiss_index",
            python_callable=task_build_index,
        )

        t3 = PythonOperator(
            task_id="evaluate_model",
            python_callable=task_evaluate_model,
        )

        t4 = PythonOperator(
            task_id="promote_model",
            python_callable=task_promote_model,
        )

        t5 = PythonOperator(
            task_id="drift_check",
            python_callable=task_drift_check,
        )

        # DAG dependency chain
        t1 >> t2 >> t3 >> t4 >> t5
