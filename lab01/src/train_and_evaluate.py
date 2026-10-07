import os
import time
import random
import numpy as np
import pandas as pd
import psutil
import joblib
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
    
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_model(model_name, model_cls, model_kwargs, X_train, y_train, X_test, y_test, results_dir):
    warmup_model = model_cls(**model_kwargs)
    warmup_model.fit(X_train, y_train)

    train_times = []
    mem_before = get_process_memory_mb()
    for _ in range(5):
        m = model_cls(**model_kwargs)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        t1 = time.perf_counter()
        train_times.append(t1 - t0)
    median_train_sec = float(np.median(train_times))
    mem_after = get_process_memory_mb()
    peak_train_mem_mb = max(mem_after, mem_before)

    final_model = model_cls(**model_kwargs)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    test_accuracy = round(accuracy_score(y_test, y_pred), 4)

    model_filename = f"{model_name.lower().replace(' ', '_')}.joblib"
    model_path = os.path.join(results_dir, model_filename)
    joblib.dump(final_model, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    single_sample = X_test[0:1]
    final_model.predict(single_sample)

    infer_times_ms = []
    mem_infer_before = get_process_memory_mb()
    for _ in range(100):
        t0 = time.perf_counter()
        final_model.predict(single_sample)
        t1 = time.perf_counter()
        infer_times_ms.append((t1 - t0) * 1000.0)
    median_infer_ms = float(np.median(infer_times_ms))
    mem_infer_after = get_process_memory_mb()
    peak_infer_mem_mb = max(mem_infer_after, mem_infer_before)

    return {
        "model": model_name,
        "test_accuracy": test_accuracy,
        "train_time_sec": median_train_sec,
        "latency_ms": median_infer_ms,
        "size_bytes": size_bytes,
        "size_kb": size_kb,
        "train_mem_mb": peak_train_mem_mb,
        "infer_mem_mb": peak_infer_mem_mb,
    }

def main():
    results_dir = os.path.join("lab01", "results")
    os.makedirs(results_dir, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )

    lr_data = benchmark_model(
        "Logistic Regression",
        LogisticRegression,
        {"max_iter": 1000, "random_state": SEED},
        X_train, y_train, X_test, y_test, results_dir
    )

    rf_data = benchmark_model(
        "Random Forest",
        RandomForestClassifier,
        {"n_estimators": 100, "random_state": SEED},
        X_train, y_train, X_test, y_test, results_dir
    )

    all_results = [lr_data, rf_data]

    acc_df = pd.DataFrame([
        {"model": r["model"], "accuracy": f"{r['test_accuracy']:.4f}"}
        for r in all_results
    ])
    acc_df.to_csv(os.path.join(results_dir, "baseline_accuracy.csv"), index=False)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(results_dir, "system_metrics.csv"), index=False)

    print("\n================== Results ==================")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()