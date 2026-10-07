# Lab 1: Environment and First System Measurements

## 1. Goal
The goal of this laboratory work is to set up a reproducible isolated Python environment with pinned dependencies, train baseline classification models (Logistic Regression and Random Forest) on the Breast Cancer Wisconsin dataset using a stratified split, and measure their system costs (training time, inference latency, model storage footprint, and peak RAM consumption). Finally, the models are evaluated against standard hardware deployment budgets (Cloud, Edge, Mobile, TinyML).

## 2. Method
- **Environment & Reproducibility:** An isolated virtual environment was set up with Python 3.11.8. All random seeds were globally pinned to 42 (`random`, `numpy`, `torch`).
- **Dataset:** The Breast Cancer Wisconsin dataset was loaded via `sklearn.datasets.load_breast_cancer(return_X_y=True)`. The dataset was split into training and test sets in a 70/30 stratified ratio (`random_state=42`).
- **Models:**
  - Linear baseline: `LogisticRegression(max_iter=1000, random_state=42)`
  - Ensemble baseline: `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Measurement Protocol:**
  - *Training time:* Median of 5 runs after 1 warm-up run.
  - *Inference latency:* Median of 100 single-sample inferences after 1 warm-up run.
  - *Model size:* Serialized using `joblib.dump` and measured on disk in bytes and KB.
  - *Peak memory:* Measured via process Resident Set Size (`psutil.Process().memory_info().rss`) during training and inference.

## 3. Results

### Measured System Metrics
| Model | Test Accuracy | Train Time (s) | Single-Sample Latency (ms) | Model Size (bytes) | Model Size (KB) | Peak RAM Train (MB) | Peak RAM Infer (MB) |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.9415 | 0.2131 | 0.0491 | 1,055 | 1.03 | 269.06 | 269.09 |
| **Random Forest** | 0.9357 | 0.1057 | 1.9867 | 290,889 | 284.07 | 269.96 | 270.75 |

### Deployment Budget Compatibility
*Hardware Budget Constraints:*
- **Cloud:** Memory $\ge$ 1 GB, Latency $\le$ 100 ms, Size $\le$ 500 MB
- **Edge:** Memory 256–1024 MB, Latency $\le$ 50 ms, Size $\le$ 50 MB
- **Mobile:** Memory 64–256 MB, Latency $\le$ 20 ms, Size $\le$ 10 MB
- **TinyML:** Memory $\le$ 256 KB, Latency $\le$ 10 ms, Size $\le$ 100 KB

| Model | Cloud | Edge | Mobile | TinyML | Justification |
|---|---|---|---|---|---|
| **Logistic Regression** | Fit | Fit | Partial / Fit* | Exceeded (RAM) | Fits comfortably in Cloud and Edge budgets. Meets Mobile latency (0.05 ms $\le$ 20 ms) and size (1.03 KB $\le$ 10 MB), though standard Python runtime RSS baseline exceeds pure mobile sandbox without lightweight C runtime compilation. In TinyML, storage fits (1.03 KB $\le$ 100 KB), but process RAM exceeds the 256 KB limit. |
| **Random Forest** | Fit | Fit | Partial / Fit* | Exceeded (Size & RAM) | Fits Cloud and Edge budgets with ease. Meets Mobile latency (1.99 ms $\le$ 20 ms) and storage (0.28 MB $\le$ 10 MB). Exceeds TinyML memory ($\le$ 256 KB) and storage ($\le$ 100 KB vs 284 KB). |

*\*Note: In mobile and embedded environments, raw Python runtime overhead dominates baseline RAM; compiled C/ONNX runtimes easily satisfy the sub-10 MB mobile budget for both models.*

## 4. Conclusions
1. Logistic Regression achieved slightly higher classification accuracy (0.9415 vs 0.9357) on the Breast Cancer dataset while having an inference latency approximately 40 times lower than Random Forest (0.049 ms vs 1.987 ms).
2. The serialized storage footprint differs by over two orders of magnitude: Logistic Regression requires only ~1 KB of storage, whereas the 100-tree ensemble of Random Forest requires ~284 KB.
3. Both models easily satisfy latency and size requirements for Cloud, Edge, and Mobile tiers. However, neither model can be deployed out-of-the-box onto TinyML targets in standard Python due to system RAM overhead, and Random Forest additionally exceeds the TinyML model storage cap (284 KB > 100 KB).