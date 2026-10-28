# Lab 4: Data pipeline and quality

**Run (from ML_Labs root):** `python lab04/src/lab04.py`

## 1. Problem definition
The task is binary classification on the UCI Adult dataset (1994 US Census), loaded with
`fetch_openml(name='adult', version=2, as_frame=True)`. The goal is to predict whether a person's
annual income exceeds $50K.

**Inputs:** 14 features — age, workclass, fnlwgt, education, education-num, marital-status, occupation,
relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.

**Output:** income >50K (1) or <=50K (0).

**Success metrics:** F1-score for the >50K class and accuracy on a held-out 30% test set.
F1 is important because the classes are imbalanced (about 24% >50K), so accuracy alone is misleading.

**Constraints:** no data leakage — the train/test split is made before cleaning, and all statistics
(imputation values, clipping percentiles, encoder categories) are learned on the training set only;
the test set is only transformed. The pipeline must be reproducible: fixed random seed (42), saved config
with a SHA256 hash of the processed data, and a single command that runs everything.

## 2. Ingestion
The raw data is read with `fetch_openml` and is not modified; all later steps work on a copy.

- Row count: **48842**
- Column count: **15**

Log: `results/pipeline.log`.

## 3. Validation
Schema: 6 integer columns (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week)
and 9 categorical columns (including income). Ranges: age 17–90, hours-per-week 1–99,
capital-gain 0–99999, capital-loss 0–4356. Null policy: `'?'` means missing and is an allowed category.

| Check | Result |
|---|---|
| Type violations | none |
| Range violations | age 0, hours-per-week 0, capital-gain 0, capital-loss 0 |
| Category violations | none |
| Missing `'?'` | workclass 2799, occupation 2809, native-country 857 |
| Rows dropped | 0 |

All checks passed. Violations would be reported, not silently dropped.
Full report: `results/validation_report.md`.

## 4. Cleaning and transformation
- Stratified 70/30 split **before** cleaning: train = 34189 rows, test = 14653 rows.
- Exact duplicates dropped from train only: **24** rows. Test set untouched (14653 rows).
- Imputer fitted on train only: median for numeric, most frequent value for categorical.
- Outliers: numeric features clipped to the 1st/99th percentiles computed on train.
- Categorical features one-hot encoded (fitted on train).
- Fitted transformer saved with `joblib.dump` to `results/preprocessor.joblib`.

## 5. Derived features (LogisticRegression baseline)
- `capital_net = capital-gain - capital-loss`
- `hours_per_age = hours-per-week / (age - 16)`

Both features are computed row by row, so they do not use any information from the test set.

| | Accuracy | F1 |
|---|---|---|
| Without derived features | 0.7989 | 0.3860 |
| With derived features | 0.7988 | 0.3838 |

The derived features did not improve the model: accuracy and F1 changed by less than 0.003.
`capital_net` mostly duplicates information already present in capital-gain and capital-loss,
and `hours_per_age` adds little signal for a linear model. The low F1 is explained by the fact
that numeric features are not scaled, so large-valued features (e.g. fnlwgt) dominate the
linear model; scaling was not part of the required pipeline.

Metrics: `results/metrics.json`.

## 6. Label quality
- **Duplicate-conflict audit:** 5 groups of rows with identical features but conflicting income labels
  (10 rows in total). This is a small amount of label noise (0.02% of the dataset).
- **Manual check of 50 random rows** against the data dictionary (ranges, allowed categories,
  education ↔ education-num consistency): **0** rows with impossible values.

Results: `results/label_audit.json`, `results/label_audit_sample50.csv`.

## 7. Reproducibility
- SHA256 of the processed dataset:
  `b72746551ad1a1c82271227f7a805b3f3a68878ad46c30b15acafdcfad68eb82`
- The hash is stored together with the pipeline configuration in `results/config.yaml`.
- Running the pipeline twice gives the same hash.
- Data card (source, license, size, known biases, privacy concerns): `results/data_card.md`.
- Whole pipeline runs with one command: `python lab04/src/lab04.py`.