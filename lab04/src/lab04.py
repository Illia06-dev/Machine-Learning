import hashlib
import json
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data" / "lab04"
RESULTS = ROOT / "results"
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "dataset": "fetch_openml(name='adult', version=2, as_frame=True)",
    "seed": 42,
    "test_size": 0.3,
    "imputer": {"numeric": "median", "categorical": "most_frequent"},
    "clip_percentiles": [1, 99],
    "derived_features": ["capital_net", "hours_per_age"],
    "model": "LogisticRegression(solver='liblinear', max_iter=1000)",
    "label_audit_sample_size": 50,
}
SEED = CONFIG["seed"]

SCHEMA_TYPES = {
    "age": "int", "workclass": "category", "fnlwgt": "int", "education": "category",
    "education-num": "int", "marital-status": "category", "occupation": "category",
    "relationship": "category", "race": "category", "sex": "category",
    "capital-gain": "int", "capital-loss": "int", "hours-per-week": "int",
    "native-country": "category", "income": "category",
}
NUM_COLS = [c for c, t in SCHEMA_TYPES.items() if t == "int"]
CAT_COLS = [c for c, t in SCHEMA_TYPES.items() if t == "category"]
CAT_FEATURES = [c for c in CAT_COLS if c != "income"]
FEATURES = [c for c in SCHEMA_TYPES if c != "income"]
DERIVED = ["capital_net", "hours_per_age"]

RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

EDU_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
    "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
    "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}
ALLOWED_CATS = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"],
    "education": list(EDU_NUM),
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
                       "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                       "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                       "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                       "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
                       "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
}
for _c in ALLOWED_CATS:
    ALLOWED_CATS[_c] = ALLOWED_CATS[_c] + ["?"]
ALLOWED_CATS["income"] = [">50K", "<=50K"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(RESULTS / "pipeline.log", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lab04")


def ingest() -> pd.DataFrame:
    raw = fetch_openml(name="adult", version=2, as_frame=True,
                       data_home=str(DATA), parser="auto").frame
    log.info("[2] Ingestion: row count = %d, column count = %d", *raw.shape)
    return raw


def to_schema_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().rename(columns={"class": "income"})
    for c in CAT_COLS:
        df[c] = df[c].astype(object).where(df[c].notna(), "?")
    return df


def validate(df: pd.DataFrame) -> dict:
    rep = {"type_violations": {}, "range_violations": {}, "category_violations": {},
           "missing_qmark": {}, "rows_dropped": 0}
    for col, typ in SCHEMA_TYPES.items():
        s = df[col]
        if typ == "int":
            ok = pd.api.types.is_numeric_dtype(s) and bool(((s.dropna() % 1) == 0).all())
        else:
            ok = s.dtype == object
        if not ok:
            rep["type_violations"][col] = str(s.dtype)
    for col, (lo, hi) in RANGES.items():
        rep["range_violations"][col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    for col, allowed in ALLOWED_CATS.items():
        mask = ~df[col].isin(allowed)
        rep["category_violations"][col] = {
            "count": int(mask.sum()),
            "values": sorted(map(str, df.loc[mask, col].unique())),
        }
    for col in CAT_COLS:
        rep["missing_qmark"][col] = int((df[col] == "?").sum())

    L = ["# Validation Report", "",
         f"Rows: {len(df)}, columns: {df.shape[1]}. Rows dropped: 0.", "",
         "## Types"]
    L += [f"- {c}: {t}" for c, t in rep["type_violations"].items()] or ["- No violations"]
    L += ["", "## Ranges", "", "| column | allowed | violations |", "|---|---|---|"]
    L += [f"| {c} | {RANGES[c][0]}–{RANGES[c][1]} | {n} |" for c, n in rep["range_violations"].items()]
    L += ["", "## Categories ('?' allowed)", "", "| column | violations | values |", "|---|---|---|"]
    L += [f"| {c} | {v['count']} | {', '.join(v['values']) or '-'} |"
          for c, v in rep["category_violations"].items()]
    L += ["", "## Missing ('?')", "", "| column | count |", "|---|---|"]
    L += [f"| {c} | {n} |" for c, n in rep["missing_qmark"].items()]
    (RESULTS / "validation_report.md").write_text("\n".join(L), encoding="utf-8")

    log.info("[3] Type violations: %s", rep["type_violations"] or "none")
    log.info("[3] Range violations: %s", rep["range_violations"])
    log.info("[3] Category violations: %s",
             {c: v["count"] for c, v in rep["category_violations"].items() if v["count"]} or "none")
    log.info("[3] Missing '?': %s", {c: n for c, n in rep["missing_qmark"].items() if n})
    return rep


class QMarkToNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in CAT_FEATURES:
            X[c] = X[c].where(X[c] != "?", np.nan)
        return X


class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.enabled:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
        return X


class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.percentile(X, self.lower, axis=0)
        self.hi_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(with_features: bool) -> Pipeline:
    num_cols = NUM_COLS + (DERIVED if with_features else [])
    lo, hi = CONFIG["clip_percentiles"]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clip", PercentileClipper(lo, hi))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, CAT_FEATURES)])
    return Pipeline([("qmark_to_nan", QMarkToNaN()),
                     ("features", FeatureAdder(with_features)),
                     ("columns", ct)])


def split_and_clean(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["income"].str.startswith(">50K").astype(int).rename("target")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=CONFIG["test_size"], stratify=y, random_state=SEED)
    log.info("[4] Stratified split before cleaning: train=%d, test=%d", len(X_tr), len(X_te))

    train = pd.concat([X_tr, y_tr], axis=1)
    n_before = len(train)
    train = train.drop_duplicates()
    log.info("[4] Dropped %d exact duplicates from train; test untouched (%d rows)",
             n_before - len(train), len(X_te))
    return train[FEATURES], X_te, train["target"], y_te


def run_experiments(X_tr, X_te, y_tr, y_te):
    metrics, fitted = {}, {}
    for name, flag in [("without_derived", False), ("with_derived", True)]:
        pre = build_preprocessor(flag)
        model = Pipeline([("pre", pre),
                          ("clf", LogisticRegression(solver="liblinear", max_iter=1000))])
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        metrics[name] = {"accuracy": round(float(accuracy_score(y_te, pred)), 4),
                         "f1": round(float(f1_score(y_te, pred)), 4)}
        fitted[name] = pre
        log.info("[5] %s: accuracy=%.4f, f1=%.4f", name, metrics[name]["accuracy"], metrics[name]["f1"])
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(fitted["with_derived"], RESULTS / "preprocessor.joblib")
    log.info("[4] Saved results/preprocessor.joblib")
    return metrics, fitted["with_derived"]


def row_issues(r) -> str:
    issues = []
    for col, (lo, hi) in RANGES.items():
        if not lo <= r[col] <= hi:
            issues.append(f"{col}={r[col]} out of range")
    for col, allowed in ALLOWED_CATS.items():
        if r[col] not in allowed:
            issues.append(f"{col}='{r[col]}' not in dictionary")
    if EDU_NUM.get(r["education"]) != r["education-num"]:
        issues.append(f"education-num={r['education-num']} does not match {r['education']}")
    return "; ".join(issues)


def label_audit(df: pd.DataFrame) -> dict:
    n_labels = df.groupby(FEATURES)["income"].transform("nunique")
    conflict = df[n_labels > 1]
    n_groups = conflict.groupby(FEATURES).ngroups

    sample = df.sample(n=CONFIG["label_audit_sample_size"], random_state=SEED).copy()
    sample["issues"] = sample.apply(row_issues, axis=1)
    sample["manual_verdict"] = ""
    sample.to_csv(RESULTS / "label_audit_sample50.csv")

    audit = {"conflicting_duplicate_groups": int(n_groups),
             "rows_in_conflicting_groups": int(len(conflict)),
             "sample_rows_with_impossible_values": int((sample["issues"] != "").sum())}
    (RESULTS / "label_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log.info("[6] Duplicate-conflict groups: %d (%d rows)", n_groups, len(conflict))
    log.info("[6] 50-row check: %d rows with impossible values", audit["sample_rows_with_impossible_values"])
    return audit


def save_processed_and_hash(pre, X_tr, X_te, y_tr, y_te) -> dict:
    names = pre.named_steps["columns"].get_feature_names_out()
    tr = pd.DataFrame(pre.transform(X_tr), columns=names).assign(target=y_tr.to_numpy())
    te = pd.DataFrame(pre.transform(X_te), columns=names).assign(target=y_te.to_numpy())
    p_tr, p_te = DATA / "train_processed.csv", DATA / "test_processed.csv"
    tr.to_csv(p_tr, index=False, float_format="%.10g", lineterminator="\n")
    te.to_csv(p_te, index=False, float_format="%.10g", lineterminator="\n")

    h = hashlib.sha256()
    for p in (p_tr, p_te):
        h.update(p.read_bytes())
    digest = h.hexdigest()

    with open(RESULTS / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**CONFIG, "processed_sha256": digest}, f, sort_keys=False)
    log.info("[7] Processed dataset SHA256 = %s (saved in results/config.yaml)", digest)
    return {"sha256": digest, "train_shape": tr.shape, "test_shape": te.shape}


def write_data_card(df: pd.DataFrame, info: dict):
    pos = df["income"].str.startswith(">50K").mean()
    sex = df["sex"].value_counts(normalize=True)
    race = df["race"].value_counts(normalize=True)
    us = (df["native-country"] == "United-States").mean()

    card = f"""# Data Card — UCI Adult

## Source
UCI Machine Learning Repository, Adult dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census.
Loaded via `fetch_openml(name='adult', version=2, as_frame=True)`.

## License
CC BY 4.0 (UCI Machine Learning Repository).

## Size
- Raw: {df.shape[0]} rows, {df.shape[1]} columns (14 features + income).
- Processed train: {info['train_shape'][0]} rows, {info['train_shape'][1]} columns.
- Processed test: {info['test_shape'][0]} rows, {info['test_shape'][1]} columns.
- Processed SHA256: `{info['sha256']}`

## Known biases
- Data from 1994; the $50K threshold is not adjusted for inflation.
- Class imbalance: {pos:.1%} of rows are `>50K`.
- Sex imbalance: {', '.join(f'{k} {v:.1%}' for k, v in sex.items())}.
- Race imbalance: {', '.join(f'{k} {v:.1%}' for k, v in race.items())}.
- {us:.1%} of rows are from the United States; other countries are under-represented.
- Missing values ('?') are concentrated in workclass, occupation and native-country.

## Privacy concerns
- Census microdata about real people; contains sensitive attributes (race, sex, marital status, country of origin, income).
- No names or IDs, but combinations of age, sex, race, country and occupation can re-identify rare individuals.
"""
    (RESULTS / "data_card.md").write_text(card, encoding="utf-8")
    log.info("[7] Saved results/data_card.md")


def main():
    raw = ingest()
    df = to_schema_frame(raw)
    validate(df)
    X_tr, X_te, y_tr, y_te = split_and_clean(df)
    _, pre = run_experiments(X_tr, X_te, y_tr, y_te)
    label_audit(df)
    info = save_processed_and_hash(pre, X_tr, X_te, y_tr, y_te)
    write_data_card(df, info)
    log.info("DONE")


if __name__ == "__main__":
    main()