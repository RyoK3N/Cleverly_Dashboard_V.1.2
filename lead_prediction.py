from datetime import datetime
import json
import os
import cloudpickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import argparse
import logging

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Data Preprocessor
class DataPreprocessor:
    UNNEEDED = {
        "Follow Up Tracker", "Notes", "Email Template #1", "Email Template #2",
        "Send Panda Doc?", "UTM Source", "UTM Campaign", "UTM Content",
        "Lead Source", "Channel FOR FUNNEL METRICS", "Subitems",
        "text_mkr7frmd", "Item ID", "Phone", "Last updated", "Sales Call Date",
        "Deal Status Date", "Date Created", "Item Name"
    }
    HIGH_NULL = 0.50

    def load_and_clean(self, won_csv: str, lost_csv: str) -> pd.DataFrame:
        df = pd.concat(
            [pd.read_csv(won_csv), pd.read_csv(lost_csv)], ignore_index=True)
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        hi_null_cols = df.columns[df.isna().mean() > self.HIGH_NULL]
        df = df.drop(columns=hi_null_cols, errors="ignore")
        df = df[df["Owner"].notna()]
        df = df.drop(columns=[c for c in self.UNNEEDED if c in df.columns],
                     errors="ignore")
        return df

    def split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df[~df["Deal Status"].isin(["Scheduled", "Re-Schedule"])]
        return df.drop(columns=["Deal Status"]), df["Deal Status"]


# Model Class
class LeadStatusModel:

    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.pipeline: Pipeline | None = None
        self.metadata: dict = {}

    def _build_pipeline(self, X: pd.DataFrame) -> None:
        num_cols = X.select_dtypes(
            include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(
            include=["object", "category", "bool"]).columns.tolist()

        pre = ColumnTransformer([
            ("num",
             Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())]), num_cols),
            ("cat",
             Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("oh",
                        OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False))]), cat_cols)
        ])

        self.pipeline = Pipeline([
            ("preprocessor", pre),
            ("clf", HistGradientBoostingClassifier(random_state=42))
        ])

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self._build_pipeline(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X,
                                                  y,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=y)
        self.pipeline.fit(X_tr, y_tr)
        y_pred = self.pipeline.predict(X_te)
        y_proba = self.pipeline.predict_proba(X_te)[:, 1]

        pre = self.pipeline.named_steps["preprocessor"]
        self.metadata = {
            "trained_at": datetime.now().isoformat(),
            "metrics": {
                "accuracy":
                float(accuracy_score(y_te, y_pred)),
                "roc_auc":
                float(roc_auc_score(y_te, y_proba)),
                "classification_report":
                classification_report(y_te, y_pred, output_dict=True)
            },
            "raw_numeric_cols": pre.transformers_[0][2],
            "raw_categorical_cols": pre.transformers_[1][2],
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        }
        return self.metadata["metrics"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def save(self) -> Tuple[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        m_path = os.path.join(self.model_dir, f"model_{ts}.pkl")
        j_path = os.path.join(self.model_dir, f"metadata_{ts}.json")
        with open(m_path, 'wb') as f:
            cloudpickle.dump(self.pipeline, f)
        with open(j_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        return m_path, j_path

    def load(self, model_path: str, meta_path: str) -> None:
        with open(model_path, 'rb') as f:
            self.pipeline = cloudpickle.load(f)
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)


# CLI
def cmd_train(ns: argparse.Namespace) -> None:
    prep, model = DataPreprocessor(), LeadStatusModel(ns.model_dir)
    logger.info("🔄 Loading data …")
    df = prep.load_and_clean(ns.won, ns.lost)
    X, y = prep.split_xy(df)
    logger.info("⚙️  Training …")
    metrics = model.train(X, y)
    m_path, j_path = model.save()
    logger.info("✅ Model  → %s", m_path)
    logger.info("✅ Meta   → %s", j_path)
    logger.info("🏁 Accuracy = %.4f | AUC = %.4f", metrics["accuracy"],
                metrics["roc_auc"])


def cmd_predict(ns: argparse.Namespace) -> None:
    prep, model = DataPreprocessor(), LeadStatusModel()
    model.load(ns.model, ns.meta)
    logger.info("🔄 Reading inference CSV …")
    df_raw = pd.read_csv(ns.input)
    X = prep._clean(df_raw.copy())
    want: List[str] = model.metadata["raw_numeric_cols"] + model.metadata[
        "raw_categorical_cols"]
    for col in want:
        if col not in X.columns:
            X[col] = np.nan
    X = X.reindex(columns=want)
    if X.empty:
        logger.error("No rows left after cleaning; nothing to predict.")
        df_raw["prediction"] = np.nan
        df_raw["prediction_confidence"] = np.nan
        df_raw.to_csv(ns.output, index=False)
        return
    logger.info("🔮 Scoring …")
    preds = model.predict(X)
    conf = model.predict_proba(X).max(axis=1)
    df_out = df_raw.copy()
    df_out["prediction"] = np.nan
    df_out["prediction_confidence"] = np.nan
    df_out.loc[X.index, "prediction"] = preds
    df_out.loc[X.index, "prediction_confidence"] = conf
    df_out.to_csv(ns.output, index=False)
    logger.info("✅ Predictions → %s", ns.output)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lead win/loss prediction CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train new model")
    t.add_argument("--won", required=True, help="CSV of WON deals")
    t.add_argument("--lost", required=True, help="CSV of LOST deals")
    t.add_argument("--model-dir", default="models", help="save directory")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="run inference")
    pr.add_argument("--model", required=True, help="*.pkl path")
    pr.add_argument("--meta", required=True, help="metadata JSON path")
    pr.add_argument("--input", required=True, help="CSV of new leads")
    pr.add_argument("--output",
                    required=True,
                    help="output CSV with predictions")
    pr.set_defaults(func=cmd_predict)
    return p


if __name__ == "__main__":
    ns = build_parser().parse_args()
    ns.func(ns)
