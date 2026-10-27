"""Fraud classifier independent of PPO. Explicit leakage-safe feature allowlist."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Target fraud_label is deliberately NOT a feature. Scenario, wallet IDs,
# agent roles, PPO internal decisions, scenario_type and transaction IDs are never predictors.
NUMERIC_FEATURES = [
    "amount_btc", "transaction_fee_btc", "sender_balance_before_btc",
    "receiver_balance_before_btc", "sender_wallet_age_days", "receiver_wallet_age_days",
    "sender_tx_count_before", "receiver_tx_count_before",
    "sender_unique_counterparties_before", "receiver_unique_counterparties_before",
    "sender_out_degree_before", "receiver_in_degree_before",
    "sender_total_sent_before_btc", "receiver_total_received_before_btc",
    "sender_repeat_counterparty", "seconds_since_sender_last_tx",
    "sender_tx_last_hour", "sender_tx_last_24h",
]
MODEL_FEATURES = NUMERIC_FEATURES + ["hour_sin", "hour_cos", "is_weekend"]


def make_model_frame(df: pd.DataFrame):
    data = df.copy()
    dt = pd.to_datetime(data["timestamp"], errors="raise", utc=True)
    fractional_hour = dt.dt.hour + dt.dt.minute / 60 + dt.dt.second / 3600
    data["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24)
    data["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    X = data[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    y = data["fraud_label"].astype(int)
    return X, y


def make_classifier(seed=42):
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=130, min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed))


def _metric(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return {
        "samples": int(len(y_true)),
        "fraud_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)) if len(set(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def evaluate_runs(runs: dict[str, pd.DataFrame]):
    """Independent run split: first=train, second=validation, third=test.

    Threshold is chosen ONLY on validation labels; test labels are used solely
    after the model is fixed. Return evidence, not a realism guarantee.
    """
    if len(runs) < 3:
        raise ValueError("At least 3 independent runs are needed for held-out-run evaluation")
    names = list(runs)[:3]
    xtr, ytr = make_model_frame(runs[names[0]])
    xva, yva = make_model_frame(runs[names[1]])
    xte, yte = make_model_frame(runs[names[2]])
    if min(ytr.nunique(), yva.nunique(), yte.nunique()) < 2:
        raise ValueError("Training, validation and test each require both fraud and genuine samples")
    model = make_classifier()
    model.fit(xtr, ytr)
    pva = model.predict_proba(xva)[:, 1]
    candidates = np.arange(0.05, 0.96, 0.05)
    threshold = float(max(candidates, key=lambda t: (f1_score(yva, pva >= t, zero_division=0), -abs(t - .5))))
    pte = model.predict_proba(xte)[:, 1]
    importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": model[-1].feature_importances_})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    return model, {
        "training_run": names[0], "validation_run": names[1], "testing_run": names[2],
        "selected_threshold_on_validation": threshold,
        "validation": _metric(yva, pva, threshold),
        "test": _metric(yte, pte, threshold),
        "note": "Evaluated on held-out synthetic runs, not real police/blockchain data.",
    }, importance


def evaluate_temporal(df: pd.DataFrame):
    """Demo fallback for one dataset. Temporal separation, not independent runs."""
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(df) < 90:
        raise ValueError("Need at least 90 rows for a temporal split")
    n = len(df)
    ix1, ix2 = int(n*.7), int(n*.85)
    chunks = [df.iloc[:ix1], df.iloc[ix1:ix2], df.iloc[ix2:]]
    if any(c.fraud_label.nunique() < 2 for c in chunks):
        raise ValueError("Each chronological partition must contain both fraud and genuine labels")
    return evaluate_runs(dict(zip(["chronological_train", "chronological_val", "chronological_test"], chunks)))
