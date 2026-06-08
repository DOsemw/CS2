"""
model.py
--------
Trains and evaluates LightGBM regression models for CS2 player stats.

Targets: kills, deaths, hs_pct

Uses random 70/15/15 split since data is aggregate (one row per player),
not time-series like the LoL model.
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

TARGETS = ["kills", "deaths", "hs_pct"]
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 300,
    "early_stopping_rounds": 30,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# ── Data splitting ────────────────────────────────────────────────────────────

def time_split(df, feature_cols, target, test_cutoff=None, val_cutoff=None):
    """Random 70/15/15 split for aggregate (non time-series) data."""
    df = df.dropna(subset=[target] + feature_cols).copy()

    if len(df) < 10:
        raise ValueError(f"Not enough data for target '{target}': only {len(df)} rows")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"  [{target}] Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return (train[feature_cols], train[target],
            val[feature_cols],   val[target],
            test[feature_cols],  test[target])


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, params=None, target_name="stat"):
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(p["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    print(f"  [{target_name}] Best iteration: {model.best_iteration_}")
    return model


def evaluate(model, X_test, y_test, target_name):
    preds = np.clip(model.predict(X_test), 0, None)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    within_half = np.mean(np.abs(preds - y_test) <= 0.5)
    within_one  = np.mean(np.abs(preds - y_test) <= 1.0)
    metrics = {
        "target": target_name,
        "n_test": len(y_test),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "within_0.5": round(within_half, 4),
        "within_1.0": round(within_one, 4),
        "mean_actual": round(float(y_test.mean()), 4),
        "mean_pred": round(float(preds.mean()), 4),
    }
    return metrics, preds


def print_metrics(m):
    print(f"\n  ── {m['target'].upper()} ──")
    print(f"  MAE:         {m['mae']:.3f}")
    print(f"  RMSE:        {m['rmse']:.3f}")
    print(f"  Within ±0.5: {m['within_0.5']*100:.1f}%")
    print(f"  Within ±1.0: {m['within_1.0']*100:.1f}%")
    print(f"  Mean actual: {m['mean_actual']:.2f}  Mean pred: {m['mean_pred']:.2f}")


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, target, metrics, feature_cols):
    path = MODEL_DIR / f"{target}_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "metrics": metrics}, f)
    with open(MODEL_DIR / f"{target}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [save] {path}")


def load_model(target):
    with open(MODEL_DIR / f"{target}_model.pkl", "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_cols"], payload["metrics"]


# ── Full training run ─────────────────────────────────────────────────────────

def train_all(df, feature_cols, test_cutoff=None, val_cutoff=None, shap_n=50):
    all_metrics = {}

    for target in TARGETS:
        print(f"\n{'='*50}\n  Training: {target.upper()}\n{'='*50}")

        # Skip if target is all NaN
        if df[target].isna().all():
            print(f"  [{target}] Skipping — all values are NaN")
            continue

        try:
            X_tr, y_tr, X_val, y_val, X_te, y_te = time_split(
                df, feature_cols, target, test_cutoff, val_cutoff
            )
        except ValueError as e:
            print(f"  [{target}] Skipping — {e}")
            continue

        model = train_model(X_tr, y_tr, X_val, y_val, target_name=target)
        metrics, _ = evaluate(model, X_te, y_te, target)
        print_metrics(metrics)

        # Feature importance
        imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print(f"\n  Top 10 features:\n{imp.head(10).to_string(index=False)}")

        # SHAP
        try:
            sample = X_te.sample(min(shap_n, len(X_te)), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(sample)
            shap_df = pd.DataFrame({
                "feature": sample.columns,
                "mean_|shap|": np.abs(shap_vals).mean(axis=0),
            }).sort_values("mean_|shap|", ascending=False)
            print(f"\n  SHAP top features:\n{shap_df.head(10).to_string(index=False)}")
        except Exception as e:
            print(f"  [shap] Skipped: {e}")

        save_model(model, target, metrics, feature_cols)
        all_metrics[target] = metrics

    return all_metrics


if __name__ == "__main__":
    from data_ingestion import load_raw, filter_top_teams
    from feature_engineering import build_features, get_feature_columns

    print("=== CS2 Model Training ===\n")
    raw = load_raw()
    raw = filter_top_teams(raw, min_maps=20)
    feat = build_features(raw)
    fcols = get_feature_columns(feat)
    results = train_all(feat, fcols)

    print("\n=== FINAL RESULTS ===")
    for t, m in results.items():
        print(f"  {t:10s}  MAE={m['mae']:.3f}  within±1={m['within_1.0']*100:.1f}%")
