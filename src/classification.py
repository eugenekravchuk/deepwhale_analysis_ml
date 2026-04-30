from __future__ import annotations

"""
classification.py
=================
Supervised whale-type classification using KMeans cluster labels as target.

The clustering step (clustering.py) groups whales into behavioural clusters
using unsupervised learning. This script trains XGBoost to predict those
cluster labels on new, unseen addresses in real-time, and produces SHAP
explanations showing which features drove each prediction.

Pipeline:
  1. Load clustered_addresses.csv (from clustering.py)
  2. Apply the same log1p + RobustScaler preprocessing as clustering
  3. Time-based train/test split (80/20 by first_seen)
  4. StratifiedKFold cross-validation for robust F1 estimate
  5. Train XGBoost (primary) and Random Forest (comparison)
  6. Evaluate: F1 macro, confusion matrix, classification report
  7. SHAP summary + per-class SHAP plots
  8. Save trained model to models/whale_classifier.pkl

Run:
  python classification.py
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("shap not installed — SHAP plots will be skipped.")

FEATURE_COLS = [
    "tx_count_out",
    "total_eth_out",
    "avg_tx_eth",
    "median_tx_eth",
    "max_tx_eth",
    "std_tx_eth",
    "unique_receivers",
    "exchange_ratio",
    "top1_receiver_ratio",
    "round_number_ratio",
    "avg_gas_gwei",
    "gas_variability",
    "hour_entropy",
    "active_days",
    "net_flow_eth",
]

# Financial columns that need log1p (must match clustering.py exactly)
LOG_COLS = [
    "tx_count_out", "total_eth_out", "avg_tx_eth", "median_tx_eth",
    "max_tx_eth", "std_tx_eth", "unique_receivers", "avg_gas_gwei",
    "gas_variability", "active_days",
]

TARGET_COL = "kmeans_cluster"

CLUSTER_NAMES = {
    0: "Large One-Shot Movers",
    1: "Mega-Whale Traders",
    2: "Small Whales",
    3: "Mid-Range Regulars",
}


# ── Data loading & preprocessing ───────────────────────────────────────────

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to heavy-tailed columns (mirrors clustering.py)."""
    df = df.copy()
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    if "net_flow_eth" in df.columns:
        nf = df["net_flow_eth"]
        df["net_flow_eth"] = np.sign(nf) * np.log1p(nf.abs())
    return df


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clustered data and split into train/test by time."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} clustered addresses")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Column '{TARGET_COL}' not found. Run clustering.py first."
        )

    # Impute missing numeric features
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"  Cluster distribution:\n{df[TARGET_COL].value_counts().sort_index().to_string()}")

    # Time-based split: 80% earliest → train, 20% latest → test
    if "first_seen" in df.columns:
        df_sorted = df.sort_values("first_seen")
    elif "block_number" in df.columns:
        df_sorted = df.sort_values("block_number")
    else:
        df_sorted = df.copy()

    split_idx = int(len(df_sorted) * 0.80)
    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()
    print(f"  Train: {len(train)}  |  Test: {len(test)}")
    return train, test


def prepare_xy(df: pd.DataFrame, scaler: RobustScaler,
               fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Extract features, apply log1p + scale, return X and y."""
    avail = [c for c in FEATURE_COLS if c in df.columns]
    df_feat = log_transform(df[avail])
    X = df_feat.values

    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    y = df[TARGET_COL].values.astype(int)
    return X, y


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred, class_names: list[str], out_path: Path,
    labels: list[int] | None = None,
):
    if labels is None:
        labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Whale Cluster Classifier", color="white", fontsize=13)
    for text in ax.texts:
        text.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    plt.xticks(rotation=30, ha="right", color="white")
    plt.yticks(color="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_feature_importance(model, feature_names: list[str], out_path: Path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(importances)))
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color=colors,
    )
    ax.set_xlabel("Feature Importance (gain)", color="white")
    ax.set_title("XGBoost Feature Importance — Cluster Prediction", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_shap(model, X_train: np.ndarray, feature_names: list[str], out_path: Path):
    if not SHAP_AVAILABLE:
        return
    print("  Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        stacked = np.stack([np.asarray(s) for s in shap_values], axis=0)
        mean_shap = np.abs(stacked).mean(axis=(0, 1))
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            mean_shap = np.abs(sv).mean(axis=(0, 2))
        else:
            mean_shap = np.abs(sv).mean(axis=0)

    mean_shap = np.asarray(mean_shap).ravel()
    n_feat = min(len(feature_names), len(mean_shap))
    mean_shap = mean_shap[:n_feat]
    sorted_idx = np.argsort(mean_shap)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    fnames = feature_names[:n_feat]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_idx)))
    ax.barh(
        [fnames[int(i)] for i in sorted_idx],
        mean_shap[sorted_idx],
        color=colors,
    )
    ax.set_xlabel("Mean |SHAP value|", color="white")
    ax.set_title("SHAP Feature Impact (mean absolute, all clusters)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_shap_per_class(model, X_train: np.ndarray, feature_names: list[str],
                        class_names: list[str], out_dir: Path):
    """Generate one SHAP importance bar chart per cluster."""
    if not SHAP_AVAILABLE:
        return
    print("  Computing per-cluster SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        per_class = [np.asarray(s) for s in shap_values]
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            per_class = [sv[:, :, c] for c in range(sv.shape[2])]
        else:
            per_class = [sv]

    n_feat = min(len(feature_names), per_class[0].shape[1])
    n_classes = min(len(class_names), len(per_class))
    fnames = feature_names[:n_feat]

    cols = min(n_classes, 2)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))
    fig.patch.set_facecolor("#0d1117")
    if n_classes == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    class_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i in range(n_classes):
        ax = axes[i]
        ax.set_facecolor("#0d1117")
        mean_abs = np.abs(per_class[i][:, :n_feat]).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)
        color = class_colors[i % len(class_colors)]
        ax.barh(
            [fnames[int(j)] for j in sorted_idx],
            mean_abs[sorted_idx],
            color=color, alpha=0.85,
        )
        ax.set_xlabel("Mean |SHAP value|", color="white")
        ax.set_title(f"SHAP — {class_names[i]}", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")

    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Per-Cluster SHAP Feature Impact", color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = out_dir / "shap_per_class.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


# ── Cross-validation ────────────────────────────────────────────────────────

def run_cross_validation(X: np.ndarray, y: np.ndarray, n_classes: int, n_splits: int = 5):
    """StratifiedKFold CV on XGBoost for robust F1 macro estimate."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
        random_state=42,
        verbosity=0,
    )

    cv_scores = cross_val_score(
        XGBClassifier(**xgb_params), X, y,
        cv=skf, scoring="f1_macro", n_jobs=-1,
    )
    print(f"\n[Cross-Validation] {n_splits}-fold Stratified CV (XGBoost)")
    print(f"  Fold F1 scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean F1 macro:  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    return cv_scores


# ── Model persistence ───────────────────────────────────────────────────────

def save_model(xgb_model, rf_model, scaler, cluster_names, feature_names, path: Path):
    bundle = {
        "xgb": xgb_model,
        "rf": rf_model,
        "scaler": scaler,
        "cluster_names": cluster_names,
        "feature_cols": feature_names,
        "log_cols": LOG_COLS,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Model bundle saved to {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def run(clustered_csv, model_pkl, figures_dir) -> dict:
    """Module entry point: train classifiers and save model bundle."""
    clustered_csv = Path(clustered_csv)
    model_pkl = Path(model_pkl)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_pkl.parent.mkdir(parents=True, exist_ok=True)

    if not clustered_csv.exists():
        raise FileNotFoundError(f"{clustered_csv} not found. Run clustering first.")

    train_df, test_df = load_data(clustered_csv)

    n_classes = train_df[TARGET_COL].nunique()
    class_names = [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in range(n_classes)]
    label_indices = list(range(n_classes))
    print(f"Classes ({n_classes}): {class_names}")

    scaler = RobustScaler()
    avail_features = [c for c in FEATURE_COLS if c in train_df.columns]

    X_train, y_train = prepare_xy(train_df, scaler, fit=True)
    X_test, y_test = prepare_xy(test_df, scaler, fit=False)

    # ── Cross-Validation ────────────────────────────────────────────────────
    cv_scores = run_cross_validation(X_train, y_train, n_classes=n_classes)

    # ── XGBoost ─────────────────────────────────────────────────────────────
    print("\n[XGBoost] Training ...")
    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
        random_state=42,
        verbosity=0,
    )
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb.predict(X_test)

    f1_xgb = f1_score(
        y_test, y_pred_xgb, average="macro", zero_division=0, labels=label_indices
    )
    print(f"  F1 macro (XGBoost): {f1_xgb:.4f}")
    print(
        classification_report(
            y_test, y_pred_xgb,
            labels=label_indices,
            target_names=class_names,
            zero_division=0,
        )
    )

    plot_confusion_matrix(
        y_test, y_pred_xgb, class_names, figures_dir / "cm_xgboost.png",
        labels=label_indices,
    )
    plot_feature_importance(xgb, avail_features, figures_dir / "feature_importance.png")
    plot_shap(xgb, X_train, avail_features, figures_dir / "shap_summary.png")
    plot_shap_per_class(xgb, X_train, avail_features, class_names, figures_dir)

    # ── Random Forest (comparison) ──────────────────────────────────────────
    print("\n[Random Forest] Training ...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    f1_rf = f1_score(
        y_test, y_pred_rf, average="macro", zero_division=0, labels=label_indices
    )
    print(f"  F1 macro (Random Forest): {f1_rf:.4f}")
    print(
        classification_report(
            y_test, y_pred_rf,
            labels=label_indices,
            target_names=class_names,
            zero_division=0,
        )
    )

    plot_confusion_matrix(
        y_test, y_pred_rf, class_names, figures_dir / "cm_randomforest.png",
        labels=label_indices,
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n=== Model Comparison ===")
    print(f"  XGBoost   F1-macro (test): {f1_xgb:.4f}")
    print(f"  XGBoost   F1-macro (CV):   {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  RF        F1-macro (test): {f1_rf:.4f}")
    winner = "XGBoost" if f1_xgb >= f1_rf else "Random Forest"
    print(f"  Best model: {winner}")

    if f1_xgb >= 0.65:
        print("  Classification quality: GOOD (F1 >= 0.65)")
    elif f1_xgb >= 0.40:
        print("  Classification quality: FAIR — collect more data")
    else:
        print("  Classification quality: WEAK — need more diverse training samples")

    # ── Save ────────────────────────────────────────────────────────────────
    save_model(xgb, rf, scaler, CLUSTER_NAMES, avail_features, model_pkl)
    return {"f1_xgb": f1_xgb, "f1_rf": f1_rf, "cv_mean": cv_scores.mean()}


if __name__ == "__main__":
    _root = Path(__file__).parent.parent
    run(
        clustered_csv=_root / "data" / "clustered_addresses.csv",
        model_pkl=_root / "models" / "whale_classifier.pkl",
        figures_dir=_root / "data",
    )
