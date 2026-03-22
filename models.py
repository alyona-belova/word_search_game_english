#!/usr/bin/env python3
"""
Comparative ML analysis of level completion in the word-search game.

Target variable : level_status  (completed = 1 / dropped = 0)
                  "in_progress" rows are excluded — they have no conclusive outcome.

Usage:
    python3 models.py                        # auto-picks the latest TSV in reports/
    python3 models.py reports/my_file.tsv   # explicit file
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠  xgboost not found — XGBoost model will be skipped. "
          "Install with: pip install xgboost")

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUTPUT_DIR = Path("reports")


# Load data

def find_latest_tsv() -> Path:
    tsvs = sorted(OUTPUT_DIR.glob("metrica-sessions-*.tsv"))
    if not tsvs:
        raise FileNotFoundError(
            "No metrica-sessions-*.tsv found in reports/. "
            "Run fetch_logs.py first."
        )
    return tsvs[-1]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    print(f"Loaded {len(df)} rows from {path.name}")

    # Keep only conclusive level outcomes
    df = df[df["level_status"].isin(["completed", "dropped"])].copy()
    print(f"After filtering in_progress: {len(df)} rows  "
          f"(completed={( df['level_status']=='completed').sum()}, "
          f"dropped={(df['level_status']=='dropped').sum()})")

    if len(df) < 30:
        print("\n⚠  Very few samples — collect more sessions before relying on results.\n")

    return df


# Feature engineering

NUMERIC_FEATURES = [
    "level",                    # level number (difficulty proxy)
    "hints_used",               # how many hints the user requested
    "words_found",              # words found before outcome
    "words_total",              # total words in the level
    "visit_duration_sec",       # total session length
    "time_to_first_word_sec",   # seconds from level load to first found word
    "level_seq",                # ordinal position of level in session (1, 2, 3…)
    "visit_count",              # cumulative visits by this user
    "hour_of_day",              # local hour when session started (0–23)
]

CATEGORICAL_FEATURES = [
    "ab_group",         # A / B interface variant
    "is_returning",     # true / false / 1 / 0
    "is_new_user",      # 0 / 1
    "device_category",  # desktop / mobile / tablet
    "utm_source",       # UTM source (google, vk, direct, …)
    "utm_medium",       # UTM medium (cpc, organic, referral, …)
]

# completion_pct is intentionally excluded — for "dropped" sessions it
# directly encodes where the user stopped, making it near-perfectly predictive
# and causing data leakage. duration_sec (level-level) is also omitted as it
# is often missing for short/bounced attempts.


def engineer_features(df: pd.DataFrame):
    data = df.copy()

    # Coerce numeric columns
    for col in NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        else:
            data[col] = 0.0

    # Normalise boolean-like fields
    for col in ("is_returning", "is_new_user"):
        if col in data.columns:
            data[col] = (
                data[col].str.lower()
                .map({"true": 1, "1": 1, "false": 0, "0": 0})
                .fillna(0)
                .astype(int)
            )

    # Encode categorical columns with LabelEncoder
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in data.columns and col not in ("is_returning", "is_new_user"):
            le = LabelEncoder()
            data[col] = le.fit_transform(
                data[col].fillna("unknown").astype(str)
            )
            encoders[col] = le
        elif col not in data.columns:
            data[col] = 0

    # Binary target
    data["target"] = (data["level_status"] == "completed").astype(int)

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
                    if c in data.columns]

    X = data[feature_cols].values
    y = data["target"].values
    return X, y, feature_cols


# Models

def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
            verbosity=0,
        )
    return models


# Cross-validation

SCORERS = {
    "accuracy":  make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall":    make_scorer(recall_score, zero_division=0),
    "f1":        make_scorer(f1_score, zero_division=0),
    "roc_auc":   make_scorer(roc_auc_score, needs_proba=True),
}

N_SPLITS = 5


def evaluate_models(models, X, y) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for name, model in models.items():
        print(f"  Evaluating {name}…")
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORERS)
        records.append({
            "Model": name,
            "Accuracy":  scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall":    scores["test_recall"].mean(),
            "F1":        scores["test_f1"].mean(),
            "ROC-AUC":   scores["test_roc_auc"].mean(),
            # standard deviations (useful for thesis)
            "Acc±":      scores["test_accuracy"].std(),
            "F1±":       scores["test_f1"].std(),
            "AUC±":      scores["test_roc_auc"].std(),
        })
    return pd.DataFrame(records).set_index("Model")


# Visualisations

def plot_metric_comparison(results: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    errors  = {"Accuracy": "Acc±", "F1": "F1±", "ROC-AUC": "AUC±"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle("Model Comparison — 5-Fold Cross-Validation", fontsize=14)

    colors = plt.cm.tab10.colors

    for ax, metric in zip(axes, metrics):
        vals = results[metric]
        errs = results.get(errors.get(metric, ""), pd.Series(0, index=results.index))
        bars = ax.bar(range(len(vals)), vals, yerr=errs, capsize=4,
                      color=colors[:len(vals)], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(results.index, rotation=25, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_roc_curves(models, X, y):
    """Train each model on full data and plot ROC curves."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for (name, model), color in zip(models.items(), colors):
        tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            viz = RocCurveDisplay.from_estimator(
                model, X[test_idx], y[test_idx], ax=ax, alpha=0
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        ax.plot(mean_fpr, mean_tpr, color=color,
                label=f"{name} (AUC={np.mean(aucs):.2f}±{np.std(aucs):.2f})",
                linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 5-Fold Cross-Validation")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_confusion_matrices(models, X, y):
    """Confusion matrix for each model trained on the full dataset."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices (trained on full dataset)", fontsize=12)

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X, y)
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["dropped", "completed"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)

    plt.tight_layout()
    out = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_feature_importance(models, feature_cols, X, y):
    """Feature importances for tree-based models and LR coefficients."""
    importances = {}

    # Random Forest
    if "Random Forest" in models:
        rf = models["Random Forest"]
        rf.fit(X, y)
        importances["Random Forest"] = rf.feature_importances_

    # Decision Tree
    if "Decision Tree" in models:
        dt = models["Decision Tree"]
        dt.fit(X, y)
        importances["Decision Tree"] = dt.feature_importances_

    # XGBoost
    if HAS_XGBOOST and "XGBoost" in models:
        xgb = models["XGBoost"]
        xgb.fit(X, y)
        importances["XGBoost"] = xgb.feature_importances_

    if not importances:
        return

    n = len(importances)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Feature Importances", fontsize=13)

    for ax, (name, imp) in zip(axes, importances.items()):
        idx = np.argsort(imp)[::-1]
        ax.barh([feature_cols[i] for i in idx], imp[idx], color="steelblue", alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title(name, fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout()
    out = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# User clustering (Analysis Block 2)

USER_AGG_FEATURES = [
    "completion_rate",        # share of levels completed
    "avg_hints_used",         # mean hints per level
    "avg_time_to_first_word", # mean seconds to first word
    "avg_drop_off_pct",       # mean % reached on dropped levels (0 if none)
    "total_levels_played",    # engagement volume
    "visit_count",            # sessions count
]


def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate attempt-level rows into one row per unique user (client_id)."""
    raw = df.copy()
    for col in ("hints_used", "time_to_first_word_sec", "drop_off_pct",
                "visit_count"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)

    completed = raw["level_status"] == "completed"
    dropped   = raw["level_status"] == "dropped"

    users = raw.groupby("client_id").apply(lambda g: pd.Series({
        "completion_rate":        completed[g.index].sum() / max(len(g), 1),
        "avg_hints_used":         g["hints_used"].mean(),
        "avg_time_to_first_word": g["time_to_first_word_sec"].mean()
                                  if "time_to_first_word_sec" in g else 0,
        "avg_drop_off_pct":       g.loc[dropped[g.index], "drop_off_pct"].mean()
                                  if dropped[g.index].any() else 0,
        "total_levels_played":    len(g),
        "visit_count":            pd.to_numeric(g["visit_count"],
                                  errors="coerce").max()
                                  if "visit_count" in g.columns else 1,
        "ab_group":               g["ab_group"].iloc[0],
    }), include_groups=False).reset_index()

    users[USER_AGG_FEATURES] = users[USER_AGG_FEATURES].fillna(0)
    return users


def select_k(X_scaled: np.ndarray, k_range=range(2, 7)) -> int:
    """Pick k with the best silhouette score (elbow fallback if n too small)."""
    if len(X_scaled) < max(k_range):
        return 2
    scores = {}
    for k in k_range:
        if k >= len(X_scaled):
            continue
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=10).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.get)
    print(f"  Silhouette scores: { {k: round(v,3) for k,v in scores.items()} }")
    print(f"  Selected k = {best_k}")
    return best_k


def plot_user_clusters(users: pd.DataFrame, labels: np.ndarray,
                       X_scaled: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("User Clustering", fontsize=13)

    # PCA scatter
    ax = axes[0]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    palette = plt.cm.tab10.colors
    for cl in np.unique(labels):
        mask = labels == cl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=palette[cl], label=f"Cluster {cl}", s=60, alpha=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("PCA projection")
    ax.legend()

    # Cluster profile radar / bar
    ax = axes[1]
    profile = (users.copy()
               .assign(cluster=labels)
               .groupby("cluster")[USER_AGG_FEATURES]
               .mean())
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    x = np.arange(len(USER_AGG_FEATURES))
    width = 0.8 / len(profile)
    for i, (cl, row) in enumerate(profile_norm.iterrows()):
        bars = ax.bar(x + i * width, row.values, width,
                      label=f"Cluster {cl}", color=palette[i], alpha=0.85)
    ax.set_xticks(x + width * (len(profile) - 1) / 2)
    ax.set_xticklabels(USER_AGG_FEATURES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Normalised mean")
    ax.set_title("Cluster profiles")
    ax.legend()

    plt.tight_layout()
    out = OUTPUT_DIR / "user_clusters.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def run_user_clustering(df: pd.DataFrame):
    print("\n=== User clustering (Block 2) ===")
    users = build_user_profiles(df)
    print(f"  Unique users: {len(users)}")

    if len(users) < 4:
        print("  ⚠  Too few users for meaningful clustering — collect more sessions.")
        return

    scaler = MinMaxScaler()
    X_u = scaler.fit_transform(users[USER_AGG_FEATURES].values)

    k = select_k(X_u)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_u)
    users["cluster"] = labels

    # Print cluster summary
    summary = (users.groupby("cluster")[USER_AGG_FEATURES + ["ab_group"]]
               .agg({**{f: "mean" for f in USER_AGG_FEATURES}, "ab_group": "count"})
               .rename(columns={"ab_group": "n_users"}))
    print("\n  Cluster summary (means):")
    print(summary.round(2).to_string())

    # Label clusters by completion_rate rank for readability
    _all_tiers = ["Struggling", "Casual", "Engaged", "Power", "Expert", "Elite"]
    rank = summary["completion_rate"].rank(method="first").astype(int)
    tier_names = {r: _all_tiers[i] for i, r in enumerate(sorted(rank.unique()))}
    users["segment"] = users["cluster"].map(lambda c: tier_names[rank.loc[c]])
    print("\n  Segment distribution:")
    print(users["segment"].value_counts().to_string())

    # Derived supervised label: retained (visited more than once)
    users["retained"] = (users["visit_count"] > 1).astype(int)
    ret = users["retained"].sum()
    print(f"\n  Retained users (visit_count > 1): {ret}/{len(users)}")

    out_csv = OUTPUT_DIR / "user_segments.csv"
    users.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")

    plot_user_clusters(users, labels, X_u)


# Main
def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_tsv()

    print("\n=== Loading data ===")
    df = load_data(path)

    print("\n=== Feature engineering ===")
    X, y, feature_cols = engineer_features(df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Class balance: completed={y.sum()}, dropped={(y==0).sum()}")

    print("\n=== Training models ===")
    models = build_models()

    print(f"\n=== Cross-validation ({N_SPLITS}-fold) ===")
    results = evaluate_models(models, X, y)

    print("\n" + "=" * 65)
    print("RESULTS (mean over 5 folds)")
    print("=" * 65)
    display_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    print(results[display_cols].round(3).to_string())
    print("=" * 65)

    best = results["F1"].idxmax()
    print(f"\nBest model by F1: {best} ({results.loc[best, 'F1']:.3f})")

    # Save results table
    out_csv = OUTPUT_DIR / "results_summary.csv"
    results.round(4).to_csv(out_csv)
    print(f"Results table saved → {out_csv}")

    print("\n=== Generating plots ===")
    plot_metric_comparison(results)
    plot_roc_curves(models, X, y)
    plot_confusion_matrices(models, X, y)
    plot_feature_importance(models, feature_cols, X, y)

    # Block 2: user clustering
    run_user_clustering(df)

    print("\nDone. Check the reports/ folder for all output files.")


if __name__ == "__main__":
    main()
