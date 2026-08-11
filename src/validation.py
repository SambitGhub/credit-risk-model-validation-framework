import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    brier_score_loss,
    precision_recall_fscore_support,
)
from scipy.stats import chi2

from src.config import FIGURES_DIR, PLOT_COLORS
from src.utils import (
    compute_ks_statistic,
    compute_gini,
    compute_psi,
    interpret_psi,
    set_plot_style,
    save_figure,
    print_section_header,
    print_subsection,
)


def compute_discrimination_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    ks_stat, opt_threshold = compute_ks_statistic(y_true, y_prob)
    gini = compute_gini(roc_auc)

    y_pred_opt = (y_prob >= opt_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_opt, average="binary", zero_division=0
    )

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ks_statistic": ks_stat,
        "gini": gini,
        "optimal_threshold": opt_threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    brier = brier_score_loss(y_true, y_prob)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    try:
        df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["y_prob"], bins=n_bins)

    calib = (
        df.groupby("bin", observed=False)
        .agg(
            obs_count=("y_true", "count"),
            obs_events=("y_true", "sum"),
            pred_prob=("y_prob", "mean"),
        )
        .reset_index()
    )
    calib["exp_events"] = calib["obs_count"] * calib["pred_prob"]
    calib["obs_rate"] = calib["obs_events"] / calib["obs_count"]

    hl_stat = np.sum(
        (calib["obs_events"] - calib["exp_events"]) ** 2
        / (calib["exp_events"] * (1 - calib["pred_prob"]) + 1e-10)
    )
    hl_pvalue = 1 - chi2.cdf(hl_stat, df=max(1, len(calib) - 2))

    return {
        "brier_score": brier,
        "calibration_table": calib,
        "hl_statistic": round(hl_stat, 4),
        "hl_pvalue": round(hl_pvalue, 4),
    }


def compute_stability_metrics(
    train_probs: np.ndarray,
    oot_probs: np.ndarray,
    train_features: pd.DataFrame,
    oot_features: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    score_psi = compute_psi(train_probs, oot_probs)

    feature_psi: Dict[str, float] = {}
    for feat in feature_names:
        if feat in train_features.columns and feat in oot_features.columns:
            feature_psi[feat] = compute_psi(
                train_features[feat].values, oot_features[feat].values
            )

    return {
        "score_psi": score_psi,
        "score_interpretation": interpret_psi(score_psi),
        "feature_psi": feature_psi,
    }


def plot_roc_curves(results_dict: Dict[str, Dict[str, np.ndarray]], save: bool = True):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = list(PLOT_COLORS.values())

    for i, (name, data) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc_val = roc_auc_score(data["y_true"], data["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")

    if save:
        save_figure(fig, "roc_curves_comparison.png")
    else:
        plt.show()


def plot_ks_chart(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save: bool = True):
    set_plot_style()

    idx = np.argsort(y_prob)
    y_sorted = np.asarray(y_true)[idx]
    p_sorted = np.asarray(y_prob)[idx]

    cum_bads = np.cumsum(y_sorted == 1) / np.sum(y_sorted == 1)
    cum_goods = np.cumsum(y_sorted == 0) / np.sum(y_sorted == 0)

    ks_vals = np.abs(cum_bads - cum_goods)
    ks_stat = np.max(ks_vals)
    ks_idx = np.argmax(ks_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(p_sorted, cum_bads, color=PLOT_COLORS["danger"], lw=2, label="CDF Bads (Target=1)")
    ax.plot(p_sorted, cum_goods, color=PLOT_COLORS["success"], lw=2, label="CDF Goods (Target=0)")
    ax.vlines(p_sorted[ks_idx], cum_goods[ks_idx], cum_bads[ks_idx],
              color="black", linestyle="--", lw=2, label=f"KS = {ks_stat:.3f}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title(f"KS Chart — {model_name}")
    ax.legend()

    if save:
        safe_name = model_name.replace(" ", "_").lower()
        save_figure(fig, f"ks_chart_{safe_name}.png")
    else:
        plt.show()


def plot_calibration_curves(
    calibration_results: Dict[str, Dict[str, Any]], save: bool = True
):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = list(PLOT_COLORS.values())

    for i, (name, res) in enumerate(calibration_results.items()):
        ct = res["calibration_table"]
        brier = res["brier_score"]
        ax.plot(ct["pred_prob"], ct["obs_rate"], "o-", lw=2,
                color=colors[i % len(colors)],
                label=f"{name} (Brier = {brier:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Event Rate")
    ax.set_title("Calibration Curves — Model Comparison")
    ax.legend()

    if save:
        save_figure(fig, "calibration_curves_comparison.png")
    else:
        plt.show()


def plot_psi_summary(stability_results: Dict[str, Any], save: bool = True):
    set_plot_style()
    fpsi = stability_results["feature_psi"]

    sorted_items = sorted(fpsi.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    bar_colors = []
    for v in values:
        if v >= 0.25:
            bar_colors.append(PLOT_COLORS["danger"])
        elif v >= 0.10:
            bar_colors.append(PLOT_COLORS["secondary"])
        else:
            bar_colors.append(PLOT_COLORS["success"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    ax.barh(names, values, color=bar_colors)
    ax.axvline(0.10, color="gray", ls="--", alpha=0.7, label="Stable (0.10)")
    ax.axvline(0.25, color="black", ls="--", alpha=0.7, label="Shift (0.25)")
    ax.set_xlabel("PSI")
    ax.set_title("Feature Stability (Train vs OOT)")
    ax.invert_yaxis()
    ax.legend()

    if save:
        save_figure(fig, "feature_psi_summary.png")
    else:
        plt.show()


def run_full_validation(
    models: Dict[str, Any],
    val_predictions: Dict[str, np.ndarray],
    y_val: np.ndarray,
    oot_predictions: Dict[str, np.ndarray],
    y_oot: np.ndarray,
    train_predictions: Dict[str, np.ndarray],
    train_features: pd.DataFrame,
    oot_features: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    print_section_header("Running Full Model Validation (Side B)")

    results: Dict[str, Any] = {
        "discrimination": {},
        "calibration": {},
        "stability": {},
    }
    roc_data: Dict[str, Dict[str, np.ndarray]] = {}

    y_val = np.asarray(y_val)
    y_oot = np.asarray(y_oot)

    for model_name in models:
        print(f"\n  [RUN] Validating {model_name} ...")

        val_probs = val_predictions[model_name]
        oot_probs = oot_predictions[model_name]
        trn_probs = train_predictions[model_name]

        roc_data[model_name] = {"y_true": y_val, "y_prob": val_probs}

        disc = compute_discrimination_metrics(y_val, val_probs, model_name)
        results["discrimination"][model_name] = disc
        print(f"     ROC-AUC = {disc['roc_auc']:.4f}  |  KS = {disc['ks_statistic']:.4f}  |  Gini = {disc['gini']:.4f}")

        plot_ks_chart(y_val, val_probs, model_name)

        cal = compute_calibration_metrics(y_val, val_probs)
        results["calibration"][model_name] = cal
        print(f"     Brier = {cal['brier_score']:.4f}  |  H-L = {cal['hl_statistic']}  (p = {cal['hl_pvalue']})")

        stab = compute_stability_metrics(
            trn_probs, oot_probs, train_features, oot_features, feature_names
        )
        results["stability"][model_name] = stab
        print(f"     Score PSI = {stab['score_psi']:.4f}  ({stab['score_interpretation']})")

    print("\n  [RUN] Generating comparison plots ...")
    plot_roc_curves(roc_data)
    plot_calibration_curves(results["calibration"])

    first_model = list(results["stability"].keys())[0]
    plot_psi_summary(results["stability"][first_model])

    print("\n  [OK] Full validation complete. Plots saved to outputs/figures/.")
    return results
