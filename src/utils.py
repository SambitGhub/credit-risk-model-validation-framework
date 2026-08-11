"""
utils.py — Shared Plotting & Metric Utilities
===============================================

Reusable helper functions for:
  - Styled matplotlib/seaborn plots
  - Common metric calculations
  - Table formatting for reports
  - File I/O helpers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from src.config import (
    PLOT_STYLE,
    PLOT_DPI,
    PLOT_FIGSIZE,
    PLOT_COLORS,
    FIGURES_DIR,
    FEATURE_LABELS,
)


# ──────────────────────────────────────────────
# 1. PLOT STYLING
# ──────────────────────────────────────────────

def set_plot_style():
    """Apply consistent plot styling across all figures."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    
    plt.rcParams.update({
        "figure.figsize": PLOT_FIGSIZE,
        "figure.dpi": PLOT_DPI,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "grid.alpha": 0.3,
    })


def save_figure(fig: plt.Figure, filename: str, tight: bool = True) -> Path:
    """Save figure to the outputs/figures/ directory.
    
    Args:
        fig: Matplotlib figure object.
        filename: Name of the file (e.g. 'roc_curve.png').
        tight: Whether to use tight_layout before saving.
    
    Returns:
        Path to the saved figure.
    """
    filepath = FIGURES_DIR / filename
    if tight:
        fig.tight_layout()
    fig.savefig(filepath, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [PLOT] Figure saved: {filepath}")
    return filepath


def get_feature_label(feature_name: str) -> str:
    """Get human-readable label for a feature."""
    return FEATURE_LABELS.get(feature_name, feature_name)


# ──────────────────────────────────────────────
# 2. COMMON METRIC HELPERS
# ──────────────────────────────────────────────

def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Compute the Kolmogorov-Smirnov statistic.
    
    KS = max |CDF_good(s) - CDF_bad(s)| over all score thresholds s.
    
    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities of the positive class.
    
    Returns:
        Tuple of (KS statistic, threshold at which KS is maximized).
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    ks_stat = np.max(ks_values)
    ks_threshold = thresholds[np.argmax(ks_values)]
    return ks_stat, ks_threshold


def compute_gini(auc_score: float) -> float:
    """Compute Gini coefficient from AUC.
    
    Gini = 2 × AUC - 1
    
    Args:
        auc_score: Area Under the ROC Curve.
    
    Returns:
        Gini coefficient.
    """
    return 2 * auc_score - 1


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI).
    
    PSI = Σ (% Actual_i - % Expected_i) × ln(% Actual_i / % Expected_i)
    
    Args:
        expected: Array of values from the reference (training) population.
        actual: Array of values from the new (OOT/production) population.
        n_bins: Number of quantile bins.
        epsilon: Small value to avoid division by zero.
    
    Returns:
        PSI value.
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Remove duplicate breakpoints
    breakpoints = np.unique(breakpoints)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct = actual_counts / len(actual) + epsilon
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def interpret_psi(psi_value: float) -> str:
    """Interpret PSI value with traffic-light label.
    
    Args:
        psi_value: Computed PSI.
    
    Returns:
        Interpretation string.
    """
    if psi_value < 0.10:
        return "Stable (PSI < 0.10) - No significant shift"
    elif psi_value < 0.25:
        return "Moderate Shift (0.10 <= PSI < 0.25) - Investigation recommended"
    else:
        return "Significant Drift (PSI >= 0.25) - Model recalibration required"


def interpret_iv(iv_value: float) -> str:
    """Interpret Information Value.
    
    Args:
        iv_value: Computed IV for a feature.
    
    Returns:
        Interpretation string.
    """
    if iv_value < 0.02:
        return "Not predictive"
    elif iv_value < 0.10:
        return "Weak predictor"
    elif iv_value < 0.30:
        return "Medium predictor"
    elif iv_value < 0.50:
        return "Strong predictor"
    else:
        return "Suspicious (possible leakage)"


# ──────────────────────────────────────────────
# 3. TABLE FORMATTING
# ──────────────────────────────────────────────

def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Format a model comparison metrics table.
    
    Args:
        metrics_dict: Dict of {model_name: {metric_name: value}}.
    
    Returns:
        Formatted DataFrame.
    """
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Model"
    return df.round(4)


def print_section_header(title: str, char: str = "=", width: int = 60):
    """Print a formatted section header for console output."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_subsection(title: str, char: str = "-", width: int = 50):
    """Print a formatted subsection header."""
    print(f"\n  {char * width}")
    print(f"  {title}")
    print(f"  {char * width}")


# ──────────────────────────────────────────────
# 4. DATA QUALITY HELPERS
# ──────────────────────────────────────────────

def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a comprehensive data quality summary.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Summary DataFrame with dtype, missing count/%, unique count, and basic stats.
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique_count": df.nunique(),
        "mean": df.select_dtypes(include="number").mean(),
        "median": df.select_dtypes(include="number").median(),
        "std": df.select_dtypes(include="number").std(),
        "min": df.select_dtypes(include="number").min(),
        "max": df.select_dtypes(include="number").max(),
    })
    return summary


# ──────────────────────────────────────────────
# 5. DECILE / BINNING HELPERS
# ──────────────────────────────────────────────

def create_decile_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create a decile analysis table (lift chart data).
    
    Sorts predictions into deciles and computes observed default rate,
    cumulative capture rate, and lift for each decile.
    
    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins (default 10 for deciles).
    
    Returns:
        DataFrame with decile analysis.
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], q=n_bins, labels=False, duplicates="drop") + 1
    
    agg = df.groupby("decile").agg(
        n_total=("y_true", "count"),
        n_defaults=("y_true", "sum"),
        avg_predicted_pd=("y_prob", "mean"),
    ).reset_index()
    
    agg["observed_default_rate"] = agg["n_defaults"] / agg["n_total"]
    agg["cumulative_defaults"] = agg["n_defaults"].cumsum()
    agg["cumulative_capture_rate"] = agg["cumulative_defaults"] / agg["n_defaults"].sum()
    
    total_default_rate = y_true.mean()
    agg["lift"] = agg["observed_default_rate"] / total_default_rate
    
    return agg.sort_values("decile", ascending=False).reset_index(drop=True)
