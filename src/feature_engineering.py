import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from pathlib import Path

from src.config import (
    CONTINUOUS_FEATURES,
    DISCRETE_FEATURES,
    FEATURE_COLS,
    WOE_INITIAL_BINS,
    WOE_MIN_BIN_FRACTION,
    IV_THRESHOLDS,
    FIGURES_DIR,
)
from src.utils import interpret_iv, save_figure, print_section_header, get_feature_label


def compute_woe_iv_for_feature(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    n_bins: int = 20,
    min_bin_frac: float = 0.05,
) -> pd.DataFrame:
    df_temp = df[[feature, target_col]].copy()

    if feature in CONTINUOUS_FEATURES:
        bins = pd.qcut(df_temp[feature], q=n_bins, duplicates="drop", retbins=True)[1]
        bins[0] = -np.inf
        bins[-1] = np.inf
        df_temp["Bin"] = pd.cut(df_temp[feature], bins=bins).astype(str)
    else:
        df_temp["Bin"] = df_temp[feature].astype(str)
        val_counts = df_temp["Bin"].value_counts(normalize=True)
        rare_vals = val_counts[val_counts < min_bin_frac].index
        df_temp.loc[df_temp["Bin"].isin(rare_vals), "Bin"] = "Rare/Other"

    df_temp.loc[df_temp[feature].isnull(), "Bin"] = "Missing"

    grouped = df_temp.groupby("Bin", observed=False).agg(
        Total=(target_col, "count"),
        Events=(target_col, "sum"),
    )
    grouped["Non_Events"] = grouped["Total"] - grouped["Events"]

    total_events = grouped["Events"].sum()
    total_non_events = grouped["Non_Events"].sum()

    grouped["Pct_Events"] = np.maximum(grouped["Events"] / total_events, 1e-6)
    grouped["Pct_Non_Events"] = np.maximum(grouped["Non_Events"] / total_non_events, 1e-6)

    grouped["WOE"] = np.log(grouped["Pct_Events"] / grouped["Pct_Non_Events"])
    grouped["IV"] = (grouped["Pct_Events"] - grouped["Pct_Non_Events"]) * grouped["WOE"]

    grouped = grouped.reset_index()
    grouped["Feature"] = feature

    if feature in CONTINUOUS_FEATURES:
        grouped["Bin_Edges"] = [bins] * len(grouped)

    return grouped


def compute_all_woe_iv(
    df: pd.DataFrame, features: List[str], target_col: str
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    woe_tables: Dict[str, pd.DataFrame] = {}
    iv_records = []

    for f in features:
        if f not in df.columns:
            continue
        table = compute_woe_iv_for_feature(
            df, f, target_col, WOE_INITIAL_BINS, WOE_MIN_BIN_FRACTION
        )
        woe_tables[f] = table
        total_iv = table["IV"].sum()
        iv_records.append({"Feature": f, "IV": total_iv})

    iv_summary = (
        pd.DataFrame(iv_records)
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )
    return woe_tables, iv_summary


def select_features_by_iv(
    iv_summary: pd.DataFrame, min_iv: float = 0.02, max_iv: float = 0.50
) -> List[str]:
    selected = []
    for _, row in iv_summary.iterrows():
        f = row["Feature"]
        iv = row["IV"]
        if iv > max_iv:
            print(f"  [WARN] {f}: IV = {iv:.4f} — Suspiciously high. Including with caution.")
            selected.append(f)
        elif iv >= min_iv:
            interp = interpret_iv(iv)
            print(f"  [OK]   {f}: IV = {iv:.4f} — {interp}")
            selected.append(f)
        else:
            print(f"  [DROP] {f}: IV = {iv:.4f} — Dropped (not predictive)")
    return selected


def transform_to_woe(
    df: pd.DataFrame,
    woe_tables: Dict[str, pd.DataFrame],
    features: List[str],
) -> pd.DataFrame:
    df_woe = df.copy()

    for f in features:
        if f not in woe_tables:
            continue
        table = woe_tables[f]
        woe_map = dict(zip(table["Bin"], table["WOE"]))

        if f in CONTINUOUS_FEATURES:
            bin_edges = table["Bin_Edges"].iloc[0] if "Bin_Edges" in table.columns else None
            if bin_edges is not None:
                binned = pd.cut(df_woe[f], bins=bin_edges).astype(str)
                binned.loc[df_woe[f].isnull()] = "Missing"
                df_woe[f] = binned.map(woe_map).fillna(0.0)
            else:
                df_woe[f] = 0.0
        else:
            mapped = df_woe[f].astype(str)
            mapped.loc[df_woe[f].isnull()] = "Missing"
            mapped = mapped.apply(lambda x: x if x in woe_map else "Rare/Other")
            df_woe[f] = mapped.map(woe_map).fillna(0.0)

    return df_woe


def plot_woe_iv_charts(
    woe_tables: Dict[str, pd.DataFrame], iv_summary: pd.DataFrame
) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, len(iv_summary) * 0.5)))
    sns.barplot(data=iv_summary, x="IV", y="Feature", ax=ax, palette="viridis")
    ax.set_title("Information Value (IV) by Feature")
    ax.axvline(x=0.02, color="gray", linestyle="--", alpha=0.5, label="Weak (0.02)")
    ax.axvline(x=0.10, color="orange", linestyle="--", alpha=0.5, label="Medium (0.10)")
    ax.axvline(x=0.30, color="red", linestyle="--", alpha=0.5, label="Strong (0.30)")
    ax.legend(fontsize=8)
    save_figure(fig, "iv_summary.png")

    for f, table in woe_tables.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=table, x="Bin", y="WOE", ax=ax, palette="coolwarm")
        ax.set_title(f"WOE Pattern — {get_feature_label(f)}")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, f"woe_pattern_{f}.png")


def woe_iv_pipeline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_oot: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    features = [c for c in FEATURE_COLS if c in df_train.columns]

    print_section_header("WOE/IV Pipeline")
    print("  [RUN] Computing WOE and IV on training data ...")
    woe_tables, iv_summary = compute_all_woe_iv(df_train, features, target_col)

    print("\n  [SUMMARY] IV Summary:")
    for _, row in iv_summary.iterrows():
        print(f"     {row['Feature']:45s}  IV = {row['IV']:.4f}  ({interpret_iv(row['IV'])})")

    print("\n  [RUN] Selecting features by IV ...")
    selected_features = select_features_by_iv(iv_summary)

    print(f"\n  [OK] {len(selected_features)} features selected out of {len(features)}")

    print("\n  [RUN] Generating WOE/IV charts ...")
    plot_woe_iv_charts(woe_tables, iv_summary)

    print("\n  [RUN] Transforming datasets to WOE ...")
    train_woe = transform_to_woe(df_train, woe_tables, selected_features)
    val_woe = transform_to_woe(df_val, woe_tables, selected_features)
    oot_woe = transform_to_woe(df_oot, woe_tables, selected_features)

    return {
        "X_train_woe": train_woe[selected_features],
        "X_val_woe": val_woe[selected_features],
        "X_oot_woe": oot_woe[selected_features],
        "woe_tables": woe_tables,
        "iv_summary": iv_summary,
        "selected_features": selected_features,
    }
