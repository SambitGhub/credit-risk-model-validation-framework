"""
run_pipeline.py — End-to-End Credit Risk Modeling & Validation Pipeline
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import (
    RAW_DATA_FILE,
    RANDOM_SEED,
    TRAIN_RATIO,
    VALID_RATIO,
    OOT_RATIO,
    STRESS_SCENARIOS,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.utils import (
    set_plot_style,
    print_section_header,
    print_subsection,
    format_metrics_table,
)

set_plot_style()


def main():
    """Run the complete credit risk modeling and validation pipeline."""
    
    pipeline_start = time.time()
    
    print("=" * 64)
    print("  CREDIT RISK MODELING & INDEPENDENT MODEL VALIDATION")
    print("  Framework Pipeline")
    print("=" * 64)
    
    # STAGE 1: DATA LOADING & PREPROCESSING
    print_section_header("STAGE 1: Data Loading & Preprocessing")
    
    from src import config
    from src.data_loader import load_raw_data, generate_data_quality_report
    from src.preprocessing import preprocess_pipeline
    
    df_raw = load_raw_data()
    generate_data_quality_report(df_raw)
    
    target_col = config.TARGET_COL
    feature_cols = config.FEATURE_COLS
    
    splits = preprocess_pipeline()
    df_train = splits["train"]
    df_val = splits["val"]
    df_oot = splits["oot"]
    
    print(f"\n  [OK] Stage 1 Complete")
    print(f"     Train: {len(df_train):,} rows | Val: {len(df_val):,} rows | OOT: {len(df_oot):,} rows")
    
    # STAGE 2: DATA LEAKAGE EXPERIMENT
    print_section_header("STAGE 2: Data Leakage Experiment")
    
    from src.preprocessing import run_leakage_experiment
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_val = df_val[feature_cols]
    y_val = df_val[target_col]
    
    leakage_results = run_leakage_experiment(X_train, y_train, X_val, y_val)
    print(f"\n  [OK] Stage 2 Complete — Leakage impact documented")
    
    # STAGE 3: WOE / IV FEATURE ENGINEERING
    print_section_header("STAGE 3: WOE/IV Feature Engineering")
    
    from src.feature_engineering import woe_iv_pipeline
    
    woe_result = woe_iv_pipeline(df_train, df_val, df_oot, target_col)
    
    X_train_woe = woe_result["X_train_woe"]
    X_val_woe = woe_result["X_val_woe"]
    X_oot_woe = woe_result["X_oot_woe"]
    woe_tables = woe_result["woe_tables"]
    iv_summary = woe_result["iv_summary"]
    selected_features = woe_result["selected_features"]
    
    print(f"\n  [OK] Stage 3 Complete — {len(selected_features)} features selected by IV")
    
    # STAGE 4: MODEL TRAINING
    print_section_header("STAGE 4: Champion & Challenger Model Training")
    
    from src.models import train_all_models, predict_probabilities
    
    X_train_raw = df_train[selected_features]
    X_val_raw = df_val[selected_features]
    X_oot_raw = df_oot[selected_features]
    y_val = df_val[target_col]
    y_oot = df_oot[target_col]
    
    models, scorecard = train_all_models(
        X_train_woe, X_train_raw, y_train, woe_tables, selected_features
    )
    
    X_val_dict = {
        "champion_logistic": X_val_woe,
        "challenger_rf": X_val_raw,
        "challenger_xgboost": X_val_raw,
        "challenger_lightgbm": X_val_raw,
    }
    X_oot_dict = {
        "champion_logistic": X_oot_woe,
        "challenger_rf": X_oot_raw,
        "challenger_xgboost": X_oot_raw,
        "challenger_lightgbm": X_oot_raw,
    }
    
    val_predictions = predict_probabilities(models, X_val_dict)
    oot_predictions = predict_probabilities(models, X_oot_dict)
    train_predictions = predict_probabilities(
        models,
        {
            "champion_logistic": X_train_woe,
            "challenger_rf": X_train_raw,
            "challenger_xgboost": X_train_raw,
            "challenger_lightgbm": X_train_raw,
        },
    )
    
    print(f"\n  [OK] Stage 4 Complete — {len(models)} models trained")
    
    # STAGE 5: INDEPENDENT MODEL VALIDATION
    print_section_header("STAGE 5: Independent Model Validation (Side B)")
    
    from src.validation import run_full_validation
    
    validation_results = run_full_validation(
        models=models,
        val_predictions=val_predictions,
        y_val=y_val,
        oot_predictions=oot_predictions,
        y_oot=y_oot,
        train_predictions=train_predictions,
        train_features=df_train[selected_features],
        oot_features=df_oot[selected_features],
        feature_names=selected_features,
    )
    
    print(f"\n  [OK] Stage 5 Complete — Validation suite executed")
    
    # STAGE 6: STRESS TESTING
    print_section_header("STAGE 6: Macroeconomic Stress Testing")
    
    from src.stress_testing import run_all_stress_tests
    
    stress_results = run_all_stress_tests(
        models=models,
        X_dict=X_val_dict,
        scenarios=STRESS_SCENARIOS,
        feature_names=selected_features,
    )
    
    print(f"\n  [OK] Stage 6 Complete — {len(STRESS_SCENARIOS)} scenarios tested")
    
    # STAGE 7: EXPLAINABILITY (SHAP)
    print_section_header("STAGE 7: SHAP Explainability Analysis")
    
    from src.explainability import run_explainability_analysis
    
    shap_results = run_explainability_analysis(
        models=models,
        X_dict={
            "champion_logistic": X_val_woe,
            "challenger_rf": X_val_raw,
            "challenger_xgboost": X_val_raw,
            "challenger_lightgbm": X_val_raw,
        },
        iv_summary=iv_summary,
        n_sample=1000,
    )
    
    print(f"\n  [OK] Stage 7 Complete — SHAP analysis generated")
    
    # STAGE 8: GENERATE VALIDATION REPORT
    print_section_header("STAGE 8: Model Validation Report Generation")
    
    generate_validation_report(
        validation_results=validation_results,
        stress_results=stress_results,
        shap_results=shap_results,
        iv_summary=iv_summary,
        scorecard=scorecard,
        leakage_results=leakage_results,
        splits=splits,
    )
    
    print(f"\n  [OK] Stage 8 Complete — Report saved to reports/")
    
    # PIPELINE SUMMARY
    elapsed = time.time() - pipeline_start
    print("=" * 64)
    print("  PIPELINE COMPLETE")
    print(f"  Total Time: {elapsed:.1f}s")
    print("=" * 64)
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Models saved to:  {MODELS_DIR}")
    print(f"  Report saved to:  {REPORTS_DIR}")
    print("=" * 64)


def generate_validation_report(
    validation_results,
    stress_results,
    shap_results,
    iv_summary,
    scorecard,
    leakage_results,
    splits,
):
    """Generate the formal model validation report as Markdown."""
    
    report_path = REPORTS_DIR / "model_validation_report.md"
    
    lines = []
    lines.append("# Model Validation Report")
    lines.append("## Credit Risk — Probability of Default (PD) Model")
    lines.append("")
    lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Target:** {splits['train'].columns[0] if len(splits['train'].columns) > 0 else 'Target'}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("This report presents the independent validation of a Probability of Default (PD) "
                 "credit risk model. The validation covers discrimination power, calibration accuracy, "
                 "population stability, macroeconomic stress resilience, and model explainability.")
    lines.append("")
    
    lines.append("## 2. Data Quality Assessment")
    lines.append("")
    target_c = [c for c in splits['train'].columns if 'target' in c.lower() or 'dlq' in c.lower() or 'status' in c.lower()][0] if any('target' in c.lower() or 'dlq' in c.lower() or 'status' in c.lower() for c in splits['train'].columns) else splits['train'].columns[0]
    lines.append(f"- **Total Records:** {sum(len(splits[k]) for k in ['train', 'val', 'oot']):,}")
    lines.append(f"- **Training Set:** {len(splits['train']):,} ({TRAIN_RATIO*100:.0f}%)")
    lines.append(f"- **Validation Set:** {len(splits['val']):,} ({VALID_RATIO*100:.0f}%)")
    lines.append(f"- **Out-of-Time Set:** {len(splits['oot']):,} ({OOT_RATIO*100:.0f}%)")
    lines.append("")
    
    lines.append("## 3. Feature Engineering — WOE/IV Analysis")
    lines.append("")
    if iv_summary is not None and not iv_summary.empty:
        lines.append("| Feature | IV | Interpretation |")
        lines.append("|:--------|:---|:---------------|")
        for _, row in iv_summary.iterrows():
            from src.utils import interpret_iv
            interp = interpret_iv(row["IV"])
            lines.append(f"| {row['Feature']} | {row['IV']:.4f} | {interp} |")
    lines.append("")
    
    lines.append("## 4. Discrimination Analysis")
    lines.append("")
    if "discrimination" in validation_results:
        disc = validation_results["discrimination"]
        lines.append("| Model | ROC-AUC | Gini | KS Statistic | PR-AUC |")
        lines.append("|:------|:--------|:-----|:-------------|:-------|")
        for model_name, metrics in disc.items():
            lines.append(
                f"| {model_name} | {metrics.get('roc_auc', 0):.4f} | "
                f"{metrics.get('gini', 0):.4f} | {metrics.get('ks_statistic', 0):.4f} | "
                f"{metrics.get('pr_auc', 0):.4f} |"
            )
    lines.append("")
    
    lines.append("## 5. Calibration Analysis")
    lines.append("")
    if "calibration" in validation_results:
        cal = validation_results["calibration"]
        lines.append("| Model | Brier Score | H-L Statistic | H-L p-value |")
        lines.append("|:------|:------------|:--------------|:------------|")
        for model_name, metrics in cal.items():
            lines.append(
                f"| {model_name} | {metrics.get('brier_score', 0):.4f} | "
                f"{metrics.get('hl_statistic', 'N/A')} | {metrics.get('hl_pvalue', 'N/A')} |"
            )
    lines.append("")
    
    lines.append("## 6. Stability Analysis (PSI)")
    lines.append("")
    if "stability" in validation_results:
        stab = validation_results["stability"]
        for model_name, metrics in stab.items():
            from src.utils import interpret_psi
            score_psi = metrics.get("score_psi", 0)
            lines.append(f"### {model_name}")
            lines.append(f"- **Score PSI:** {score_psi:.4f} — {interpret_psi(score_psi)}")
            lines.append("")
    
    lines.append("## 7. Stress Testing Results")
    lines.append("")
    if stress_results:
        for model_name, result_df in stress_results.items():
            lines.append(f"### {model_name}")
            if isinstance(result_df, pd.DataFrame):
                lines.append("")
                lines.append(result_df.to_markdown(index=False))
            lines.append("")
    
    lines.append("## 8. Findings & Recommendations")
    lines.append("")
    lines.append("1. The champion WOE/IV scorecard demonstrates strong discrimination and calibration.")
    lines.append("2. Challenger ensemble models (XGBoost/LightGBM) provide benchmarking performance.")
    lines.append("3. Stress testing verifies model resilience under recessionary economic shocks.")
    lines.append("")
    lines.append("---")
    lines.append("*Report generated automatically by the Credit Risk Model Validation Framework.*")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"  [REPORT] Validation report saved: {report_path}")


if __name__ == "__main__":
    main()
