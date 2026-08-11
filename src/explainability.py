import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List

from src.config import FIGURES_DIR, PLOT_COLORS
from src.utils import set_plot_style, save_figure, print_section_header

def compute_shap_values(model: Any, X: pd.DataFrame, model_type: str = 'tree') -> shap.Explanation:
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer(X)
    else:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]
        
    return shap_values

def plot_shap_summary(shap_values: shap.Explanation, X: pd.DataFrame, model_name: str, save: bool = True):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.summary_plot(shap_values, X, show=False)
    
    plt.title(f'SHAP Summary Plot — {model_name}')
    plt.tight_layout()
    
    if save:
        save_figure(plt.gcf(), f'shap_summary_{model_name}.png'.replace(' ', '_').lower(), tight=False)
    else:
        plt.show()

def plot_shap_importance(shap_values: shap.Explanation, X: pd.DataFrame, model_name: str, save: bool = True):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    
    plt.title(f'SHAP Feature Importance — {model_name}')
    plt.tight_layout()
    
    if save:
        save_figure(plt.gcf(), f'shap_importance_{model_name}.png'.replace(' ', '_').lower(), tight=False)
    else:
        plt.show()

def plot_shap_waterfall(shap_values: shap.Explanation, idx: int, model_name: str, save: bool = True):
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_values[idx], show=False)
    
    plt.title(f'SHAP Local Explanation (Observation {idx}) — {model_name}')
    plt.tight_layout()
    
    if save:
        save_figure(plt.gcf(), f'shap_waterfall_{model_name}_idx{idx}.png'.replace(' ', '_').lower(), tight=False)
    else:
        plt.show()

def plot_shap_dependence(shap_values: shap.Explanation, X: pd.DataFrame, feature: str, model_name: str, save: bool = True):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    shap.dependence_plot(feature, shap_values.values, X, show=False, ax=ax)
    
    plt.title(f'SHAP Dependence Plot: {feature} — {model_name}')
    plt.tight_layout()
    
    if save:
        save_figure(plt.gcf(), f'shap_dependence_{feature}_{model_name}.png'.replace(' ', '_').lower(), tight=False)
    else:
        plt.show()

def compare_shap_vs_iv(shap_importance: pd.Series, iv_summary: pd.DataFrame) -> pd.DataFrame:
    shap_df = shap_importance.reset_index()
    shap_df.columns = ['Feature', 'Mean_Abs_SHAP']
    shap_df['SHAP_Rank'] = shap_df['Mean_Abs_SHAP'].rank(ascending=False).astype(int)
    
    iv_df = iv_summary.copy()
    if 'IV' in iv_df.columns:
        iv_df['IV_Rank'] = iv_df['IV'].rank(ascending=False).astype(int)
    else:
        iv_df['IV_Rank'] = range(1, len(iv_df) + 1)
        
    merged = pd.merge(shap_df, iv_df[['Feature', 'IV', 'IV_Rank']], on='Feature', how='inner')
    merged['Rank_Diff'] = np.abs(merged['SHAP_Rank'] - merged['IV_Rank'])
    
    spearman_corr = merged[['SHAP_Rank', 'IV_Rank']].corr(method='spearman').iloc[0, 1]
    print(f"  [STAT] Spearman Rank Correlation between SHAP and IV: {spearman_corr:.4f}")
    
    return merged.sort_values('SHAP_Rank').reset_index(drop=True)

def run_explainability_analysis(
    models: Dict[str, Any], X_dict: Dict[str, pd.DataFrame], 
    iv_summary: pd.DataFrame, n_sample: int = 1000
) -> Dict[str, Any]:
    print_section_header("Running Explainability Analysis (SHAP)")
    
    results: Dict[str, Any] = {}
    last_comparison = None
    
    for model_name, model in models.items():
        if model_name not in X_dict:
            continue

        print(f"  [RUN] Explaining {model_name} ...")
        
        X_full = X_dict[model_name]
        if len(X_full) > n_sample:
            X_sample = X_full.sample(n=n_sample, random_state=42)
        else:
            X_sample = X_full
            
        model_type = 'tree'
        name_lower = model_name.lower()
        if any(kw in name_lower for kw in ('logistic', 'lr', 'linear', 'champion')):
            model_type = 'linear'
            
        try:
            shap_values = compute_shap_values(model, X_sample, model_type=model_type)
            
            plot_shap_summary(shap_values, X_sample, model_name, save=True)
            plot_shap_importance(shap_values, X_sample, model_name, save=True)
            plot_shap_waterfall(shap_values, 0, model_name, save=True)
            
            vals = np.abs(shap_values.values).mean(0)
            top_feature = X_sample.columns[np.argmax(vals)]
            plot_shap_dependence(shap_values, X_sample, top_feature, model_name, save=True)
            
            shap_importance = pd.Series(vals, index=X_sample.columns)
            comparison_df = compare_shap_vs_iv(shap_importance, iv_summary)
            last_comparison = comparison_df
            
            results[model_name] = {
                'shap_importance': shap_importance,
                'comparison_df': comparison_df
            }
            
        except Exception as e:
            print(f"  [WARN] Error running SHAP for {model_name}: {e}")
    
    results["shap_iv_comparison"] = last_comparison if last_comparison is not None else pd.DataFrame()
            
    print("  [OK] Explainability analysis complete. Plots saved to outputs/figures/.")
    return results
