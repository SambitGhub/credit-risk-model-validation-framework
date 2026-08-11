import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from src.config import FIGURES_DIR, PLOT_COLORS
from src.utils import set_plot_style, save_figure, print_section_header

def apply_stress_scenario(df: pd.DataFrame, scenario_config: Dict[str, Any]) -> pd.DataFrame:
    stressed_df = df.copy()
    
    for key, value in scenario_config.items():
        if key.endswith('_multiplier'):
            feature = key.replace('_multiplier', '')
            if feature in stressed_df.columns:
                stressed_df[feature] = stressed_df[feature] * value
                
    return stressed_df

def run_stress_test(
    model: Any, X_baseline: pd.DataFrame, scenarios: Dict[str, Dict[str, Any]], 
    feature_names: List[str], model_name: str = 'Model'
) -> pd.DataFrame:
    results = []
    
    for scenario_name, config in scenarios.items():
        stressed_X = apply_stress_scenario(X_baseline, config)
        
        if hasattr(model, "predict_proba"):
            pds = model.predict_proba(stressed_X)[:, 1]
        else:
            pds = model.predict(stressed_X)
            
        res = {
            'Scenario': scenario_name,
            'Description': config.get('description', ''),
            'Mean_PD': np.mean(pds),
            'Median_PD': np.median(pds),
            'PD_90th_Percentile': np.percentile(pds, 90),
            'PD_99th_Percentile': np.percentile(pds, 99)
        }
        results.append(res)
        
    return pd.DataFrame(results)

def plot_stress_results(stress_summary: pd.DataFrame, model_name: str, save: bool = True):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = stress_summary['Scenario']
    mean_pds = stress_summary['Mean_PD']
    
    colors = [PLOT_COLORS['primary'] if s == 'baseline' else PLOT_COLORS['danger'] for s in scenarios]
    bars = ax.bar(scenarios, mean_pds, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    ax.set_ylabel('Mean Predicted Probability of Default (PD)')
    ax.set_title(f'Stress Testing Results — Mean PD by Scenario ({model_name})')
    plt.xticks(rotation=45, ha='right')
    
    if save:
        save_figure(fig, f'stress_test_results_{model_name}.png'.replace(' ', '_').lower())
    else:
        plt.show()

def run_all_stress_tests(
    models: Dict[str, Any], X_dict: Dict[str, pd.DataFrame], 
    scenarios: Dict[str, Dict[str, Any]], feature_names: List[str]
) -> Dict[str, pd.DataFrame]:
    print_section_header("Running Macroeconomic Stress Testing")
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"  [RUN] Stress testing {model_name} ...")
        X_baseline = X_dict[model_name]
        
        summary_df = run_stress_test(model, X_baseline, scenarios, feature_names, model_name)
        all_results[model_name] = summary_df
        plot_stress_results(summary_df, model_name, save=True)
        
    print("  [OK] Stress testing complete. Plots saved to outputs/figures/.")
    return all_results
