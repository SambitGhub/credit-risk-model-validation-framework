import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path

# Add project root to path if needed to run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import config
from src import utils
from src import data_loader

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Impute missing values for numeric features."""
    df_out = df.copy()
    utils.print_section_header("Handling Missing Values & Anomalies")
    
    # Handle Home Credit DAYS_EMPLOYED anomaly (365243 means unemployed/retired)
    if "DAYS_EMPLOYED" in df_out.columns:
        anom_mask = df_out["DAYS_EMPLOYED"] == 365243
        if anom_mask.sum() > 0:
            df_out.loc[anom_mask, "DAYS_EMPLOYED"] = np.nan
            print(f"  [FIX] Handled {anom_mask.sum()} anomalous 365243 values in DAYS_EMPLOYED -> converted to NaN")
            
    # Numeric column imputation
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col == config.TARGET_COL:
            continue
        missing_count = df_out[col].isnull().sum()
        if missing_count > 0:
            if col in config.DISCRETE_FEATURES:
                fill_val = df_out[col].mode()[0] if not df_out[col].mode().empty else 0
            else:
                fill_val = df_out[col].median()
            df_out[col] = df_out[col].fillna(fill_val)
            print(f"  [IMPUTE] Imputed {missing_count} missing values in {col} with fill value: {fill_val}")
            
    return df_out

def cap_outliers(df: pd.DataFrame, hard_caps: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    """Apply IQR-based and domain-informed hard caps to features."""
    df_out = df.copy()
    if hard_caps is None:
        hard_caps = config.HARD_CAPS
        
    utils.print_section_header("Capping Outliers")
    
    for feature in config.CONTINUOUS_FEATURES + config.DISCRETE_FEATURES:
        if feature not in df_out.columns:
            continue
            
        capped_count = 0
        if feature in hard_caps:
            min_val, max_val = hard_caps[feature]
            lower_mask = df_out[feature] < min_val
            upper_mask = df_out[feature] > max_val
            capped_count += lower_mask.sum() + upper_mask.sum()
            df_out.loc[lower_mask, feature] = min_val
            df_out.loc[upper_mask, feature] = max_val
        elif feature in config.CONTINUOUS_FEATURES:
            Q1 = df_out[feature].quantile(0.25)
            Q3 = df_out[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - config.IQR_MULTIPLIER * IQR
            upper_bound = Q3 + config.IQR_MULTIPLIER * IQR
            lower_mask = df_out[feature] < lower_bound
            upper_mask = df_out[feature] > upper_bound
            capped_count += lower_mask.sum() + upper_mask.sum()
            df_out.loc[lower_mask, feature] = lower_bound
            df_out.loc[upper_mask, feature] = upper_bound
            
        if capped_count > 0:
            print(f"  [CAP] Capped {capped_count} outliers in {feature}")
            
    return df_out

def create_train_val_oot_split(
    df: pd.DataFrame, 
    train_ratio: float = 0.60, 
    val_ratio: float = 0.20, 
    oot_ratio: float = 0.20, 
    target_col: Optional[str] = None, 
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split of the dataset into Train, Validation, and OOT sets."""
    utils.print_section_header("Creating Data Splits")
    
    # Read target column dynamically from config if not explicitly supplied
    if target_col is None or target_col not in df.columns:
        target_col = config.TARGET_COL
        
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")
        
    temp_ratio = train_ratio + val_ratio
    oot_test_size = oot_ratio / (temp_ratio + oot_ratio)
    
    # First split: Extract OOT set
    df_temp, oot_df = train_test_split(
        df, 
        test_size=oot_test_size, 
        stratify=df[target_col], 
        random_state=random_seed
    )
    
    # Second split: Train vs Validation
    val_test_size = val_ratio / temp_ratio
    train_df, val_df = train_test_split(
        df_temp, 
        test_size=val_test_size, 
        stratify=df_temp[target_col], 
        random_state=random_seed
    )
    
    splits = {"Train": train_df, "Validation": val_df, "OOT": oot_df}
    for name, split_df in splits.items():
        dr = split_df[target_col].mean() * 100
        print(f"  [SPLIT] {name} Set: {len(split_df)} rows | Default Rate: {dr:.2f}%")
        
    return train_df, val_df, oot_df

def run_leakage_experiment(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
    utils.print_section_header("Data Leakage Experiment")
    
    past_due_features = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate"
    ]
    existing_past_due = [f for f in past_due_features if f in X_train.columns]
    
    if not existing_past_due:
        print("  [INFO] No past-due features found for experiment (skipping).")
        return {"auc_with_leakage": "N/A", "auc_without_leakage": "N/A", "auc_inflation": "N/A"}
        
    model_leak = LogisticRegression(**config.LOGISTIC_PARAMS)
    model_leak.fit(X_train, y_train)
    preds_leak = model_leak.predict_proba(X_val)[:, 1]
    auc_leak = roc_auc_score(y_val, preds_leak)
    
    X_train_no_leak = X_train.drop(columns=existing_past_due)
    X_val_no_leak = X_val.drop(columns=existing_past_due)
    
    model_no_leak = LogisticRegression(**config.LOGISTIC_PARAMS)
    model_no_leak.fit(X_train_no_leak, y_train)
    preds_no_leak = model_no_leak.predict_proba(X_val_no_leak)[:, 1]
    auc_no_leak = roc_auc_score(y_val, preds_no_leak)
    
    auc_inflation = auc_leak - auc_no_leak
    print(f"  [LEAKAGE] AUC With Potential Leakage Features: {auc_leak:.4f}")
    print(f"  [CLEAN]   AUC Without Leakage Features:       {auc_no_leak:.4f}")
    print(f"  [WARN]    AUC Inflation:                     +{auc_inflation:.4f}")
    
    return {
        "auc_with_leakage": round(auc_leak, 4),
        "auc_without_leakage": round(auc_no_leak, 4),
        "auc_inflation": round(auc_inflation, 4),
    }

def preprocess_pipeline(filepath: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    utils.print_section_header("Starting Preprocessing Pipeline")
    
    df = data_loader.load_raw_data(filepath)
    df = data_loader.clean_column_names(df)
    df = handle_missing_values(df)
    df = cap_outliers(df, config.HARD_CAPS)
    
    train_df, val_df, oot_df = create_train_val_oot_split(
        df,
        config.TRAIN_RATIO,
        config.VALID_RATIO,
        config.OOT_RATIO,
        config.TARGET_COL,
        config.RANDOM_SEED
    )
    
    print("\n  [OK] Preprocessing Pipeline Complete!")
    return {
        "train": train_df,
        "val": val_df,
        "oot": oot_df
    }

if __name__ == "__main__":
    splits = preprocess_pipeline()
