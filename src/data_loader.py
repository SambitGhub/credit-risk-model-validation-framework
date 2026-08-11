import pandas as pd
from typing import Optional
import sys
from pathlib import Path

# Add project root to path if needed to run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import config
from src import utils

def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw dataset from data/raw/ and auto-detect target column if needed.
    """
    if filepath is None:
        filepath = config.RAW_DATA_FILE
        
    print(f"[DATA] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    # Auto-detect target column
    target_candidates = ["TARGET", "target", "SeriousDlqin2yrs", "loan_status", "default.payment.next.month"]
    found_target = None
    for cand in target_candidates:
        if cand in df.columns:
            found_target = cand
            break
            
    if found_target:
        print(f"[INFO] Target column identified: '{found_target}'")
        config.TARGET_COL = found_target
        
        # Configure features dynamically if needed
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != found_target and "ID" not in c.upper()]
        if len(numeric_cols) > 0:
            config.FEATURE_COLS = numeric_cols[:12]
            config.CONTINUOUS_FEATURES = [c for c in config.FEATURE_COLS if df[c].nunique() > 20]
            config.DISCRETE_FEATURES = [c for c in config.FEATURE_COLS if c not in config.CONTINUOUS_FEATURES]
            config.FEATURES_WITH_MISSING = [c for c in config.FEATURE_COLS if df[c].isnull().sum() > 0]
            config.FEATURE_LABELS = {c: c.replace("_", " ").title() for c in config.FEATURE_COLS}
            print(f"  [OK] Configured {len(config.FEATURE_COLS)} features: {config.FEATURE_COLS}")
    else:
        raise KeyError(f"None of the target candidates {target_candidates} found in CSV columns: {list(df.columns)}")
        
    print(f"[OK] Data loaded successfully. Shape: {df.shape}")
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df

def generate_data_quality_report(df: pd.DataFrame) -> None:
    utils.print_section_header("Data Quality Report")
    
    dq_summary = utils.data_quality_summary(df)
    print("\n[INFO] Missing Values & Basic Stats:")
    print(dq_summary.to_string())
    
    target_col = config.TARGET_COL
    if target_col in df.columns:
        utils.print_subsection("Target Class Imbalance")
        target_counts = df[target_col].value_counts()
        target_pcts = df[target_col].value_counts(normalize=True) * 100
        
        for val in target_counts.index:
            print(f"  Class {val}: {target_counts[val]} records ({target_pcts[val]:.2f}%)")
    else:
        print(f"\n[WARN] Target column '{target_col}' not found in dataframe.")

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_column_names(df_raw)
    generate_data_quality_report(df_clean)
