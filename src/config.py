"""
config.py — Central Configuration for Credit Risk Model Validation
"""

import os
from pathlib import Path

# 1. Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 2. Raw Data File Selection
raw_files = list(RAW_DATA_DIR.glob("*.csv"))
if raw_files:
    RAW_DATA_FILE = raw_files[0]
else:
    RAW_DATA_FILE = RAW_DATA_DIR / "cs-training.csv"

# 3. Random Seed & Ratios
RANDOM_SEED = 42
TRAIN_RATIO = 0.60
VALID_RATIO = 0.20
OOT_RATIO = 0.20

# 4. Target Variable & Features (Will be auto-detected per CSV)
TARGET_COL = "SeriousDlqin2yrs"

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Past Due 30-59 Days",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberOfTimes90DaysLate": "Past Due 90+ Days",
    "NumberRealEstateLoansOrLines": "Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Past Due 60-89 Days",
    "NumberOfDependents": "Num Dependents",
}

FEATURES_WITH_MISSING = ["MonthlyIncome", "NumberOfDependents"]
CONTINUOUS_FEATURES = ["RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio", "MonthlyIncome"]
DISCRETE_FEATURES = [c for c in FEATURE_COLS if c not in CONTINUOUS_FEATURES]

# 5. Outliers & Scorecard
IQR_MULTIPLIER = 1.5
HARD_CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": (0.0, 2.0),
    "age": (18, 100),
    "DebtRatio": (0.0, 10.0),
    "MonthlyIncome": (0, 200_000),
    "NumberOfTime30-59DaysPastDueNotWorse": (0, 15),
    "NumberOfTimes90DaysLate": (0, 15),
    "NumberOfTime60-89DaysPastDueNotWorse": (0, 15),
    "NumberOfDependents": (0, 10),
}

WOE_INITIAL_BINS = 20
WOE_MIN_BIN_FRACTION = 0.05
IV_THRESHOLDS = {"useless": 0.02, "weak": 0.10, "medium": 0.30, "strong": 0.50}

SCORECARD_PDO = 20
SCORECARD_TARGET_SCORE = 600
SCORECARD_TARGET_ODDS = 50

LOGISTIC_PARAMS = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "random_state": RANDOM_SEED}
RANDOM_FOREST_PARAMS = {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 50, "max_features": "sqrt", "random_state": RANDOM_SEED, "n_jobs": -1}
XGBOOST_PARAMS = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 50, "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": RANDOM_SEED, "eval_metric": "auc", "use_label_encoder": False}
LIGHTGBM_PARAMS = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": RANDOM_SEED, "verbose": -1}

PSI_BINS = 10
PSI_THRESHOLDS = {"stable": 0.10, "moderate": 0.25}

STRESS_SCENARIOS = {
    "baseline": {"description": "No changes — current economic conditions"},
    "mild_stress": {"description": "Mild recession scenario"},
    "severe_stress": {"description": "Severe recession scenario"},
    "extreme_stress": {"description": "Extreme crisis scenario"},
}

PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_DPI = 150
PLOT_FIGSIZE = (10, 6)
PLOT_COLORS = {
    "primary": "#1E88E5", "secondary": "#FF6F00", "success": "#43A047", "danger": "#E53935",
    "neutral": "#78909C", "champion": "#1E88E5", "challenger_rf": "#FF6F00", "challenger_xgb": "#43A047", "challenger_lgbm": "#AB47BC",
}
