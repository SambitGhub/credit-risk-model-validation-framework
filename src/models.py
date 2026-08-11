import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    LOGISTIC_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    SCORECARD_PDO,
    SCORECARD_TARGET_SCORE,
    SCORECARD_TARGET_ODDS,
    MODELS_DIR,
)
from src.utils import print_section_header


def train_champion_scorecard(
    X_train_woe: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> LogisticRegression:
    p = params or LOGISTIC_PARAMS
    model = LogisticRegression(**p)
    model.fit(X_train_woe, y_train)
    return model


def convert_to_scorecard(
    model: LogisticRegression,
    woe_tables: Dict[str, pd.DataFrame],
    selected_features: List[str],
    pdo: float = SCORECARD_PDO,
    target_score: float = SCORECARD_TARGET_SCORE,
    target_odds: float = SCORECARD_TARGET_ODDS,
) -> pd.DataFrame:
    factor = pdo / np.log(2)
    offset = target_score - factor * np.log(target_odds)

    intercept = model.intercept_[0]
    coefs = dict(zip(selected_features, model.coef_[0]))

    records = []
    base_points = offset - factor * intercept
    records.append({
        "Feature": "Base",
        "Bin": "Base",
        "WOE": 0.0,
        "Coefficient": intercept,
        "Points": round(base_points, 2),
    })

    for f in selected_features:
        coef = coefs.get(f, 0.0)
        table = woe_tables.get(f)
        if table is None:
            continue
        for _, row in table.iterrows():
            woe_val = row["WOE"]
            pts = -factor * coef * woe_val
            records.append({
                "Feature": f,
                "Bin": row["Bin"],
                "WOE": round(woe_val, 4),
                "Coefficient": round(coef, 6),
                "Points": round(pts, 2),
            })

    return pd.DataFrame(records)


def train_challenger_rf(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict = None
) -> RandomForestClassifier:
    p = params or RANDOM_FOREST_PARAMS
    model = RandomForestClassifier(**p)
    model.fit(X_train, y_train)
    return model


def train_challenger_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict = None
) -> xgb.XGBClassifier:
    p = params or XGBOOST_PARAMS
    model = xgb.XGBClassifier(**p)
    model.fit(X_train, y_train)
    return model


def train_challenger_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict = None
) -> lgb.LGBMClassifier:
    p = params or LIGHTGBM_PARAMS
    model = lgb.LGBMClassifier(**p)
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train_woe: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    woe_tables: Dict[str, pd.DataFrame],
    selected_features: List[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    print_section_header("Training Champion & Challenger Models")

    models: Dict[str, Any] = {}

    print("  [RUN] Training Champion Scorecard (Logistic Regression on WOE) ...")
    lr = train_champion_scorecard(X_train_woe, y_train)
    models["champion_logistic"] = lr

    scorecard = convert_to_scorecard(lr, woe_tables, selected_features)
    scorecard.to_csv(MODELS_DIR / "scorecard.csv", index=False)
    print(f"     [SAVE] Scorecard saved ({len(scorecard)} rows)")

    print("  [RUN] Training Challenger: Random Forest ...")
    models["challenger_rf"] = train_challenger_rf(X_train_raw, y_train)

    print("  [RUN] Training Challenger: XGBoost ...")
    models["challenger_xgboost"] = train_challenger_xgboost(X_train_raw, y_train)

    print("  [RUN] Training Challenger: LightGBM ...")
    models["challenger_lightgbm"] = train_challenger_lightgbm(X_train_raw, y_train)

    print("\n  [SAVE] Saving models to disk ...")
    for name, model in models.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"     Saved {path.name}")

    return models, scorecard


def predict_probabilities(
    models: Dict[str, Any],
    X_dict: Dict[str, pd.DataFrame],
) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        if name not in X_dict:
            continue
        X = X_dict[name]
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]
        preds[name] = model.predict_proba(X)[:, 1]
    return preds
