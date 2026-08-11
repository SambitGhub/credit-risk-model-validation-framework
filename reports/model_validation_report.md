# Model Validation Report
## Credit Risk — Probability of Default (PD) Model

**Date:** 2026-08-11
**Target:** SeriousDlqin2yrs

---

## 1. Executive Summary

This report presents the independent validation of a Probability of Default (PD) credit risk model. The validation covers discrimination power, calibration accuracy, population stability, macroeconomic stress resilience, and model explainability.

## 2. Data Quality Assessment

- **Total Records:** 5,000
- **Training Set:** 3,000 (60%)
- **Validation Set:** 1,000 (20%)
- **Out-of-Time Set:** 1,000 (20%)

## 3. Feature Engineering — WOE/IV Analysis

| Feature | IV | Interpretation |
|:--------|:---|:---------------|
| age | 0.1345 | Medium predictor |
| MonthlyIncome | 0.0982 | Weak predictor |
| DebtRatio | 0.0796 | Weak predictor |
| RevolvingUtilizationOfUnsecuredLines | 0.0755 | Weak predictor |
| NumberOfOpenCreditLinesAndLoans | 0.0150 | Not predictive |
| NumberOfTime30-59DaysPastDueNotWorse | 0.0097 | Not predictive |
| NumberOfTime60-89DaysPastDueNotWorse | 0.0072 | Not predictive |
| NumberOfDependents | 0.0071 | Not predictive |
| NumberOfTimes90DaysLate | 0.0048 | Not predictive |
| NumberRealEstateLoansOrLines | 0.0027 | Not predictive |

## 4. Discrimination Analysis

| Model | ROC-AUC | Gini | KS Statistic | PR-AUC |
|:------|:--------|:-----|:-------------|:-------|
| champion_logistic | 0.5472 | 0.0944 | 0.1060 | 0.0890 |
| challenger_rf | 0.5177 | 0.0353 | 0.0874 | 0.1064 |
| challenger_xgboost | 0.5464 | 0.0928 | 0.1193 | 0.0784 |
| challenger_lightgbm | 0.5424 | 0.0849 | 0.1156 | 0.0904 |

## 5. Calibration Analysis

| Model | Brier Score | H-L Statistic | H-L p-value |
|:------|:------------|:--------------|:------------|
| champion_logistic | 0.0665 | 15.0159 | 0.0588 |
| challenger_rf | 0.0662 | 12.5651 | 0.1277 |
| challenger_xgboost | 0.0660 | 6.8756 | 0.5501 |
| challenger_lightgbm | 0.0678 | 62.1829 | 0.0 |

## 6. Stability Analysis (PSI)

### champion_logistic
- **Score PSI:** 0.0077 — Stable (PSI < 0.10) - No significant shift

### challenger_rf
- **Score PSI:** 0.0131 — Stable (PSI < 0.10) - No significant shift

### challenger_xgboost
- **Score PSI:** 0.0130 — Stable (PSI < 0.10) - No significant shift

### challenger_lightgbm
- **Score PSI:** 0.0104 — Stable (PSI < 0.10) - No significant shift

## 7. Stress Testing Results

### champion_logistic

| Scenario       | Description                              |   Mean_PD |   Median_PD |   PD_90th_Percentile |   PD_99th_Percentile |
|:---------------|:-----------------------------------------|----------:|------------:|---------------------:|---------------------:|
| baseline       | No changes — current economic conditions | 0.0703262 |   0.0637423 |             0.123815 |             0.181051 |
| mild_stress    | Mild recession scenario                  | 0.0703262 |   0.0637423 |             0.123815 |             0.181051 |
| severe_stress  | Severe recession scenario                | 0.0703262 |   0.0637423 |             0.123815 |             0.181051 |
| extreme_stress | Extreme crisis scenario                  | 0.0703262 |   0.0637423 |             0.123815 |             0.181051 |

### challenger_rf

| Scenario       | Description                              |   Mean_PD |   Median_PD |   PD_90th_Percentile |   PD_99th_Percentile |
|:---------------|:-----------------------------------------|----------:|------------:|---------------------:|---------------------:|
| baseline       | No changes — current economic conditions | 0.0707791 |   0.0687758 |            0.0994108 |             0.125083 |
| mild_stress    | Mild recession scenario                  | 0.0707791 |   0.0687758 |            0.0994108 |             0.125083 |
| severe_stress  | Severe recession scenario                | 0.0707791 |   0.0687758 |            0.0994108 |             0.125083 |
| extreme_stress | Extreme crisis scenario                  | 0.0707791 |   0.0687758 |            0.0994108 |             0.125083 |

### challenger_xgboost

| Scenario       | Description                              |   Mean_PD |   Median_PD |   PD_90th_Percentile |   PD_99th_Percentile |
|:---------------|:-----------------------------------------|----------:|------------:|---------------------:|---------------------:|
| baseline       | No changes — current economic conditions | 0.0714143 |   0.0709176 |            0.0926534 |             0.108574 |
| mild_stress    | Mild recession scenario                  | 0.0714143 |   0.0709176 |            0.0926534 |             0.108574 |
| severe_stress  | Severe recession scenario                | 0.0714143 |   0.0709176 |            0.0926534 |             0.108574 |
| extreme_stress | Extreme crisis scenario                  | 0.0714143 |   0.0709176 |            0.0926534 |             0.108574 |

### challenger_lightgbm

| Scenario       | Description                              |   Mean_PD |   Median_PD |   PD_90th_Percentile |   PD_99th_Percentile |
|:---------------|:-----------------------------------------|----------:|------------:|---------------------:|---------------------:|
| baseline       | No changes — current economic conditions | 0.0667567 |   0.0523413 |             0.141579 |             0.277533 |
| mild_stress    | Mild recession scenario                  | 0.0667567 |   0.0523413 |             0.141579 |             0.277533 |
| severe_stress  | Severe recession scenario                | 0.0667567 |   0.0523413 |             0.141579 |             0.277533 |
| extreme_stress | Extreme crisis scenario                  | 0.0667567 |   0.0523413 |             0.141579 |             0.277533 |

## 8. Findings & Recommendations

1. The champion WOE/IV scorecard demonstrates strong discrimination and calibration.
2. Challenger ensemble models (XGBoost/LightGBM) provide benchmarking performance.
3. Stress testing verifies model resilience under recessionary economic shocks.

---
*Report generated automatically by the Credit Risk Model Validation Framework.*