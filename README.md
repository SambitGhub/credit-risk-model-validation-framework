# Credit Risk Modeling & Independent Model Validation Framework

> A production-grade credit risk assessment pipeline implementing **Side A (Model Development)** and **Side B (Independent Model Validation)** — mirroring real-world banking governance frameworks (SR 11-7, EBA Guidelines).

## 🎯 Project Overview

This project builds an end-to-end credit risk modeling system that goes far beyond basic ML classification. It demonstrates:

1. **Credit Scorecard Development** — WOE/IV feature engineering with logistic regression baseline
2. **Champion vs. Challenger Framework** — Comparing traditional scorecards against gradient-boosted models
3. **Independent Model Validation** — Discrimination, calibration, stability (PSI), and stress testing
4. **Model Explainability** — SHAP-based global and local explanations
5. **Data Leakage Detection** — Intentional experiment demonstrating AUC inflation from leakage

## 📊 Dataset

**Home Credit Default Risk** (Kaggle Flagship) — 307,511 borrower records with 122 features (12 key features configured) predicting loan default (`TARGET` = 1 for payment difficulties, 0 otherwise).

| Feature | Description |
|:--------|:------------|
| `AMT_INCOME_TOTAL` | Borrower gross annual income |
| `AMT_CREDIT` | Total credit amount of the loan |
| `AMT_ANNUITY` | Loan annuity payment amount |
| `AMT_GOODS_PRICE` | Price of the goods for which the loan is given |
| `EXT_SOURCE_1` | Normalized score from external credit bureau 1 |
| `EXT_SOURCE_2` | Normalized score from external credit bureau 2 |
| `EXT_SOURCE_3` | Normalized score from external credit bureau 3 |
| `DAYS_BIRTH` | Client's age at application (negative days) |
| `DAYS_EMPLOYED` | Days before application client started current job |
| `CNT_CHILDREN` | Number of children the client has |
| `CNT_FAM_MEMBERS` | Number of family members |
| `DAYS_LAST_PHONE_CHANGE` | Days before application client changed phone |


## 🏗️ Project Structure

```
credit-risk-model-validation/
├── README.md
├── requirements.txt
├── run_pipeline.py              
├── data/
│   ├── raw/                   
│   └── processed/            
├── src/
│   ├── config.py               
│   ├── data_loader.py       
│   ├── preprocessing.py       
│   ├── feature_engineering.py  
│   ├── models.py               
│   ├── validation.py        
│   ├── stress_testing.py       
│   ├── explainability.py      
│   └── utils.py           
├── outputs/
│   ├── figures/               
│   └── models/             
└── reports/
    └── model_validation_report.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) dataset and place `cs-training.csv` in `data/raw/`.

```bash
# If you have Kaggle CLI configured:
kaggle competitions download -c GiveMeSomeCredit -p data/raw/
```

### 3. Run the Full Pipeline
```bash
python run_pipeline.py
```

This will execute the complete pipeline:
- Data loading, cleaning, and quality assessment
- Data leakage experiment
- WOE/IV feature engineering
- Champion (Logistic Regression Scorecard) and Challenger (RF, XGBoost, LightGBM) training
- Full validation suite (discrimination, calibration, stability)
- Macroeconomic stress testing
- SHAP explainability analysis
- Generation of all figures and the validation report

## 📈 Key Methodologies

### Weight of Evidence (WOE) & Information Value (IV)
$$WOE_i = \ln\left(\frac{\% \text{Events}_i}{\% \text{Non-Events}_i}\right)$$
$$IV = \sum_{i=1}^{N} (\% \text{Events}_i - \% \text{Non-Events}_i) \times WOE_i$$

### Credit Scorecard Conversion
$$\text{Score} = \text{Offset} + \text{Factor} \times \sum(\beta_i \times WOE_i)$$
Where: $\text{Factor} = \frac{PDO}{\ln(2)}$, $\text{Offset} = \text{TargetScore} - \text{Factor} \times \ln(\text{TargetOdds})$

### KS Statistic
$$KS = \max_s |F_{\text{good}}(s) - F_{\text{bad}}(s)|$$

### Population Stability Index (PSI)
$$PSI = \sum_{i=1}^{N} (\%\text{Actual}_i - \%\text{Expected}_i) \times \ln\left(\frac{\%\text{Actual}_i}{\%\text{Expected}_i}\right)$$

## 🧪 Model Validation Framework

| Validation Dimension | Metrics | Thresholds |
|:---------------------|:--------|:-----------|
| **Discrimination** | ROC-AUC, PR-AUC, KS, Gini | AUC > 0.70, KS > 0.30 |
| **Calibration** | Brier Score, Hosmer-Lemeshow | Brier < 0.25, H-L p > 0.05 |
| **Stability** | PSI (Score & Feature-level) | PSI < 0.10 stable, < 0.25 moderate |
| **Stress Testing** | PD shift under macro shocks | Domain judgment |
| **Explainability** | SHAP vs IV rank correlation | Spearman ρ > 0.70 |

## 📋 Technologies

- **Python 3.9+**
- pandas, numpy, scipy
- scikit-learn, XGBoost, LightGBM
- SHAP
- matplotlib, seaborn

## 👤 Author

Sambit Ranjan Rout

## 📄 License

This project is for educational and portfolio demonstration purposes.
