# 1. Project Title
**Telecommunications Customer Churn Prediction & Data-Driven Retention System**

---

# 2. Project Overview
This project implements an end-to-end supervised machine learning pipeline to analyze, predict, and reduce customer churn across an 8,000-subscriber telecommunications dataset. Spanning 15 structured project milestones (Days 36 to 50), the workflow encompasses business problem framing, data cleaning, exploratory data analysis (EDA), statistical hypothesis testing, domain feature engineering, leakage-free Scikit-learn preprocessing, model benchmarking (Logistic Regression, Decision Tree, Random Forest), hyperparameter tuning via Stratified Cross-Validation, model interpretation, and tiered customer retention recommendations.

---

# 3. Business Problem
The telecommunications company experiences an **18.00% customer churn rate** (**1,440** churned subscribers out of **8,000** tracked accounts), resulting in **$126,843.12 per month** (**18.70%** of total $678,315.86 monthly recurring billing, or **$1.52M annualized**) in lost recurring revenue. Because acquiring new telecommunications customers costs significantly more than retaining existing subscribers, leadership needs an accurate, interpretable early-warning system to identify high-risk customers prior to cancellation and deploy targeted retention interventions.

---

# 4. Objectives
1. **Quantify Historical Churn & Revenue Exposure:** Establish baseline attrition (18.0%) and recurring revenue at risk ($126,843.12/month).
2. **Identify Behavioral & Contractual Patterns:** Segment subscribers by tenure, contract length, service adoption, billing method, and support friction.
3. **Validate Churn Drivers Statistically:** Apply Chi-Square, Welch's t-test, and Mann–Whitney U tests ($lpha = 0.05$) with effect sizes (Cohen's $d$, Cramér's $V$).
4. **Engineer Leakage-Free Domain Features:** Construct lifecycle tenure cohorts, spending tiers, service interaction counters, and a composite 0–100 `Engagement_Score`.
5. **Build Reproducible Preprocessing & Modeling Pipelines:** Prevent data leakage by fitting all imputers, scalers, encoders, and models strictly on stratified training data.
6. **Evaluate & Tune Multiple Classifiers:** Benchmark Logistic Regression, Decision Tree, and Random Forest before and after hyperparameter tuning using Recall, Precision, F1-Score, and ROC-AUC.
7. **Deliver Actionable Retention Playbooks:** Translate predicted churn probabilities into validation-calibrated Low, Medium, and High Risk tiers.

---

# 5. Dataset Description
- **Source File:** `data/raw/customer_churn.csv` (**8,000 rows × 27 columns**, 0 duplicate rows, 8,000 unique `CustomerID`s).
- **Target Variable (`Churn`):** Binary class label — `No` (**6,560** retained subscribers / **82.0%**) and `Yes` (**1,440** churned subscribers / **18.0%**), representing a **4.56 : 1** class imbalance.
- **Predictor Categories (25 raw + 9 derived = 34 features):**
  - *Demographics:* `Gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - *Account & Contract:* `TenureMonths` (0–72), `ContractType` (`Month-to-month`, `One year`, `Two year`), `PaperlessBilling`, `PaymentMethod`
  - *Services & Add-ons:* `PhoneService`, `MultipleLines`, `InternetService` (`Fiber optic`, `DSL`, `5G Home`, `No`), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - *Billing, Usage & Friction:* `MonthlyCharges` ($18.00–$127.17), `TotalCharges` ($0.00–$8,768.90), `SupportCallsLast90Days` (0–7), `ComplaintRaised`, `AvgMonthlyDataUsageGB`, `AvgMonthlyCalls`, `SatisfactionScore` (1–5), `DaysSinceLastContact`

---

# 6. Tools and Technologies
- **Programming Language:** Python 3.10+
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Statistical Testing:** `scipy.stats` (Chi-Square, Welch's t-test, Mann–Whitney U)
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Serialization:** `scikit-learn` (`ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, `RandomizedSearchCV`, `StratifiedKFold`), `joblib`
- **Reporting & Presentation:** Jupyter Notebook, `openpyxl`, `python-docx`, `reportlab`, `python-pptx`, Git & GitHub

---

# 7. Project Structure
```text
Customer_Churn_Prediction/
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv
│   └── processed/
│       └── cleaned_customer_churn.csv
├── notebooks/
│   ├── 01_Dataset_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Churn_EDA.ipynb
│   ├── 04_Statistical_Analysis.ipynb
│   ├── 05_Feature_Engineering.ipynb
│   ├── 06_Preprocessing.ipynb
│   ├── 07_Train_Test_Split.ipynb
│   ├── 08_Logistic_Regression.ipynb
│   ├── 09_Decision_Tree.ipynb
│   ├── 10_Random_Forest.ipynb
│   ├── 11_Model_Evaluation.ipynb
│   ├── 12_Hyperparameter_Tuning.ipynb
│   └── 13_Feature_Importance.ipynb
├── src/
│   ├── clean_data.py
│   ├── preprocessing.py
│   └── predict_churn.py
├── models/
│   └── final_model.joblib
├── reports/
│   ├── Final_Report.pdf
│   └── Model_Comparison.csv
├── presentation/
│   └── Customer_Churn_Presentation.pptx
├── README.md
└── requirements.txt
```

---

# 8. Data Cleaning Process
- **Missing Values Audit (464 missing cells / 0.21% of dataset):**
  - `SatisfactionScore`: **202** missing (2.53%) $ightarrow$ imputed with training median (`3`) and cast to `int64`.
  - `AvgMonthlyDataUsageGB`: **101** missing (1.26%) $ightarrow$ deterministically set to `0.0` GB where `InternetService == 'No'` (10 rows) and imputed with active-internet median (`41.5` GB) for remaining 91 rows.
  - `DaysSinceLastContact`: **99** missing (1.24%) $ightarrow$ imputed with median (`19` days) and cast to `int64`.
  - `OnlineBackup`: **62** missing (0.78%) $ightarrow$ deterministically set to `'No internet service'` where `InternetService == 'No'` (7 rows) and imputed with mode (`'No'`) for remaining 55 rows.
- **Duplicates, Types & Outliers:** Confirmed **0** exact duplicate rows and **8,000** unique `CustomerID`s; trimmed whitespace across string columns; verified all statistical IQR outliers represent legitimate customer plan tiers (e.g., phone-only plans at $18–$47/mo and 72-month Fiber bundles up to $8,768.90), preserving **100% of records (8,000 × 27)** in `data/processed/cleaned_customer_churn.csv`.

---

# 9. Exploratory Data Analysis
- **Contract Commitment:** `Month-to-month` subscribers (55.5% of customers) account for **85.8% of all churn** (`27.83%` churn rate vs. `7.41%` for `One year` and `3.72%` for `Two year`).
- **Service Friction & Satisfaction:** Subscribers who raised a complaint (`ComplaintRaised = Yes`) churn at **49.42%** vs. **15.84%** for non-complainants (**3.12× lift**). Churned customers report significantly lower satisfaction (mean **2.85** vs. **3.56**, $p = 7.17 	imes 10^{-135}$, Cohen's $d = -0.78$).
- **Tenure & Compounding Interactions:** Churned customers leave earlier (median **19.0 months** vs. **27.0 months**). Combining `Month-to-month` with `ComplaintRaised = Yes` or `SatisfactionScore <= 2` produces a **54.34% churn rate** (`86.21%` when `ComplaintRaised = Yes` and `SatisfactionScore = 1`).
- **Demographic Neutrality:** `Gender` ($p = 0.559$), `PhoneService` ($p = 0.245$), and `AvgMonthlyCalls` ($p = 0.879$) have no statistically significant association with churn.

---

# 10. Feature Engineering
Nine leakage-free domain features were constructed prior to modeling:
1. `Tenure_Group`: Lifecycle cohorts (`New (0-12m)`: 27.07% churn, `Medium (13-36m)`: 18.33%, `Long-term (37m+)`: 11.29%).
2. `Spending_Category`: Monthly charge tiers (`Low Spending (<=50)`: 11.29% churn, `Medium Spending (50-95)`: 17.70%, `High Spending (>95)`: 20.57%).
3. `Avg_Historical_Monthly_Spend`: `TotalCharges / TenureMonths` (decouples spend rate from $r = 0.91$ tenure collinearity).
4. `Billing_Drift_Ratio`: `MonthlyCharges / Avg_Historical_Monthly_Spend` (flags recent bill increases).
5. `Data_GB_Per_Dollar`: `AvgMonthlyDataUsageGB / MonthlyCharges` (measures internet data value efficiency).
6. `Active_Addon_Count`: Count of active optional security, backup, protection, support, and streaming add-ons (`0–6`).
7. `Total_Service_Interactions`: `SupportCallsLast90Days + (ComplaintRaised == 'Yes')` (churn escalates monotonically from `11.84%` at 0 interactions to `57.89%+` at 4+).
8. `Engagement_Score` (`0–100`): Weighted composite index combining Tenure Loyalty (`25 pts`), Add-on Adoption (`25 pts`), Satisfaction (`30 pts`), Contact Recency (`10 pts`), and Auto-Pay Stability (`10 pts`).
9. `High_Risk_Friction_Flag`: Binary indicator for `Month-to-month` subscribers with either `ComplaintRaised == 'Yes'` or `SatisfactionScore <= 2` (`54.34%` churn vs. `12.58%` unflagged).

---

# 11. Machine Learning Workflow
1. **Stratified Train-Test Split:** Partitioned data **80% training (6,400 rows)** and **20% holdout test (1,600 rows)** using `stratify=y` and `random_state=42`, preserving the exact **18.00%** churn ratio in both splits.
2. **End-to-End Scikit-Learn Pipeline (`models/final_model.joblib`):** Combines `TelecomDomainTransformer` (domain cleaning + 9 engineered features) with a `ColumnTransformer` (`SimpleImputer(strategy='median')` + `StandardScaler()` for 16 numeric features; `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` for 18 categorical features $ightarrow$ **67 model-ready features**) and the classifier.
3. **Hyperparameter Optimization:** Tuned all candidate models exclusively via **5-fold `StratifiedKFold` Cross-Validation** on `X_train` optimizing `scoring='f1'`, keeping `X_test` untouched until final single-pass evaluation.

---

# 12. Model Comparison
All models evaluated on the unseen **1,600-customer holdout test set**:

| Model & Stage | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline 0.50)** | **0.8600** | 0.7353 | 0.3472 | 0.4717 | **0.8568** | 1,276 | 36 | 188 | 100 |
| **Decision Tree (Unpruned Depth 25)** | 0.7844 | 0.4034 | 0.4132 | 0.4082 | 0.6395 | 1,136 | 176 | 169 | 119 |
| **Random Forest (Baseline 100 Trees)** | 0.8481 | **0.7473** | 0.2361 | 0.3588 | 0.8254 | **1,289** | **23** | 220 | 68 |
| **Decision Tree (Tuned: depth=5, leaf=30)** | 0.7525 | 0.3916 | 0.6771 | 0.4962 | 0.7811 | 1,009 | 303 | 93 | 195 |
| **Random Forest (Tuned: 300 trees, depth=12)** | 0.8281 | 0.5206 | 0.5694 | 0.5439 | 0.8331 | 1,161 | 151 | 124 | 164 |
| **Logistic Regression (FINAL TUNED)** | 0.7856 | 0.4421 | **0.7292** | **0.5505** | 0.8498 | 1,047 | 265 | **78** | **210** |

---

# 13. Evaluation Metrics
Because the dataset exhibits an **82.0% / 18.0% class imbalance**, raw accuracy alone is misleading (predicting `No` for every subscriber yields `82.0%` accuracy with `0.0%` churn detection). Models were evaluated on:
- **ROC-AUC:** Threshold-independent rank discrimination between departing and retained customers.
- **Recall (Sensitivity):** Proportion of actual churners caught (`TP / (TP + FN)`)—critical because missing a churner sacrifices ~$1,015/year in average recurring revenue.
- **Precision:** Proportion of flagged churners who actually churn (`TP / (TP + FP)`)—controls promotional discount spend.
- **F1-Score:** Harmonic mean balancing Recall and Precision under class imbalance.

---

# 14. Final Model
- **Selected Model:** **Tuned Logistic Regression (`C = 0.01`, `class_weight = 'balanced'`, `max_iter = 1000`)** serialized with its complete domain-feature and preprocessing pipeline in `models/final_model.joblib`.
- **Selection Rationale:** Delivers the highest holdout **F1-Score (`0.5505`)**, highest **Recall (`72.92%` — catching `210` of `288` actual churners, more than double the `100` caught by the baseline)**, and strongest **ROC-AUC (`0.8498`)**, while providing transparent, auditable log-odds weights for business teams.

---

# 15. Important Features
- **Top Positive Churn Drivers (Tuned Logistic Regression Odds Ratios & Random Forest Importances):**
  1. `ContractType_Month-to-month` (Log-Odds `+0.891`, **2.44× odds**; RF Importance `#1` at `0.1125`)
  2. `Total_Service_Interactions` (Log-Odds `+0.306`, **1.36× odds** per $+1	ext{ SD}$)
  3. `ComplaintRaised_Yes` (Log-Odds `+0.225`, **1.25× odds**)
  4. `High_Risk_Friction_Flag` (Log-Odds `+0.193`, **1.21× odds**; RF Importance `#3` at `0.0649`)
  5. `PaymentMethod_Electronic check` (Log-Odds `+0.182`, **1.20× odds**)
- **Top Retention Drivers (Protective Factors):**
  1. `ContractType_Two year` (Log-Odds `-0.568`, **0.57× odds** — reduces churn odds by 43%)
  2. `SatisfactionScore` (Log-Odds `-0.352`, **0.70× odds** per $+1	ext{ SD}$)
  3. `Engagement_Score` (Log-Odds `-0.333`, **0.72× odds**; RF Importance `#2` at `0.1046`)
  4. `TenureMonths` (Log-Odds `-0.191`, **0.83× odds**)

---

# 16. Results and Business Interpretation
Using 5-fold out-of-fold validation calibration on `X_train`, subscriber churn probabilities segment the holdout customer base into three actionable tiers:
- **High Risk (`p >= 0.65` | `16.8%` of customers | `59.11%` empirical churn):** Captures **55.2% of all churners** (`159/288`) within just `16.8%` of the customer base (**3.28× lift**). *Action:* Proactive account manager outreach, 15% loyalty discount for upgrading to a 12-month contract, and priority technical escalation for open complaints.
- **Medium Risk (`0.35 <= p < 0.65` | `30.4%` of customers | `18.93%` empirical churn):** *Action:* Automated digital nudges—$5/month bill credit to switch from Electronic Check to Auto-Pay and 60-day free `TechSupport` / `OnlineSecurity` trials.
- **Low Risk (`p < 0.35` | `52.8%` of customers | `4.38%` empirical churn):** *Action:* Zero promotional discount spend; maintain standard self-service engagement to protect operating margins.

---

# 17. Limitations
1. **Correlation Is Not Causation:** Model coefficients and feature importances reflect observational statistical patterns rather than proven causal levers; retention offers must be validated via randomized A/B holdout tests.
2. **Multicollinearity Across Correlated Features:** Strong correlation between `TenureMonths` and `TotalCharges` ($r = 0.91$) splits Random Forest importance across correlated variables and shrinks individual L2 coefficients.
3. **Probabilistic Uncertainty:** A high churn probability indicates elevated cohort risk (~59.1%), not certainty for an individual subscriber.
4. **Synthetic Educational Dataset:** This 8,000-record dataset was synthetically generated for educational modeling practice; production rollout requires retraining on audited live telecom CRM logs with verified pre-churn timestamps.

---

# 18. How to Run the Project
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Data Cleaning & Preprocessing Scripts:**
   ```bash
   python src/clean_data.py
   python src/preprocessing.py
   ```
3. **Generate Predictions Using the Saved End-to-End Model (`models/final_model.joblib`):**
   ```bash
   python src/predict_churn.py
   ```
   Or import directly in Python:
   ```python
   import sys
   sys.path.append("src")
   from predict_churn import predict_customers
   import pandas as pd

   df_new = pd.read_csv("data/raw/customer_churn.csv").head(5)
   predictions = predict_customers(df_new, model_path="models/final_model.joblib")
   print(predictions)
   ```

---

# 19. Future Scope
- **Gradient Boosting Ensembles:** Benchmark LightGBM, XGBoost, and CatBoost with calibrated probabilities (Platt scaling / isotonic regression).
- **SHAP Local Explainability:** Generate per-customer SHAP waterfall explanations so retention agents see the exact top 3 reasons each subscriber was flagged.
- **Uplift Modeling & Cost-Sensitive ROI Optimization:** Train uplift models on A/B retention campaign outcomes to target *persuadable* churners rather than customers who will leave or stay regardless of discounts.
- **Automated Drift Monitoring & REST API Deployment:** Wrap `predict_customers()` in a FastAPI microservice with scheduled quarterly drift alerts on feature distributions and recall/precision calibration.
