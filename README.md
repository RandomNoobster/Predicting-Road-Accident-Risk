# Predicting Road Accident Risk
We attempt to solve this competition from Kaggle: https://www.kaggle.com/competitions/playground-series-s5e10/overview

You are absolutely right. In the context of software engineering and our course setup, "Module" usually refers to a Python file (like the `util.py` we use in the lectures).

To avoid confusion, let's restructure the plan into **Work Packages (WP)**. A Work Package is a logical grouping of tasks that one student can own.

Here is the step-by-step plan for 3 students, strictly following the **"AI in Industry"** curriculum logic (starting simple, adding complexity, focusing on value).

### **Project Architecture**

*   **Student 1 (Data & Baseline):** Responsible for the "Order" (from *Order and Chaos* lecture). Sets up the pipeline, cleans data, and builds the interpretable baseline.
*   **Student 2 (Advanced Modeling):** Responsible for the "Performance." Uses non-linear models and ensembles to maximize accuracy.
*   **Student 3 (Value & Decisions):** Responsible for the "Industry" aspect. Explains the model (SHAP) and optimizes decisions (Cost Models).

---

### **Step-by-Step Implementation Plan**

#### **Work Package 1: Data Pipeline & Linear Baseline (Student 1)**

**Step 1.1: Project Setup & Data Loading**
*   **Action:** Create the folder structure defined in the *Methods and Tools* lecture (`data/`, `notebooks/`, `util/`). Create a `load_data` function in `util.py`.
*   **Curriculum Reference:** *Methods and Tools* (managing reproducibility).

**Step 1.2: Exploratory Data Analysis (EDA)**
*   **Action:** Analyze feature distributions. Check for class imbalance (accidents are likely rarer than non-accidents).
*   **Curriculum Reference:** *Biomedical Data Analysis* (using histograms and correlation matrices to understand the "data generation process").

**Step 1.3: Preprocessing & Standardization**
*   **Action:** Handle missing values (imputation) and encode categorical variables (One-Hot). Standardize numerical features using `StandardScaler`.
*   **Justification:** Linear models *require* standardized data to converge and for coefficients to be comparable (*Non-Linear Models* lecture).

**Step 1.4: The Baseline Model (Lasso)**
*   **Action:** Train a Logistic Regression with L1 regularization (Lasso).
*   **Justification:** We follow **Occam's Razor** (*Anomaly Detection* lecture). We start with the simplest explainable model. L1 regularization performs implicit feature selection (*Non-Linear Models* lecture).

---

#### **Work Package 2: Non-Linear Modeling & Ensembles (Student 2)**

**Step 2.1: Gradient Boosted Trees (XGBoost)**
*   **Action:** Train an XGBoost classifier. Use `GridSearchCV` to tune `max_depth` and `n_estimators`.
*   **Justification:** The curriculum states that real-world data often contains non-linearities and interactions that linear models miss (*Non-Linear Models* lecture). XGBoost is the standard surrogate for tabular data.

**Step 2.2: Feature Importance Analysis**
*   **Action:** Compare "Gain" importance vs. "Permutation Importance".
*   **Justification:** We need to verify if the model is relying on spurious correlations. Permutation importance is more robust than simple split-counting (*All Relevant Feature Selection* lecture).

**Step 2.3: Neuro-Probabilistic Model (Optional/Advanced)**
*   **Action:** Train a simple Neural Network (MLP).
*   **Justification:** To check if a different inductive bias improves results. If the output needs to be a probability distribution rather than a point estimate, this sets the stage for *Neuro-Probabilistic Models*.

---

#### **Work Package 3: Explainability & Decision Making (Student 3)**

**Step 3.1: Post-Hoc Explanation (SHAP)**
*   **Action:** Use KernelSHAP or TreeSHAP to explain specific predictions. Generate Waterfall plots for high-risk examples.
*   **Justification:** Accuracy is not enough. We need to understand *why* a specific road or condition is flagged as high risk (*Additive Feature Attribution* lecture).

**Step 3.2: Defining a Cost Model**
*   **Action:** Define a cost matrix. Example:
    *   Cost of **False Positive** (Patrol sent, nothing happens): 1 unit.
    *   Cost of **False Negative** (No patrol, accident occurs): 50 units.
*   **Justification:** We must move from "Loss" (MSE/LogLoss) to "Cost" (Business impact). (*Anomaly Detection* and *Prediction and Optimization* lectures).

**Step 3.3: Threshold Optimization**
*   **Action:** finding the optimal probability threshold that minimizes the cost defined in 3.2 on the validation set.
*   **Justification:** Standard classifiers use 0.5 as a threshold. In industry, this is rarely optimal. We must tune this based on the specific application risks (*Anomaly Detection* lecture).

---

### **The Code Solution**

Here is the code structure. I have used our `util` module philosophy.

#### **1. The Utility Module (`util/util.py`)**
We encapsulate reusable logic here to keep notebooks clean.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def load_data(path):
    # Standard loading logic
    return pd.read_csv(path)

def calculate_cost(y_true, y_pred_prob, threshold, c_fp, c_fn):
    """
    Calculates business cost based on the Anomaly Detection lecture logic.
    c_fp: Cost of False Positive (False Alarm)
    c_fn: Cost of False Negative (Missed Accident)
    """
    preds = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    # We assume True Negatives and True Positives have 0 extra cost 
    # (or we incorporate the gain of TP as negative cost)
    total_cost = (fp * c_fp) + (fn * c_fn)
    return total_cost

def opt_threshold(y_true, y_pred_prob, c_fp, c_fn):
    """
    Line search for threshold optimization.
    """
    thresholds = np.linspace(0, 1, 101)
    costs = [calculate_cost(y_true, y_pred_prob, t, c_fp, c_fn) for t in thresholds]
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]
```

#### **2. The Main Notebook (Integrated Solution)**

```python
# ============================================================
# Notebook setup
# ============================================================
%load_ext autoreload
%autoreload 2
from util import util
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

# ============================================================
# WP1: Data & Baseline (Student 1)
# ============================================================

# 1. Load
data = util.load_data('../data/playground-series-s5e10/train.csv')
target_col = 'Accident_Risk' # Adjust based on actual column name

# 2. Preprocessing
# Simple encoding for the baseline. 
# In the curriculum, we emphasized that Linear Models need scaling.
X = data.drop(columns=['id', target_col])
y = data[target_col]

# Detect numeric/categorical
num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# One-hot encoding for categoricals
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale (Crucial for Lasso)
scaler = StandardScaler()
X_train_s = X_train.copy()
X_val_s = X_val.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val_s[num_cols] = scaler.transform(X_val[num_cols])

# 3. Lasso Baseline
# Reference: "Non-Linear Models". We use L1 penalty to sparsify features.
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lasso.fit(X_train_s, y_train)

print(f"Baseline AUC: {roc_auc_score(y_val, lasso.predict_proba(X_val_s)[:,1]):.3f}")

# Visualize Coefficients (Interpretation)
coeffs = pd.Series(lasso.coef_[0], index=X_train.columns)
coeffs.sort_values().plot(kind='barh', figsize=(10, 8))
plt.title("Lasso Coefficients (Feature Importance Baseline)")
plt.show()

# ============================================================
# WP2: Advanced Modeling (Student 2)
# ============================================================

# 1. XGBoost
# Reference: "Non-Linear Models". Trees handle interactions naturally.
# We do NOT use the scaled data here, as trees are scale-invariant.
model = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, n_jobs=-1)
model.fit(X_train, y_train)

# Get probabilities for the validation set
val_probs = model.predict_proba(X_val)[:, 1]
print(f"XGBoost AUC: {roc_auc_score(y_val, val_probs):.3f}")

# ============================================================
# WP3: Explainability & Decision (Student 3)
# ============================================================

# 1. SHAP Analysis
# Reference: "Additive Feature Attribution"
# We explain the *test* set to understand generalization behavior.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

plt.figure()
shap.summary_plot(shap_values, X_val)

# 2. Optimization
# Reference: "Anomaly Detection" (Threshold Optimization)
# Define Costs (Hypothetical business case)
C_FALSE_ALARM = 1
C_MISSED_ACCIDENT = 50 

best_thr, min_cost = util.opt_threshold(y_val, val_probs, C_FALSE_ALARM, C_MISSED_ACCIDENT)

print(f"Optimal Threshold: {best_thr:.3f}")
print(f"Business Cost at Optimal Threshold: {min_cost}")

# Compare with default threshold (0.5)
default_cost = util.calculate_cost(y_val, val_probs, 0.5, C_FALSE_ALARM, C_MISSED_ACCIDENT)
print(f"Business Cost at Default Threshold (0.5): {default_cost}")
print(f"Value Generated by Optimization: {default_cost - min_cost} units")
```