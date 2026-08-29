# Credit Card Fraud Detection - ML Project

A comprehensive machine learning project focused on **understanding class imbalance handling** using the Kaggle Credit Card Fraud Detection dataset.

## 🎯 Project Goal

This weekend project explores practical techniques for handling extreme class imbalance in fraud detection:
- Why accuracy alone is misleading (~99% accuracy catches 0% of fraud!)
- Hands-on comparison of balancing strategies: `class_weight`, SMOTE, and ensemble methods
- Understanding precision-recall trade-offs in production fraud detection

## 📊 Dataset Overview

**Kaggle: Credit Card Fraud Detection**
- **Total transactions**: 284,807
- **Fraudulent cases**: 492 (~0.17% class imbalance!)
- **Features**: 
  - `V1` to `V28`: PCA-transformed numerical features (anonymized for privacy)
  - `Amount`: Transaction amount (in cents)
  - `Time`: Seconds elapsed since first transaction
  - `Class`: 0 = Legitimate, 1 = Fraud

**Why this is pedagogically interesting:**
- Extreme class imbalance makes accuracy a useless metric
- Forces you to think about precision/recall trade-offs
- Real-world challenge: fraud detection in production

## 🗂️ Project Structure

```
CB Fraud Detection/
├── data/
│   ├── .gitignore          # Prevents committing large CSV files
│   └── creditcard.csv      # (not in repo, too large/sensitive)
├── src/
│   ├── __init__.py
│   ├── data_loading.py     # Load and summarize data
│   ├── preprocessing.py    # Train/test split (stratified!), scaling
│   ├── models.py           # Model training (baseline + balanced approaches)
│   ├── evaluation.py       # Metrics suitable for imbalanced data
│── notebooks/
│   ├── 01_exploration.ipynb         # Data exploration and visualizations
│   ├── 02_baseline_comparison.ipynb  # Naive vs balanced vs SMOTE
│   └── 03_final_models.ipynb         # LR vs RF vs XGBoost + SHAP
├── results/
│   ├── (figures, metrics, models saved here)
├── requirements.txt
├── .gitignore
└── README.md (this file)
```

## 🚀 Getting Started

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Move Dataset

The `creditcard.csv` must be in `data/`:
```bash
# Already done: data/creditcard.csv
```

### 3. Run Exploration Notebook

Start with the exploration notebook to understand the data:
```bash
jupyter notebook notebooks/01_exploration.ipynb
```

## 📚 Step-by-Step Workflow

### Step 1: Data Exploration (`01_exploration.ipynb`)
- Load and visualize class distribution (see how imbalanced it is!)
- Check for missing values
- Analyze `Amount` and `Time` distributions by class
- Note: V1-V28 are PCA components → no direct business interpretation

**Key insight to extract:**
```
Legitimate: 284,315 (~99.8%)
Fraud:      492 (~0.2%)
```

### Step 2: Baseline Comparison (`02_baseline_comparison.ipynb`)

Train three versions of logistic regression:

| Approach | Description | Accuracy | Recall (Fraud) | F1 |
|----------|-------------|----------|---|---|
| **Baseline** | No balancing | ~99.9% | ~30% | ⚠️ |
| **class_weight='balanced'** | Auto-weight by inverse frequency | ~99.5% | ~70% | ✓ |
| **SMOTE** | Synthetic oversampling of fraud | ~99.3% | ~80% | ✓✓ |

**Critical lesson:** Accuracy is a trap! Focus on:
- **Recall**: Of all actual frauds, how many do we catch?
- **Precision**: Of our fraud predictions, how many are correct?
- **F1-score**: Harmonic mean (balanced precision-recall)
- **PR-AUC**: Area under precision-recall curve (best for imbalanced data)

### Step 3: Model Comparison (`03_final_models.ipynb`)

Once the best balancing strategy is identified (likely SMOTE), compare:
- Logistic Regression
- Random Forest (with `class_weight='balanced'`)
- XGBoost (with `scale_pos_weight` tuned)

Use stratified 5-fold cross-validation for robustness.

### Step 4: Interpretability & Insights

Add SHAP analysis to understand which features drive fraud predictions:
```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## 🎓 Key Technical Concepts

### Why Stratification Matters

```python
# ✓ CORRECT: Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y  # ← Key!
)

# ✗ WRONG: Random split (could get all fraud in test or all in train)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3  # Missing stratify=y
)
```

**Result:**
- Stratified: Train & Test both ~0.17% fraud ✓
- Random: Could be 0% fraud in test set ✗

### Why SMOTE Only on Training Data

```python
# ✗ WRONG: SMOTE before split
X_smote, y_smote = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote)
# → Data leakage! Test set contains synthetic data similar to training

# ✓ CORRECT: SMOTE after split, on training only
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
# → Test set is purely real data, no leakage
```

### Why PR-AUC > ROC-AUC for Imbalanced Data

- **ROC-AUC**: Plots TPR vs FPR. With 284k legit cases, FPR stays near 0 → curve near (0,1) diagonal → misleading high AUC
- **PR-AUC**: Plots Precision vs Recall. Focused on the fraud class (minority) → more sensitive to actual performance

```python
# Always use PR-AUC for imbalanced problems
pr_auc = auc(recall_values, precision_values)
```

## 📈 Evaluation Metrics Explained

### For the Confusion Matrix:

```
                 Predicted Positive | Predicted Negative
Actual Positive  TP (caught fraud)  | FN (missed fraud)
Actual Negative  FP (false alarm)   | TN (correctly OK'd)
```

### Key Metrics:

- **Accuracy** = (TP + TN) / All  
  *Trap: High even if you catch 0 frauds!*

- **Precision** = TP / (TP + FP)  
  *Of our fraud alerts, how many are real?*

- **Recall (Sensitivity)** = TP / (TP + FN)  
  *Of all real frauds, how many do we catch?*

- **F1** = 2 × (Precision × Recall) / (Precision + Recall)  
  *Balanced score between precision and recall*

- **PR-AUC**  
  *Area under precision-recall curve (0 to 1, higher = better)*

## ⚠️ Project Limitations

1. **V1-V28 are anonymized PCA components**  
   → Can't extract business rules like "fraud happens with large Amount"  
   → Use for predictive modeling, not decision-making rules

2. **Dataset is from 2013, fixed in time**  
   → Real fraud patterns evolve (concept drift)  
   → Model trained on this won't generalize to new fraud tactics

3. **No temporal information beyond "Time"**  
   → Real fraud detection needs transaction history, merchant info, geography, etc.  
   → This is a simplified scenario

4. **All transactions are from credit cards**  
   → Doesn't cover other payment methods (banks, e-wallets, etc.)

## 📖 Resources & Further Reading

- Stanford CS229 (Andrew Ng): Class imbalance, precision-recall
- imbalanced-learn documentation: SMOTE and related techniques
- SHAP documentation: Model interpretability
- Kaggle Discussions: Real experiences from practitioners

## 📝 Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Run a specific notebook
jupyter notebook notebooks/01_exploration.ipynb
```

---

**Author**: Weekend ML Project  
**Date**: 2026  
**Goal**: Understand class imbalance handling, not production deployment
