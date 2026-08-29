"""
Model training module for credit card fraud detection.
Implements baseline and class-imbalance-aware models.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def train_baseline_logistic_regression(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> LogisticRegression:
    """
    Train a baseline logistic regression WITHOUT handling class imbalance.
    
    This serves as the pedagogical baseline to demonstrate:
    - High accuracy (~99%) ✓ (misleading!)
    - Very low recall on fraud class ✗ (catches almost no frauds)
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Fitted LogisticRegression model
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model


def train_balanced_logistic_regression(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> LogisticRegression:
    """
    Train logistic regression with class_weight='balanced'.
    
    This automatically adjusts class weights inversely proportional to class frequencies,
    making the model pay more attention to the minority class (fraud).
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Fitted LogisticRegression model
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model


def train_smote_logistic_regression(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> tuple:
    """
    Train logistic regression with SMOTE (Synthetic Minority Over-sampling Technique).
    
    IMPORTANT: SMOTE is applied ONLY to training data. Never apply SMOTE to test data
    to avoid data leakage and unrealistic performance estimates.
    
    SMOTE creates synthetic fraud samples by interpolating between existing fraud cases,
    balancing the dataset before training.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Tuple of (fitted_model, X_train_resampled, y_train_resampled)
    """
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train on resampled data
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        verbose=0
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    return model, X_train_resampled, y_train_resampled


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str = 'balanced',
    random_state: int = 42,
    n_estimators: int = 100
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with balanced class weights.
    
    Args:
        X_train: Training features
        y_train: Training target
        class_weight: How to weight classes ('balanced' or None)
        random_state: Random seed
        n_estimators: Number of trees
        
    Returns:
        Fitted RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        max_depth=15
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = None,
    random_state: int = 42
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        scale_pos_weight: Weight for positive class. If None, computes from class ratio
        random_state: Random seed
        
    Returns:
        Fitted XGBClassifier
    """
    # Auto-compute scale_pos_weight if not provided
    # scale_pos_weight = (number of negative cases) / (number of positive cases)
    if scale_pos_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
    
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        eval_metric='aucpr',  # Use PR-AUC for imbalanced data
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model
