"""
Preprocessing module for credit card fraud detection project.
Handles train/test split with stratification and feature scaling.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def prepare_train_test_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.3, 
    random_state: int = 42
) -> tuple:
    """
    Split data into train and test sets with stratification.
    
    IMPORTANT: Stratification preserves the fraud/legitimate ratio in both sets.
    This is critical for imbalanced datasets.
    
    Args:
        X: Feature matrix
        y: Target variable (Class column)
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,  # Critical: maintains fraud ratio in both sets
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Scale 'Amount' and 'Time' features using StandardScaler.
    
    NOTE: V1-V28 are already normalized PCA components, so only Amount and Time
    need standardization.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Fit only on training data to avoid data leakage
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale Amount and Time
    X_train_scaled[['Amount', 'Time']] = scaler.fit_transform(
        X_train[['Amount', 'Time']]
    )
    X_test_scaled[['Amount', 'Time']] = scaler.transform(
        X_test[['Amount', 'Time']]
    )
    
    return X_train_scaled, X_test_scaled, scaler


def get_stratified_kfold(n_splits: int = 5, random_state: int = 42):
    """
    Get a StratifiedKFold splitter for cross-validation.
    
    Ensures each fold maintains the fraud/legitimate ratio.
    
    Args:
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        StratifiedKFold object
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
