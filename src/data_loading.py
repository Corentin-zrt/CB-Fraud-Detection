"""
Data loading module for credit card fraud detection project.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card fraud detection dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the data
    """
    df = pd.read_csv(filepath)
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary stats
    """
    summary = {
        'total_transactions': len(df),
        'total_features': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'fraud_cases': (df['Class'] == 1).sum(),
        'legitimate_cases': (df['Class'] == 0).sum(),
        'fraud_ratio': (df['Class'] == 1).sum() / len(df),
    }
    return summary
