"""
Evaluation module for model performance assessment.
Focus on metrics appropriate for imbalanced classification.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, auc, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
    model_name: str = "Model"
) -> dict:
    """
    Evaluate model performance with metrics suitable for imbalanced data.
    
    Returns multiple metrics because accuracy alone is misleading:
    - Accuracy: percentage correct (high but misleading for imbalanced data)
    - Precision: of predicted frauds, how many are actually frauds?
    - Recall (Sensitivity): of actual frauds, how many do we catch?
    - F1: harmonic mean of precision and recall
    - ROC-AUC: only useful if you care equally about all thresholds
    - PR-AUC: more informative for imbalanced data (focuses on positive class)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (hard predictions)
        y_pred_proba: Predicted probabilities (for AUC scores)
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # AUC scores only if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = None
        
        # PR-AUC: more informative for imbalanced data
        # Why? ROC-AUC includes false positives in the denominator (which are huge here).
        # PR-AUC focuses on precision/recall trade-off for the positive class only.
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = auc(recall_vals, precision_vals)
    
    return metrics


def print_metrics_comparison(metrics_list: list[dict]):
    """
    Print a formatted comparison of multiple models' metrics.
    
    Args:
        metrics_list: List of metric dictionaries from evaluate_model
    """
    df = pd.DataFrame(metrics_list)
    print("\n" + "="*80)
    print("MODEL COMPARISON - DETAILED METRICS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save figure (if None, doesn't save)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Legitimate', 'Fraud'])
    plt.yticks([0.5, 1.5], ['Legitimate', 'Fraud'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
):
    """
    Plot precision-recall curve.
    
    This curve is more informative than ROC for imbalanced data.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Path to save figure (if None, doesn't save)
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, marker='.', label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    
    plt.show()


def print_classification_report(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Classification Report - {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))
    print(f"{'='*60}\n")
