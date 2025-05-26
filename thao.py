import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)

def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Comprehensive model evaluation function that calculates and prints various metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    """
    # Basic metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    
    # Print basic metrics
    print("\n=== Basic Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # If probabilities are provided, calculate additional metrics
    if y_proba is not None:
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()

def find_optimal_threshold(y_true, y_proba, min_recall=0.6):
    """
    Find the optimal threshold that maximizes precision while maintaining minimum recall.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        min_recall: Minimum recall threshold (default: 0.6)
    
    Returns:
        optimal_threshold: The threshold that maximizes precision while maintaining min_recall
        optimal_precision: The precision at the optimal threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold with highest precision where recall ≥ min_recall
    optimal_threshold = 0.5  # fallback default
    optimal_precision = 0
    
    for p, r, t in zip(precisions, recalls, thresholds):
        if r >= min_recall and p > optimal_precision:
            optimal_precision = p
            optimal_threshold = t
    
    return optimal_threshold, optimal_precision

def plot_threshold_analysis(y_true, y_proba, min_recall=0.6):
    """
    Plot precision and recall curves with the optimal threshold marked.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        min_recall: Minimum recall threshold (default: 0.6)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    optimal_threshold, optimal_precision = find_optimal_threshold(y_true, y_proba, min_recall)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label="Precision", color='blue')
    plt.plot(thresholds, recalls[:-1], label="Recall", color='green')
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
    print(f"Precision at optimal threshold: {optimal_precision:.3f}")
    print(f"Recall at optimal threshold: {recalls[np.argmin(np.abs(thresholds - optimal_threshold))]:.3f}") 