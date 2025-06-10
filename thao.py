import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)

def evaluate_model(y_true, y_pred, y_proba=None, class_names=None):
    """
    Comprehensive model evaluation function that calculates and prints various metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        class_names: Names of classes for better visualization (optional)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Input validation
    if not isinstance(y_true, (np.ndarray, list)):
        raise TypeError("y_true must be a numpy array or list")
    if not isinstance(y_pred, (np.ndarray, list)):
        raise TypeError("y_pred must be a numpy array or list")
    
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Detect problem type
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    is_multiclass = n_classes > 2
    
    print(f"\n=== Problem Type: {'Multi-class' if is_multiclass else 'Binary'} Classification ===")
    print(f"Number of classes: {n_classes}")
    
    # Calculate basic metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted' if is_multiclass else 'binary'),
        'Recall': recall_score(y_true, y_pred, average='weighted' if is_multiclass else 'binary'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted' if is_multiclass else 'binary')
    }
    
    # Print basic metrics
    print("\n=== Basic Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print classification report
    print("\n=== Classification Report ===")
    if class_names is not None:
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    if class_names is not None and len(class_names) == n_classes:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Plot ROC and PR curves if probabilities are provided
    if y_proba is not None:
        if is_multiclass:
            _plot_multiclass_curves(y_true, y_proba, unique_classes, class_names)
        else:
            _plot_binary_curves(y_true, y_proba)
    
    return metrics

def _plot_binary_curves(y_true, y_proba):
    """Plot ROC and PR curves for binary classification."""
    # Ensure y_proba is 1D for binary case
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    ax2.plot(recall, precision, color='blue', lw=2, 
            label=f'PR curve (AP = {avg_precision:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def _plot_multiclass_curves(y_true, y_proba, classes, class_names=None):
    """Plot ROC and PR curves for multi-class classification."""
    n_classes = len(classes)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different classes
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    # ROC Curves (One-vs-Rest)
    for i, (class_label, color) in enumerate(zip(classes, colors, strict=True)):
        y_true_binary = (y_true == class_label).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
        roc_auc = roc_auc_score(y_true_binary, y_proba[:, i])
        
        class_name = class_names[i] if class_names else f'Class {class_label}'
        ax1.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Multi-class ROC Curves (One-vs-Rest)')
    ax1.legend(loc="lower right", bbox_to_anchor=(1.15, 0))
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    for i, (class_label, color) in enumerate(zip(classes, colors, strict=True)):
        y_true_binary = (y_true == class_label).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
        avg_precision = average_precision_score(y_true_binary, y_proba[:, i])
        
        class_name = class_names[i] if class_names else f'Class {class_label}'
        ax2.plot(recall, precision, color=color, lw=2,
                label=f'{class_name} (AP = {avg_precision:.3f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Multi-class PR Curves (One-vs-Rest)')
    ax2.legend(loc="lower left", bbox_to_anchor=(1.15, 0))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()