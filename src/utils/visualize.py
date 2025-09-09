
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()

def plot_roc_curve(y_true, y_scores, title='ROC Curve', save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        print(f"Saved ROC curve to {save_path}")
    plt.show()

def plot_kernel_matrix(kernel_matrix, title='Kernel Matrix', save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(kernel_matrix, cmap='viridis')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved kernel matrix to {save_path}")
    plt.show()
