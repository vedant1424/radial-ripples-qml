
import os
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
import sys
sys.path.append(os.path.abspath('src'))

# Configuration
SEED = 42
N_SAMPLES = 1000
IMAGE_SIZE = (32, 32)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Pennylane seed
import pennylane as qml
np.random.seed(SEED)

# Create output directories
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'results'), exist_ok=True)

# Save seed
with open(os.path.join(OUTPUT_DIR, 'results', 'seed.txt'), 'w') as f:
    f.write(str(SEED))

from data.generate_ripples import build_dataset

dataset_df = build_dataset(n=N_SAMPLES, out_dir=OUTPUT_DIR, size=IMAGE_SIZE, seed=SEED)
print(f'Generated {len(dataset_df)} images.')
print('Class distribution:')
print(dataset_df['label'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Parameter-features
features = ['A', 'f', 'k', 'phi', 'cx', 'cy']
X = dataset_df[features].values
y = dataset_df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from models.classical import train_logistic, train_svm_rbf
from utils.visualize import plot_confusion_matrix

# Logistic Regression
lr_model, lr_metrics = train_logistic(X_train_scaled, y_train)
print(f'Logistic Regression Accuracy: {lr_metrics["test_accuracy"]:.4f}')

# SVM with RBF Kernel
svm_model, svm_metrics = train_svm_rbf(X_train_scaled, y_train)
print(f'SVM RBF Accuracy: {svm_metrics["test_accuracy"]:.4f}')

# Confusion Matrices
lr_preds = lr_model.predict(X_test_scaled)
svm_preds = svm_model.predict(X_test_scaled)

plot_confusion_matrix(y_test, lr_preds, class_names=['0', '1'], title='Logistic Regression CM', save_path=os.path.join(OUTPUT_DIR, 'figures', 'logistic_confusion.png'))
plot_confusion_matrix(y_test, svm_preds, class_names=['0', '1'], title='SVM RBF CM', save_path=os.path.join(OUTPUT_DIR, 'figures', 'svm_confusion.png'))

from models.quantum_pennylane_kernel import train_svm_with_pennylane_kernel
from models.quantum_pennylane import train_vqc, build_vqc
from utils.visualize import plot_confusion_matrix, plot_kernel_matrix

# Train SVM with PennyLane Kernel
svm_pl_model, svm_pl_metrics, svm_pl_kernel_matrices = train_svm_with_pennylane_kernel(X_train_scaled, y_train, X_test_scaled, y_test)
print(f'SVM with PennyLane Kernel Accuracy: {svm_pl_metrics["test_accuracy"]:.4f}')

# Plot kernel matrix
plot_kernel_matrix(svm_pl_kernel_matrices['train'], title='SVM PennyLane Kernel Matrix', save_path=os.path.join(OUTPUT_DIR, 'figures', 'svm_pl_kernel_heatmap.png'))

# Confusion Matrix
svm_pl_preds = svm_pl_model.predict(svm_pl_kernel_matrices['test'])
plot_confusion_matrix(y_test, svm_pl_preds, class_names=['0', '1'], title='SVM PennyLane Kernel CM', save_path=os.path.join(OUTPUT_DIR, 'figures', 'svm_pl_confusion.png'))


 # Train SVM with PennyLane Kernel
 svm_pl_model, svm_pl_metrics, svm_pl_kernel_matrices = train_svm_with_pennylane_kernel(X_train_scaled, y_train, X_test_scaled, y_test)
 print(f'SVM with PennyLane Kernel Accuracy: {svm_pl_metrics["test_accuracy"]:.4f}')

 # Plot kernel matrix
 plot_kernel_matrix(svm_pl_kernel_matrices['train'], title='SVM PennyLane Kernel Matrix', save_path=os.path.join(OUTPUT_DIR, 'figures', 'svm_pl_kernel_heatmap.png'))

 # Confusion Matrix
 svm_pl_preds = svm_pl_model.predict(svm_pl_kernel_matrices['test'])
 plot_confusion_matrix(y_test, svm_pl_preds, class_names=['0', '1'], title='SVM PennyLane Kernel CM', save_path=os.path.join(OUTPUT_DIR, 'figures', 'svm_pl_confusion.png'))


 # Need to create a validation set for VQC training
 X_train_vqc, X_val_vqc, y_train_vqc, y_val_vqc = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=SEED, stratify=y_train)

 # PennyLane expects labels as -1 and 1
 y_train_vqc_pl = np.where(y_train_vqc == 0, -1, 1)
 y_val_vqc_pl = np.where(y_val_vqc == 0, -1, 1)

 vqc_params, vqc_history = train_vqc(X_train_vqc, y_train_vqc_pl, X_val_vqc, y_val_vqc_pl, n_qubits=6, n_layers=2, epochs=50)

 # Evaluate on the test set
 var_classifier, _ = build_vqc(n_qubits=6, n_layers=2)
 test_preds_vqc = np.sign(var_classifier(vqc_params['weights'], vqc_params['bias'], X_test_scaled))
 test_preds_vqc_binary = np.where(test_preds_vqc == -1, 0, 1)
 vqc_accuracy = np.mean(test_preds_vqc_binary == y_test)
 print(f'VQC Test Accuracy: {vqc_accuracy:.4f}')

 plot_confusion_matrix(y_test, test_preds_vqc_binary, class_names=['0', '1'], title='VQC Confusion Matrix', save_path=os.path.join(OUTPUT_DIR, 'figures', 'vqc_confusion.png'))

 results = {
     'Logistic Regression': lr_metrics['test_accuracy'],
     'SVM RBF': svm_metrics['test_accuracy'],
     'SVM PennyLane Kernel': svm_pl_metrics['test_accuracy'],
     'VQC': vqc_accuracy
 }

 results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Test Accuracy'])
 print(results_df)
 results_df.to_csv(os.path.join(OUTPUT_DIR, 'results', 'model_accuracies.csv'))



