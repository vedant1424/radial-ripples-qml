
import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def pennylane_kernel(X1, X2, feature_map):
    """Compute the kernel matrix between two datasets."""
    kernel = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            kernel[i, j] = feature_map(X1[i], X2[j])[0] # Corrected line
    return kernel

def train_svm_with_pennylane_kernel(X_train, y_train, X_test, y_test):
    """Train an SVM with a PennyLane kernel."""
    dev = qml.device("default.qubit", wires=X_train.shape[1])

    @qml.qnode(dev)
    def feature_map(x1, x2):
        qml.AngleEmbedding(x1, wires=range(X_train.shape[1]))
        qml.adjoint(qml.AngleEmbedding(x2, wires=range(X_train.shape[1])))
        return qml.probs(wires=range(X_train.shape[1]))

    print("Computing training kernel matrix...")
    kernel_train = pennylane_kernel(X_train, X_train, feature_map)

    print("Computing test kernel matrix...")
    kernel_test = pennylane_kernel(X_test, X_train, feature_map)

    model = SVC(kernel='precomputed')
    model.fit(kernel_train, y_train)

    y_pred_train = model.predict(kernel_train)
    y_pred_test = model.predict(kernel_test)

    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test)
    }

    kernel_matrices = {
        'train': kernel_train,
        'test': kernel_test
    }

    return model, metrics, kernel_matrices
