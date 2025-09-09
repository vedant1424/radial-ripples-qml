
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pennylane as qml
from src.models.quantum_pennylane_kernel import pennylane_kernel

def test_kernel_psd():
    """after computing K_train, assert eigvals.min() > -1e-6 (allow tiny numerical negative values). If failed, print a warning but don't raise an Exception."""
    # Generate some dummy data
    X = np.random.rand(10, 2)
    
    # Create a PennyLane feature map
    dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(dev)
    def feature_map(x1, x2):
        qml.AngleEmbedding(x1, wires=range(2))
        qml.adjoint(qml.AngleEmbedding(x2, wires=range(2)))
        return qml.expval(qml.Identity(0))
    
    # Compute the kernel matrix
    kernel_matrix = pennylane_kernel(X, X, feature_map)
    
    # Check for PSD
    eigenvals = np.linalg.eigvalsh(kernel_matrix)
    assert eigenvals.min() > -1e-6, f"Kernel matrix is not positive semi-definite (min eigenvalue: {eigenvals.min()})"

if __name__ == '__main__':
    test_kernel_psd()
    print("All model tests passed!")
