import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

def build_vqc(n_qubits: int = 6, n_layers: int = 2):
    """Return a PennyLane QNode and a wrapper training function that accepts X_train, y_train and returns weights + history."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    def variational_classifier(weights, bias, x):
        return circuit(weights, x) + bias

    return variational_classifier, circuit

def train_vqc(X_train, y_train, X_val, y_val, n_qubits=6, n_layers=2, epochs=100, batch_size=32, lr=0.01):
    """Train and return history (loss/accuracy per epoch) and the trained params."""
    var_classifier, circuit = build_vqc(n_qubits, n_layers)
    
    # Initialize weights and bias
    num_weights = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = np.random.uniform(0, 2 * np.pi, size=num_weights)
    bias = np.array(0.0, requires_grad=True)

    opt = AdamOptimizer(lr)

    def cost(weights, bias, x, y, v_classifier):
        predictions = v_classifier(weights, bias, x)
        return np.mean((predictions - y) ** 2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # Train in batches
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            (weights, bias), _ = opt.step_and_cost(lambda w, b: cost(w, b, X_batch, y_batch, var_classifier), weights, bias)

        # Compute loss and accuracy for training and validation sets
        train_preds = np.sign(var_classifier(weights, bias, X_train))
        train_acc = np.mean(train_preds == y_train)
        train_loss = cost(weights, bias, X_train, y_train, var_classifier)

        val_preds = np.sign(var_classifier(weights, bias, X_val))
        val_acc = np.mean(val_preds == y_val)
        val_loss = cost(weights, bias, X_val, y_val, var_classifier)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    return {"weights": weights, "bias": bias}, history