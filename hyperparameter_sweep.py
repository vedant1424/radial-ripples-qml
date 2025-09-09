import time
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple
from threading import Thread, Event

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

# --- Data Generation ---
def sample_parameters(n: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    A = np.random.uniform(0.6, 1.0, n)
    f = np.random.uniform(1.0, 8.0, n)
    k = np.random.uniform(-3, 3, n)
    phi = np.random.uniform(0, 2 * np.pi, n)
    cx = np.random.uniform(0.35, 0.65, n)
    cy = np.random.uniform(0.35, 0.65, n)
    return np.stack([A, f, k, phi, cx, cy], axis=1)

def render_ripple(params: np.ndarray, size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    A, f, k, phi, cx_rel, cy_rel = params
    H, W = size
    cx, cy = W * cx_rel, H * cy_rel
    R = min(H, W) / 2
    y, x = np.mgrid[0:H, 0:W]
    x_norm = (x - cx) / R
    y_norm = (y - cy) / R
    r = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)
    image = 0.5 * (1 + A * np.sin(2 * np.pi * f * r + k * theta + phi))
    return image.astype(np.float32)

def build_dataset(n: int, out_dir: str, size: Tuple[int, int] = (32, 32), seed: int = 42):
    params = sample_parameters(n, seed)
    s = np.sin(3 * params[:, 1]) + 0.7 * np.sin(2 * params[:, 3]) + 0.5 * (params[:, 0] - 0.8) + 0.4 * np.sign(params[:, 2])
    labels = (s > 0).astype(int)
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    records = []
    for i in range(len(params)):
        img_array = render_ripple(params[i], size)
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        filename = f"ripple_{i:04d}.png"
        img.save(os.path.join(out_dir, 'images', filename))
        record = {
            'filename': filename,
            'A': params[i, 0],
            'f': params[i, 1],
            'k': params[i, 2],
            'phi': params[i, 3],
            'cx': params[i, 4],
            'cy': params[i, 5],
            'label': labels[i]
        }
        records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'params_labels.csv'), index=False)

# --- Classical Model Training ---
def train_classical_models(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_accuracy = lr_model.score(X_test_scaled, y_test)
    
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_accuracy = svm_model.score(X_test_scaled, y_test)
    
    return {"Logistic Regression": lr_accuracy, "SVM": svm_accuracy}

# --- VQC Training ---
def build_vqc(n_qubits: int = 6, n_layers: int = 2):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))
    def variational_classifier(weights, bias, x):
        return circuit(weights, x) + bias
    return variational_classifier, circuit

def train_vqc(X_train, y_train, X_val, y_val, n_layers, lr, gui_queue, run_id, epochs=50, batch_size=32):
    var_classifier, circuit = build_vqc(n_layers=n_layers)
    num_weights = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=6)
    weights = pnp.random.uniform(0, 2 * np.pi, size=num_weights)
    bias = pnp.array(0.0, requires_grad=True)
    opt = AdamOptimizer(lr)
    def cost(weights, bias, x, y, v_classifier):
        predictions = v_classifier(weights, bias, x)
        return np.mean((predictions - y) ** 2)
    
    num_batches = len(X_train) // batch_size
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            (weights, bias), _ = opt.step_and_cost(lambda w, b: cost(w, b, X_batch, y_batch, var_classifier), weights, bias)
            
            # Send progress update to GUI
            batch_num = (i // batch_size) + 1
            gui_queue.put({"type": "update", "run_id": run_id, "status": f"Training VQC (Epoch {epoch+1}/{epochs}, Batch {batch_num}/{num_batches})"})

        val_preds = np.sign(var_classifier(weights, bias, X_val))
        val_acc = np.mean(val_preds == y_val)
        gui_queue.put({"type": "update", "run_id": run_id, "status": f"Training VQC (Epoch {epoch+1}/{epochs}) - Val Acc: {val_acc:.4f}"})


    val_preds = np.sign(var_classifier(weights, bias, X_val))
    val_acc = np.mean(val_preds == y_val)
    return val_acc

class HyperparameterSweep(Thread):
    def __init__(self, gui_queue, pause_event, resume_event, dataset_sizes, vqc_layers, vqc_lrs, z_drive_path):
        super().__init__()
        self.gui_queue = gui_queue
        self.pause_event = pause_event
        self.resume_event = resume_event
        self.dataset_sizes = dataset_sizes
        self.vqc_layers = vqc_layers
        self.vqc_lrs = vqc_lrs
        self.z_drive_path = z_drive_path
        self.run_counter = 0

    def run(self):
        for ds_size in self.dataset_sizes:
            for layers in self.vqc_layers:
                for lr in self.vqc_lrs:
                    self.run_counter += 1
                    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
                    run_path = os.path.join(self.z_drive_path, run_name)
                    os.makedirs(run_path, exist_ok=True)

                    params = {
                        "dataset_size": ds_size,
                        "vqc_layers": layers,
                        "learning_rate": lr
                    }
                    with open(os.path.join(run_path, "parameters.json"), 'w') as f:
                        json.dump(params, f, indent=4)

                    self.gui_queue.put({"type": "new_run", "run_id": self.run_counter, "params": params})

                    # --- Data Generation ---
                    self.gui_queue.put({"type": "update", "run_id": self.run_counter, "status": "Generating data..."})
                    build_dataset(n=ds_size, out_dir=run_path)
                    dataset_df = pd.read_csv(os.path.join(run_path, 'params_labels.csv'))
                    X = dataset_df[['A', 'f', 'k', 'phi', 'cx', 'cy']].values
                    y = dataset_df['label'].values
                    y = np.where(y == 0, -1, 1)
                    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
                    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

                    # --- Classical Models ---
                    self.gui_queue.put({"type": "update", "run_id": self.run_counter, "status": "Training classical models..."})
                    classical_results = train_classical_models(X_train, y_train, X_test, y_test)

                    # --- VQC Model ---
                    self.gui_queue.put({"type": "update", "run_id": self.run_counter, "status": "Training VQC..."})
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    vqc_accuracy = train_vqc(X_train_scaled, y_train, X_val_scaled, y_val, n_layers=layers, lr=lr, gui_queue=self.gui_queue, run_id=self.run_counter)

                    # --- Save Results ---
                    self.gui_queue.put({"type": "update", "run_id": self.run_counter, "status": "Saving results..."})
                    results_df = pd.DataFrame({
                        "Model": ["Logistic Regression", "SVM", "VQC"],
                        "Accuracy": [classical_results["Logistic Regression"], classical_results["SVM"], vqc_accuracy]
                    })
                    results_df.to_csv(os.path.join(run_path, "results.csv"), index=False)

                    self.gui_queue.put({"type": "update", "run_id": self.run_counter, "status": "Complete", "classical_acc": classical_results["Logistic Regression"], "qml_acc": vqc_accuracy})

                    # --- Pause Logic ---
                    if self.pause_event.is_set():
                        self.gui_queue.put({"type": "update", "run_id": self.run_counter, "status": "Paused"})
                        self.resume_event.wait()
                        self.resume_event.clear()

        self.gui_queue.put({"type": "finished"})