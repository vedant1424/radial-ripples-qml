Group Number-51
Under Guidance of
@DR.Sandeep Kumar Singh
@vedant thakur-20224075
@anurag Gupta-20204030
@Dhruv Kumar Tiwari - 20224057

Radial Ripples QML

A comprehensive Quantum Machine Learning (QML) project that demonstrates quantum classification of synthetic radial ripple images using PennyLane. This project compares classical machine learning baselines with quantum approaches including Variational Quantum Classifiers (VQC) and quantum kernel methods.

Project Overview

This project generates synthetic radial ripple images with varying parameters (amplitude, frequency, wave number, phase, center coordinates) and uses them to train and compare different machine learning models:

Classical Models:
- Logistic Regression
- Support Vector Machine (SVM) with RBF kernel

Quantum Models:
- Variational Quantum Classifier (VQC) using PennyLane with StronglyEntanglingLayers
- Quantum Kernel SVM using PennyLane feature maps with classical SVM

Key Features:
- Synthetic Data Generation: Creates radial ripple images with mathematical parameters
- Multi-Model Comparison: Classical vs Quantum approaches
- Interactive GUI: Hyperparameter sweep dashboard with real-time monitoring
- Visualization: Confusion matrices, kernel heatmaps, training histories
- Reproducible Experiments: Seeded random generation and standardized evaluation



Installation

Prerequisites
- Python 3.8 or higher
- pip package manager

Step 1: Clone or Download
If using git:
git clone <repository-url>
cd radial_ripples_qml

Or download and extract the project folder

Step 2: Create Virtual Environment
Create virtual environment:
python -m venv .venv

Activate virtual environment:
On Windows:
.venv\Scripts\activate
On macOS/Linux:
source .venv/bin/activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Verify Installation
python -c "import pennylane as qml; print('PennyLane version:', qml.__version__)"

Usage

Method 1: Interactive GUI (Recommended)

Launch the hyperparameter sweep dashboard:
python gui_app.py

GUI Features:
- Configure dataset sizes, VQC layers, and learning rates
- Run hyperparameter sweeps with real-time progress monitoring
- Visualize results with interactive plots
- Pause/resume long-running experiments

Method 2: Command Line Experiment

Run the complete experiment pipeline:
python src/run_experiment.py

This will:
1. Generate 1000 synthetic ripple images
2. Train classical baselines (Logistic Regression, RBF SVM)
3. Train quantum models (VQC, Quantum Kernel SVM)
4. Generate confusion matrices and performance comparisons
5. Save results to outputs/ directory

Method 3: Jupyter Notebook

Launch Jupyter and explore the demo:
jupyter lab notebooks/demo_notebook.ipynb

Method 4: Individual Components

Generate Dataset:
python -c "
import sys; sys.path.append('src')
from data.generate_ripples import build_dataset
build_dataset(n=1000, out_dir='outputs', size=(32,32), seed=42)
"

Train Specific Models:
python -c "
import sys; sys.path.append('src')
from models.quantum_pennylane import train_vqc, build_vqc
# Your training code here
"

Expected Results

The project typically achieves:
- Classical Models: 70-85% accuracy
- Quantum VQC: 60-80% accuracy (varies with hyperparameters)
- Quantum Kernel SVM: 65-80% accuracy

Results are saved in:
- outputs/figures/: Confusion matrices, kernel heatmaps, training plots
- outputs/results/: CSV files with accuracy metrics
- outputs/images/: Generated ripple dataset

Configuration

Key Parameters:
- Dataset Size: Number of synthetic images (default: 1000)
- Image Size: Resolution of generated images (default: 32x32)
- VQC Layers: Depth of quantum circuit (default: 2)
- Learning Rate: Optimization step size (default: 0.01)
- Epochs: Training iterations (default: 50)

Customization:
Edit parameters in:
- src/run_experiment.py for command-line experiments
- GUI interface for interactive sweeps
- src/data/generate_ripples.py for data generation parameters

Testing

Run the test suite:
python -m pytest src/tests/

Or run individual tests:
python src/tests/test_models.py
python src/tests/test_dataset.py

Performance Tips

1. Use GPU: Install pennylane-lightning-gpu for faster quantum simulations
2. Batch Size: Adjust VQC batch size based on available memory
3. Device Selection: Use lightning.qubit device for better performance
4. Parallel Sweeps: GUI supports multiple concurrent experiments

Troubleshooting

Common Issues:

1. Import Errors: Ensure virtual environment is activated
2. Memory Issues: Reduce dataset size or batch size
3. GUI Not Starting: Check tkinter installation: python -c "import tkinter"
4. Slow Performance: Use pennylane-lightning device instead of default.qubit

Dependencies Issues:
Reinstall PennyLane:
pip uninstall pennylane
pip install pennylane pennylane-lightning

Check installation:
python -c "import pennylane as qml; print(qml.devices())"

Dependencies

- Core: numpy, scipy, matplotlib, seaborn, pandas
- ML: scikit-learn, torch, torchvision
- Quantum: pennylane, pennylane-lightning
- GUI: tkinter (built-in), matplotlib
- Notebooks: jupyterlab

Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

License

This project is open source.
Acknowledgments

- PennyLane team for the quantum machine learning framework
- Qiskit community for initial inspiration

