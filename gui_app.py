import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import threading
from hyperparameter_sweep import HyperparameterSweep

class HyperparameterSweepApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("QML Hyperparameter Sweep Dashboard")
        self.geometry("1000x700")

        # --- Style ---
        style = ttk.Style(self)
        style.theme_use('clam')

        # --- Main Paned Window ---
        main_paned_window = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Top Frame (Configuration and Controls) ---
        top_frame = ttk.Frame(main_paned_window, padding="10")
        main_paned_window.add(top_frame)

        # --- Hyperparameter Configuration Frame ---
        config_frame = ttk.LabelFrame(top_frame, text="Hyperparameter Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=5)

        # Dataset Size
        ttk.Label(config_frame, text="Dataset Size (comma-separated):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.dataset_size_entry = ttk.Entry(config_frame, width=40)
        self.dataset_size_entry.insert(0, "2000, 5000, 10000")
        self.dataset_size_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # VQC Layers
        ttk.Label(config_frame, text="VQC Layers (comma-separated):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.vqc_layers_entry = ttk.Entry(config_frame, width=40)
        self.vqc_layers_entry.insert(0, "2, 4, 6")
        self.vqc_layers_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # VQC Learning Rate
        ttk.Label(config_frame, text="VQC Learning Rate (comma-separated):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.vqc_lr_entry = ttk.Entry(config_frame, width=40)
        self.vqc_lr_entry.insert(0, "0.01, 0.005, 0.001")
        self.vqc_lr_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Device Selection
        ttk.Label(config_frame, text="Quantum Device:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.device_var = tk.StringVar(value="cpu")
        device_frame = ttk.Frame(config_frame)
        device_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(device_frame, text="CPU (Lightning)", variable=self.device_var, value="cpu").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(device_frame, text="GPU (Lightning)", variable=self.device_var, value="gpu").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(device_frame, text="Default", variable=self.device_var, value="default").pack(side=tk.LEFT, padx=5)
        
        # Z drive path
        ttk.Label(config_frame, text="Results Path (Z:\\...):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.z_drive_path_entry = ttk.Entry(config_frame, width=40)
        self.z_drive_path_entry.insert(0, "Z:\\qml_experiments")
        self.z_drive_path_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)

        # --- Experiment Controls Frame ---
        controls_frame = ttk.LabelFrame(top_frame, text="Experiment Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=5)

        self.start_sweep_button = ttk.Button(controls_frame, text="Start Sweep", command=self.start_sweep)
        self.start_sweep_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = ttk.Button(controls_frame, text="Pause", state=tk.DISABLED, command=self.pause_sweep)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.resume_button = ttk.Button(controls_frame, text="Resume", state=tk.DISABLED, command=self.resume_sweep)
        self.resume_button.pack(side=tk.LEFT, padx=5)

        self.run_single_button = ttk.Button(controls_frame, text="Run Single Experiment")
        self.run_single_button.pack(side=tk.LEFT, padx=5)

        # --- Bottom Frame (Tabs for Progress and Visualization) ---
        bottom_frame = ttk.Frame(main_paned_window, padding="10")
        main_paned_window.add(bottom_frame)

        # --- Notebook (Tabs) ---
        notebook = ttk.Notebook(bottom_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # --- Progress Tab ---
        progress_tab = ttk.Frame(notebook)
        notebook.add(progress_tab, text="Progress")

        progress_frame = ttk.LabelFrame(progress_tab, text="Live Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("run", "dataset_size", "vqc_layers", "lr", "status", "classical_acc", "qml_acc")
        self.progress_tree = ttk.Treeview(progress_frame, columns=columns, show="headings")
        
        for col in columns:
            self.progress_tree.heading(col, text=col.replace("_", " ").title())
            self.progress_tree.column(col, width=100)

        self.progress_tree.pack(fill=tk.BOTH, expand=True)

        # --- Visualization Tab ---
        viz_tab = ttk.Frame(notebook)
        notebook.add(viz_tab, text="Visualization")

        viz_frame = ttk.LabelFrame(viz_tab, text="Results Visualization", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        viz_controls_frame = ttk.Frame(viz_frame)
        viz_controls_frame.pack(fill=tk.X, pady=5)

        ttk.Label(viz_controls_frame, text="Plot Type:").pack(side=tk.LEFT, padx=5)
        self.plot_type_combo = ttk.Combobox(viz_controls_frame, values=["Accuracy vs. Dataset Size", "Accuracy vs. VQC Layers", "Accuracy vs. Learning Rate"])
        self.plot_type_combo.pack(side=tk.LEFT, padx=5)
        self.plot_type_combo.current(0)

        self.refresh_plot_button = ttk.Button(viz_controls_frame, text="Refresh Plot", command=self.refresh_plot)
        self.refresh_plot_button.pack(side=tk.LEFT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Queue and Events ---
        self.gui_queue = queue.Queue()
        self.pause_event = threading.Event()
        self.resume_event = threading.Event()
        
        # --- Data Storage for Plotting ---
        self.results_data = []

        self.after(100, self.process_queue)

    def start_sweep(self):
        self.start_sweep_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.run_single_button.config(state=tk.DISABLED)

        dataset_sizes = [int(x.strip()) for x in self.dataset_size_entry.get().split(',')]
        vqc_layers = [int(x.strip()) for x in self.vqc_layers_entry.get().split(',')]
        vqc_lrs = [float(x.strip()) for x in self.vqc_lr_entry.get().split(',')]
        z_drive_path = self.z_drive_path_entry.get()
        device_type = self.device_var.get()

        self.sweep_thread = HyperparameterSweep(self.gui_queue, self.pause_event, self.resume_event, dataset_sizes, vqc_layers, vqc_lrs, z_drive_path, device_type)
        self.sweep_thread.start()

    def pause_sweep(self):
        self.pause_event.set()
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.NORMAL)

    def resume_sweep(self):
        self.pause_event.clear()
        self.resume_event.set()
        self.pause_button.config(state=tk.NORMAL)
        self.resume_button.config(state=tk.DISABLED)

    def process_queue(self):
        try:
            message = self.gui_queue.get_nowait()
            if message["type"] == "new_run":
                self.progress_tree.insert("", "end", iid=message["run_id"], values=(message["run_id"], message["params"]["dataset_size"], message["params"]["vqc_layers"], message["params"]["learning_rate"], "Starting...", "", ""))
            elif message["type"] == "update":
                self.progress_tree.set(message["run_id"], "status", message["status"])
                if "classical_acc" in message:
                    self.progress_tree.set(message["run_id"], "classical_acc", f"{message['classical_acc']:.4f}")
                if "qml_acc" in message:
                    self.progress_tree.set(message["run_id"], "qml_acc", f"{message['qml_acc']:.4f}")
            elif message["type"] == "finished":
                messagebox.showinfo("Sweep Finished", "The hyperparameter sweep is complete.")
                self.start_sweep_button.config(state=tk.NORMAL)
                self.pause_button.config(state=tk.DISABLED)
                self.resume_button.config(state=tk.DISABLED)
                self.run_single_button.config(state=tk.NORMAL)
                # Auto-refresh plot when sweep finishes
                self.refresh_plot()

        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def collect_results_data(self):
        """Collect results from the progress tree for plotting."""
        self.results_data = []
        for item in self.progress_tree.get_children():
            values = self.progress_tree.item(item)['values']
            if len(values) >= 7 and values[4] == "Complete":  # Status is Complete
                try:
                    dataset_size = int(values[1])
                    vqc_layers = int(values[2])
                    learning_rate = float(values[3])
                    classical_acc = float(values[5]) if values[5] else 0.0
                    qml_acc = float(values[6]) if values[6] else 0.0
                    
                    self.results_data.append({
                        'dataset_size': dataset_size,
                        'vqc_layers': vqc_layers,
                        'learning_rate': learning_rate,
                        'classical_acc': classical_acc,
                        'qml_acc': qml_acc
                    })
                except (ValueError, IndexError):
                    continue

    def refresh_plot(self):
        """Refresh the plot based on selected plot type."""
        self.collect_results_data()
        
        if not self.results_data:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No completed results to plot', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_title('No Data Available')
            self.canvas.draw()
            return
        
        plot_type = self.plot_type_combo.get()
        self.ax.clear()
        
        if plot_type == "Accuracy vs. Dataset Size":
            self._plot_accuracy_vs_dataset_size()
        elif plot_type == "Accuracy vs. VQC Layers":
            self._plot_accuracy_vs_vqc_layers()
        elif plot_type == "Accuracy vs. Learning Rate":
            self._plot_accuracy_vs_learning_rate()
        
        self.canvas.draw()

    def _plot_accuracy_vs_dataset_size(self):
        """Plot accuracy vs dataset size."""
        dataset_sizes = sorted(list(set([d['dataset_size'] for d in self.results_data])))
        classical_accs = []
        qml_accs = []
        
        for size in dataset_sizes:
            size_data = [d for d in self.results_data if d['dataset_size'] == size]
            if size_data:
                classical_accs.append(sum(d['classical_acc'] for d in size_data) / len(size_data))
                qml_accs.append(sum(d['qml_acc'] for d in size_data) / len(size_data))
            else:
                classical_accs.append(0)
                qml_accs.append(0)
        
        self.ax.plot(dataset_sizes, classical_accs, 'o-', label='Classical', linewidth=2, markersize=6)
        self.ax.plot(dataset_sizes, qml_accs, 's-', label='Quantum', linewidth=2, markersize=6)
        self.ax.set_xlabel('Dataset Size')
        self.ax.set_ylabel('Accuracy')
        self.ax.set_title('Accuracy vs Dataset Size')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _plot_accuracy_vs_vqc_layers(self):
        """Plot accuracy vs VQC layers."""
        vqc_layers = sorted(list(set([d['vqc_layers'] for d in self.results_data])))
        classical_accs = []
        qml_accs = []
        
        for layers in vqc_layers:
            layer_data = [d for d in self.results_data if d['vqc_layers'] == layers]
            if layer_data:
                classical_accs.append(sum(d['classical_acc'] for d in layer_data) / len(layer_data))
                qml_accs.append(sum(d['qml_acc'] for d in layer_data) / len(layer_data))
            else:
                classical_accs.append(0)
                qml_accs.append(0)
        
        self.ax.plot(vqc_layers, classical_accs, 'o-', label='Classical', linewidth=2, markersize=6)
        self.ax.plot(vqc_layers, qml_accs, 's-', label='Quantum', linewidth=2, markersize=6)
        self.ax.set_xlabel('VQC Layers')
        self.ax.set_ylabel('Accuracy')
        self.ax.set_title('Accuracy vs VQC Layers')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _plot_accuracy_vs_learning_rate(self):
        """Plot accuracy vs learning rate."""
        learning_rates = sorted(list(set([d['learning_rate'] for d in self.results_data])))
        classical_accs = []
        qml_accs = []
        
        for lr in learning_rates:
            lr_data = [d for d in self.results_data if d['learning_rate'] == lr]
            if lr_data:
                classical_accs.append(sum(d['classical_acc'] for d in lr_data) / len(lr_data))
                qml_accs.append(sum(d['qml_acc'] for d in lr_data) / len(lr_data))
            else:
                classical_accs.append(0)
                qml_accs.append(0)
        
        self.ax.semilogx(learning_rates, classical_accs, 'o-', label='Classical', linewidth=2, markersize=6)
        self.ax.semilogx(learning_rates, qml_accs, 's-', label='Quantum', linewidth=2, markersize=6)
        self.ax.set_xlabel('Learning Rate')
        self.ax.set_ylabel('Accuracy')
        self.ax.set_title('Accuracy vs Learning Rate')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    app = HyperparameterSweepApp()
    app.mainloop()
