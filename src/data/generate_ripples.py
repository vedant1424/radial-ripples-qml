
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple
import os

def sample_parameters(n: int, seed: int = 42) -> np.ndarray:
    """Return an (n, P) array of sampled parameters. P = 6: [A, f, k, phi, cx, cy].
    Use the ranges described above. Set RNG seed for reproducibility."""
    np.random.seed(seed)
    A = np.random.uniform(0.6, 1.0, n)
    f = np.random.uniform(1.0, 8.0, n)
    k = np.random.uniform(-3, 3, n)
    phi = np.random.uniform(0, 2 * np.pi, n)
    cx = np.random.uniform(0.35, 0.65, n)
    cy = np.random.uniform(0.35, 0.65, n)
    return np.stack([A, f, k, phi, cx, cy], axis=1)

def render_ripple(params: np.ndarray, size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Given a single parameter vector (A, f, k, phi, cx, cy) return a 2D float32 image in range [0,1]."""
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

def build_dataset(n: int, out_dir: str, size: Tuple[int, int] = (32, 32), seed: int = 42) -> pd.DataFrame:
    """Sample n parameter vectors, render images, write images as PNG into out_dir/images/, and return a DataFrame with columns:
    [filename, A, f, k, phi, cx, cy, label]
    The labeling rule is specified next."""
    params = sample_parameters(n, seed)
    
    # Labeling rule
    s = np.sin(3 * params[:, 1]) + 0.7 * np.sin(2 * params[:, 3]) + 0.5 * (params[:, 0] - 0.8) + 0.4 * np.sign(params[:, 2])
    labels = (s > 0).astype(int)

    # Balance check
    class_counts = pd.Series(labels).value_counts(normalize=True)
    if class_counts.min() < 0.3:
        print(f"Warning: Class imbalance detected: {class_counts.to_dict()}. Resampling...")
        # Simple resampling strategy: generate more data and downsample the majority class
        while class_counts.min() < 0.45 or class_counts.max() > 0.55:
            params = sample_parameters(n * 2, seed=np.random.randint(0, 10000))
            s = np.sin(3 * params[:, 1]) + 0.7 * np.sin(2 * params[:, 3]) + 0.5 * (params[:, 0] - 0.8) + 0.4 * np.sign(params[:, 2])
            labels = (s > 0).astype(int)
            
            df_temp = pd.DataFrame(params, columns=['A', 'f', 'k', 'phi', 'cx', 'cy'])
            df_temp['label'] = labels
            
            n_minority = df_temp['label'].value_counts().min()
            df_balanced = pd.concat([
                df_temp[df_temp['label'] == 0].head(n_minority),
                df_temp[df_temp['label'] == 1].head(n_minority)
            ])
            
            params = df_balanced[['A', 'f', 'k', 'phi', 'cx', 'cy']].values
            labels = df_balanced['label'].values
            class_counts = pd.Series(labels).value_counts(normalize=True)
            if len(params) >= n:
                params = params[:n]
                labels = labels[:n]
                break


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
    
    return df

if __name__ == '__main__':
    # Demonstrate the functions
    N_SAMPLES = 2000
    OUTPUT_DIR = "C:\\Users\\vedan\\Desktop\\qml\\radial_ripples_qml\\outputs"
    IMAGE_SIZE = (32, 32)
    
    print("Generating dataset...")
    dataset_df = build_dataset(n=N_SAMPLES, out_dir=OUTPUT_DIR, size=IMAGE_SIZE)
    print("Dataset generation complete.")
    print(f"Saved {len(dataset_df)} images and parameter CSV to {OUTPUT_DIR}")
    print("Class distribution:")
    print(dataset_df['label'].value_counts())
    
    # Visual checkpoint
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img_path = os.path.join(OUTPUT_DIR, 'images', dataset_df.iloc[i]['filename'])
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {dataset_df.iloc[i]['label']}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'dataset_gallery.png'))
    print(f"\nSaved dataset gallery to {os.path.join(OUTPUT_DIR, 'figures', 'dataset_gallery.png')}")
