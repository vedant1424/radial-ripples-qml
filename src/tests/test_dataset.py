
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from src.data.generate_ripples import render_ripple, build_dataset

def test_render_shape_and_range():
    """generate a single image and assert image.shape == (H,W) and image.min() >= 0.0 and image.max() <= 1.0."""
    params = np.array([0.8, 4.0, 1.0, np.pi, 0.5, 0.5])
    img = render_ripple(params, size=(32, 32))
    
    assert img.shape == (32, 32), f"Image shape is {img.shape}, expected (32, 32)"
    assert img.min() >= 0.0, f"Image min value is {img.min()}, expected >= 0.0"
    assert img.max() <= 1.0, f"Image max value is {img.max()}, expected <= 1.0"

def test_label_balance():
    """generate a sample of 500 images and ensure both classes appear and that proportion is between 0.25 and 0.75."""
    df = build_dataset(n=500, out_dir="./temp_test_data", size=(32, 32))
    
    class_counts = df['label'].value_counts(normalize=True)
    
    assert len(class_counts) == 2, "Only one class found in the dataset"
    assert 0.25 <= class_counts.min() <= 0.75, f"Class imbalance is too high: {class_counts.to_dict()}"
    assert 0.25 <= class_counts.max() <= 0.75, f"Class imbalance is too high: {class_counts.to_dict()}"

    # Clean up the temporary directory
    import shutil
    shutil.rmtree("./temp_test_data")

if __name__ == '__main__':
    test_render_shape_and_range()
    test_label_balance()
    print("All dataset tests passed!")
