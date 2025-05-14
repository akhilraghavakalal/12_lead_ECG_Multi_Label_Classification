import matplotlib

matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import random


def get_random_file(root_dir):
    """Randomly select a .npy file from the nested folder structure."""
    all_npy_files = []
    for root, dirs, files in os.walk(root_dir):
        npy_files = [os.path.join(root, f) for f in files if f.endswith(".npy")]
        all_npy_files.extend(npy_files)

    if not all_npy_files:
        return None
    return random.choice(all_npy_files)


def visualize_sample(processed_data_dir):
    print("Inside visualize.py file")
    print(f"Root directory: {processed_data_dir}")

    sample_path = get_random_file(processed_data_dir)
    if not sample_path:
        print("No .npy files found in the directory structure.")
        return

    header_path = sample_path.replace(".npy", "_header.json")

    print(f"Selected sample file: {sample_path}")
    print(f"Header path: {header_path}")

    # Load the data and header
    try:
        ecg_data = np.load(sample_path)
        print(f"Successfully loaded NPY file. Shape: {ecg_data.shape}")

        with open(header_path, "r") as f:
            header = json.load(f)
        print("Successfully loaded header file")
    except Exception as e:
        print(f"Error loading files: {str(e)}")
        return

    # Plot the data
    fig, axes = plt.subplots(6, 2, figsize=(20, 30))
    fig.suptitle(f"Sample ECG: {os.path.basename(sample_path)}", fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        if i < ecg_data.shape[0]:
            ax.plot(ecg_data[i])
            ax.set_title(f"Lead {i+1}")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Normalized Amplitude")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_ecg_visualization.png")
    plt.close()

    print("Visualization saved as sample_ecg_visualization.png")
    print(f"Header information: {header}")
