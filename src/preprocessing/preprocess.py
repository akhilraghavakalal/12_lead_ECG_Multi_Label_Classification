import os
import numpy as np
from scipy.io import loadmat
from .resample import resample_ecg
from .filter import bandpass_filter
from .normalize import normalize_ecg
from .extract_windows import extract_windows
from ..utils.io_utils import load_header, save_processed_data

def preprocess_file(file_path, target_fs=500, window_size=15):
# def preprocess_file(file_path, target_fs=500, window_size=10):
    """
    Preprocess a single ECG file.
    
    Args:
    file_path (str): Path to the .mat file
    target_fs (int): Target sampling frequency in Hz
    window_size (int): Size of the window in seconds
    
    Returns:
    tuple: Preprocessed ECG windows and header information
    """
    # Load the ECG data
    ecg_data = loadmat(file_path)['val']  
    
    # Load header information
    header = load_header(file_path.replace('.mat', '.hea'))
    original_fs = header['fs']
    
    # Resample
    ecg_resampled = resample_ecg(ecg_data, original_fs, target_fs)
    
    # Apply bandpass filter
    ecg_filtered = bandpass_filter(ecg_resampled, lowcut=3, highcut=45, fs=target_fs)
    
    # Normalize
    ecg_normalized = normalize_ecg(ecg_filtered)
    
    # Extract windows
    ecg_windows = extract_windows(ecg_normalized, window_size=window_size, target_fs=target_fs)
    
    return ecg_windows, header

def process_dataset(input_dir, output_dir):
    """
    Process all ECG files in the input directory and save results to the output directory.
    
    Args:
    input_dir (str): Path to the directory containing original ECG files
    output_dir (str): Path to the directory where processed files will be saved
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mat'):
                file_path = os.path.join(root, file)
                try:
                    ecg_windows, header = preprocess_file(file_path)
                    
                    # Save processed data
                    output_path = os.path.join(output_dir, os.path.relpath(file_path, input_dir))
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    save_processed_data(output_path, ecg_windows, header)
                    print(f"Processed: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")