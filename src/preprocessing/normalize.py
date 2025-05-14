import numpy as np

def normalize_ecg(ecg_data):
    """
    Normalize ECG data to have zero mean and unit variance.
    
    Args:
    ecg_data (numpy.ndarray): ECG data to normalize
    
    Returns:
    numpy.ndarray: Normalized ECG data
    """
    return (ecg_data - np.mean(ecg_data, axis=-1, keepdims=True)) / np.std(ecg_data, axis=-1, keepdims=True)