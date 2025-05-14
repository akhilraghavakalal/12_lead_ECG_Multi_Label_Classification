import numpy as np

def extract_windows(ecg_data, window_size, target_fs):
    """
    Extract fixed-size windows from ECG data, padding if necessary.
    
    Args:
    ecg_data (numpy.ndarray): ECG data
    window_size (int): Size of the window in seconds
    target_fs (int): Sampling frequency in Hz
    
    Returns:
    numpy.ndarray: ECG data windows
    """
    window_samples = window_size * target_fs
    
    if ecg_data.shape[-1] >= window_samples:
        return ecg_data[..., :window_samples]
    else:
        padding = np.zeros((ecg_data.shape[0], window_samples - ecg_data.shape[-1]))
        return np.concatenate([ecg_data, padding], axis=-1)