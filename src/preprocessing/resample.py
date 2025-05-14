from scipy.signal import resample_poly

def resample_ecg(ecg_data, original_fs, target_fs):
    """
    Resample ECG data to the target frequency.
    
    Args:
    ecg_data (numpy.ndarray): Original ECG data
    original_fs (int): Original sampling frequency in Hz
    target_fs (int): Target sampling frequency in Hz
    
    Returns:
    numpy.ndarray: Resampled ECG data
    """
    factor = target_fs / original_fs
    return resample_poly(ecg_data, up=target_fs, down=original_fs, axis=-1)