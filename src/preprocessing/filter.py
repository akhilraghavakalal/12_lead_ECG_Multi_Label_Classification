from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the ECG data.
    
    Args:
    data (numpy.ndarray): ECG data to filter
    lowcut (float): Lower frequency bound in Hz
    highcut (float): Upper frequency bound in Hz
    fs (int): Sampling frequency in Hz
    order (int): Order of the filter
    
    Returns:
    numpy.ndarray: Filtered ECG data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)