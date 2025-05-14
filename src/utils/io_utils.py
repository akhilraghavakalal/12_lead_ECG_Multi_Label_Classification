import json
import numpy as np

def load_header(header_file):
    """
    Load header information from a .hea file.
    
    Args:
    header_file (str): Path to the .hea file
    
    Returns:
    dict: Header information
    """
    header = {}
    with open(header_file, 'r') as f:
        lines = f.readlines()
        header['fs'] = int(lines[0].split()[2])
        header['n_leads'] = int(lines[0].split()[1])
        
        # Extract additional information
        for line in lines[1:]:
            if line.startswith('#'):
                key, value = line.strip('#').split(':', 1)
                header[key.strip()] = value.strip()
    
    return header

def save_processed_data(output_path, ecg_windows, header):
    """
    Save processed ECG data and header information.
    
    Args:
    output_path (str): Path to save the processed data
    ecg_windows (numpy.ndarray): Processed ECG data
    header (dict): Header information
    """
    np.save(output_path.replace('.mat', '.npy'), ecg_windows)
    with open(output_path.replace('.mat', '_header.json'), 'w') as f:
        json.dump(header, f, indent=4)

def load_processed_data(data_path):
    """
    Load processed ECG data and header information.
    
    Args:
    data_path (str): Path to the processed .npy file
    
    Returns:
    tuple: (ecg_windows, header)
    """
    ecg_windows = np.load(data_path)
    with open(data_path.replace('.npy', '_header.json'), 'r') as f:
        header = json.load(f)
    
    return ecg_windows, header