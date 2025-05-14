import os
import numpy as np
import json
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .determine_classes import get_unique_classes

# Global variable for CLASSES, will be set in process_wide_features
CLASSES = []

def load_data(data_dir, subset):
    ecg_paths = []
    labels = []
    subset_dir = os.path.join(data_dir, subset)
    npy_files = [f for f in os.listdir(subset_dir) if f.endswith('.npy')]

    for npy_file in tqdm(npy_files, desc=f"Loading {subset} ECG data", unit="file"):
        npy_path = os.path.join(subset_dir, npy_file)
        json_path = npy_path.replace('.npy', '_header.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                header = json.load(f)
            
            ecg_paths.append(npy_path)
            dx = header.get('Dx', [])
            if isinstance(dx, str):
                dx = dx.split(',')
            label = [1 if cls in dx else 0 for cls in CLASSES]
            labels.append(label)

    return ecg_paths, np.array(labels)

def preprocess_ecg(ecg):
    sos = signal.butter(10, [3, 45], btype='bandpass', fs=500, output='sos')
    filtered_ecg = signal.sosfilt(sos, ecg)
    normalized_ecg = np.array([2 * (lead - np.min(lead)) / (np.max(lead) - np.min(lead)) - 1 for lead in filtered_ecg])
    return normalized_ecg

def extract_wide_features(ecg_path):
    ecg = np.load(ecg_path)
    ecg = preprocess_ecg(ecg)
    lead_ii = ecg[1]  # Assuming lead II is at index 1
    
    features = []
    
    # Time domain features
    features.extend([
        np.min(lead_ii), np.max(lead_ii), np.mean(lead_ii), np.std(lead_ii),
        skew(lead_ii), kurtosis(lead_ii), np.ptp(lead_ii)
    ])
    
    # Heart rate variability features
    peaks, _ = signal.find_peaks(lead_ii, distance=50)
    rr_intervals = np.diff(peaks)
    
    if len(rr_intervals) > 1:
        heart_rate = 60 / (np.mean(rr_intervals) / 500)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
        
        features.extend([heart_rate, sdnn, rmssd, pnn50])
        
        # Additional HRV features
        nn50 = sum(np.abs(np.diff(rr_intervals)) > 50)
        features.extend([
            np.median(rr_intervals),
            np.max(rr_intervals) - np.min(rr_intervals),
            nn50,
            nn50 / len(rr_intervals),
        ])
    else:
        features.extend([0] * 8)  # Append zeros if not enough RR intervals
    
    # Frequency domain features
    f, psd = signal.welch(lead_ii, fs=500, nperseg=2048)
    lf_power = np.sum(psd[(f >= 0.04) & (f < 0.15)])
    hf_power = np.sum(psd[(f >= 0.15) & (f < 0.4)])
    total_power = np.sum(psd[(f >= 0.04) & (f < 0.4)])
    features.extend([
        lf_power, hf_power, lf_power/hf_power if hf_power != 0 else 0,
        lf_power/total_power, hf_power/total_power
    ])
    
    # Morphological features
    q_wave = np.min(lead_ii[:int(len(lead_ii)*0.1)])
    r_wave = np.max(lead_ii[int(len(lead_ii)*0.1):int(len(lead_ii)*0.4)])
    s_wave = np.min(lead_ii[int(len(lead_ii)*0.4):int(len(lead_ii)*0.5)])
    t_wave = np.max(lead_ii[int(len(lead_ii)*0.5):])
    features.extend([q_wave, r_wave, s_wave, t_wave, r_wave - s_wave])
    
    return features

def select_top_features(features, labels, n_features=20):
    print("Selecting top features using Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    importance = rf.feature_importances_
    top_indices = importance.argsort()[-n_features:][::-1]
    for idx, score in zip(top_indices, importance[top_indices]):
        print(f"Feature {idx}: Importance Score {score}")
    return top_indices

def check_data_format(train_features, train_labels, val_features, val_labels):
    print("\nChecking data format...")
    
    # Check shapes
    print(f"Train features shape: {train_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation features shape: {val_features.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    
    # Check data types
    print(f"Train features dtype: {train_features.dtype}")
    print(f"Train labels dtype: {train_labels.dtype}")
    print(f"Validation features dtype: {val_features.dtype}")
    print(f"Validation labels dtype: {val_labels.dtype}")
    
    # Check value ranges
    print(f"Train features range: ({train_features.min()}, {train_features.max()})")
    print(f"Train labels range: ({train_labels.min()}, {train_labels.max()})")
    print(f"Validation features range: ({val_features.min()}, {val_features.max()})")
    print(f"Validation labels range: ({val_labels.min()}, {val_labels.max()})")

def process_wide_features(data_dir, batch_size=1000):
    print("\nStarting wide feature extraction...")
    
    # Get the correct CLASSES
    global CLASSES
    CLASSES = get_unique_classes(data_dir)
    print("Using the following classes:", CLASSES)
    
    train_ecg_paths, train_labels = load_data(data_dir, 'train')
    val_ecg_paths, val_labels = load_data(data_dir, 'validation')
    
    def process_batch(paths, desc):
        features = []
        for i in tqdm(range(0, len(paths), batch_size), desc=desc):
            batch_paths = paths[i:i+batch_size]
            batch_features = [extract_wide_features(path) for path in batch_paths]
            features.extend(batch_features)
        return np.array(features)
    
    train_features = process_batch(train_ecg_paths, "Extracting train features")
    val_features = process_batch(val_ecg_paths, "Extracting validation features")
    
    # Handle NaN and infinite values
    imputer = SimpleImputer(strategy='mean')
    train_features = imputer.fit_transform(train_features)
    val_features = imputer.transform(val_features)
    
    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    
    print("Selecting top features...")
    top_indices = select_top_features(train_features, train_labels)
    
    selected_train_features = train_features[:, top_indices]
    selected_val_features = val_features[:, top_indices]
    
    print("Wide feature extraction complete.")
    print(f"Train features shape: {selected_train_features.shape}")
    print(f"Validation features shape: {selected_val_features.shape}")
    
    print("Sample of train features:")
    print(selected_train_features[:5])
    print("\nSample of validation features:")
    print(selected_val_features[:5])
    
    print("Train features statistics:")
    print("Mean:", np.mean(selected_train_features))
    print("Std:", np.std(selected_train_features))
    print("Min:", np.min(selected_train_features))
    print("Max:", np.max(selected_train_features))

    print("\nValidation features statistics:")
    print("Mean:", np.mean(selected_val_features))
    print("Std:", np.std(selected_val_features))
    print("Min:", np.min(selected_val_features))
    print("Max:", np.max(selected_val_features))

    print("\nLabel information:")
    print("Number of classes:", len(CLASSES))
    print("Average number of positive labels per sample:", np.mean(np.sum(train_labels, axis=1)))
    
    check_data_format(selected_train_features, train_labels, selected_val_features, val_labels)
    
    return (selected_train_features, train_labels), (selected_val_features, val_labels), top_indices