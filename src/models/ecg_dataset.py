import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple, Dict

class ECGDataset(Dataset):
    def __init__(self, wide_features: np.ndarray, labels: np.ndarray, ecg_paths: List[str], window_size: int = 7500):
        self.wide_features = wide_features
        self.labels = labels
        self.ecg_paths = ecg_paths
        self.window_size = window_size
        self.nan_count = 0
        self.total_count = 0

        if len(wide_features) != len(labels) or len(wide_features) != len(ecg_paths):
            raise ValueError("Mismatch in the number of samples between features, labels, and ECG paths.")

    def __len__(self) -> int:
        return len(self.wide_features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wide_feature = self.wide_features[idx]
        label = self.labels[idx]
        
        try:
            ecg = np.load(self.ecg_paths[idx])
            self.total_count += 1
            
            if np.isnan(ecg).any():
                self.nan_count += 1
                ecg = np.nan_to_num(ecg, nan=0.0)
            
            if ecg.shape[1] >= self.window_size:
                start = np.random.randint(0, ecg.shape[1] - self.window_size + 1)
                ecg = ecg[:, start:start+self.window_size]
            else:
                pad_width = ((0, 0), (0, self.window_size - ecg.shape[1]))
                ecg = np.pad(ecg, pad_width, mode='constant', constant_values=0)
            
        except Exception as e:
            print(f"Error loading ECG file {self.ecg_paths[idx]}: {str(e)}")
            ecg = np.zeros((12, self.window_size))

        return torch.FloatTensor(wide_feature), torch.FloatTensor(ecg), torch.FloatTensor(label)

    def get_nan_percentage(self) -> float:
        if self.total_count == 0:
            return 0
        return (self.nan_count / self.total_count) * 100

    def validate_data(self) -> Dict[str, any]:
        validation_results = {
            "wide_features_shape": self.wide_features.shape,
            "labels_shape": self.labels.shape,
            "num_ecg_files": len(self.ecg_paths),
            "wide_features_stats": {
                "mean": np.mean(self.wide_features),
                "std": np.std(self.wide_features),
                "min": np.min(self.wide_features),
                "max": np.max(self.wide_features)
            },
            "labels_stats": {
                "num_classes": self.labels.shape[1],
                "avg_positive_labels": np.mean(np.sum(self.labels, axis=1))
            },
            "data_types": {
                "wide_features": self.wide_features.dtype,
                "labels": self.labels.dtype
            },
            "nan_percentage": self.get_nan_percentage()
        }
        return validation_results

def prepare_data(features, labels, ecg_paths, batch_size=32):
    dataset = ECGDataset(features, labels, ecg_paths)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,   
        pin_memory=True,  # Enables faster data transfer to GPU
        persistent_workers=True  # Keeps workers alive between iterations
    )
    return dataset, loader

def print_validation_results(dataset: ECGDataset, dataloader: DataLoader):
    validation_results = dataset.validate_data()
    
    print("\nData Validation Results:")
    print(f"Wide features shape: {validation_results['wide_features_shape']}")
    print(f"Labels shape: {validation_results['labels_shape']}")
    print(f"Number of ECG files: {validation_results['num_ecg_files']}")
    print(f"\nWide features statistics:")
    for key, value in validation_results['wide_features_stats'].items():
        print(f"  {key}: {value}")
    print(f"\nLabels statistics:")
    for key, value in validation_results['labels_stats'].items():
        print(f"  {key}: {value}")
    print(f"\nData types:")
    for key, value in validation_results['data_types'].items():
        print(f"  {key}: {value}")
    print(f"\nPercentage of samples with NaN values: {validation_results['nan_percentage']:.2f}%")
    
    print(f"\nNumber of batches in dataloader: {len(dataloader)}")
    
    # Inspect a sample batch
    for wide_features, ecg_data, labels in dataloader:
        print("\nSample batch shapes:")
        print(f"  Wide features: {wide_features.shape}")
        print(f"  ECG data: {ecg_data.shape}")
        print(f"  Labels: {labels.shape}")
        
        print("\nChecking for NaN or Inf values in sample batch:")
        print(f"  Any NaN in wide features: {torch.isnan(wide_features).any()}")
        print(f"  Any Inf in wide features: {torch.isinf(wide_features).any()}")
        print(f"  Any NaN in ECG data: {torch.isnan(ecg_data).any()}")
        print(f"  Any Inf in ECG data: {torch.isinf(ecg_data).any()}")
        break  # We only need to check one batch