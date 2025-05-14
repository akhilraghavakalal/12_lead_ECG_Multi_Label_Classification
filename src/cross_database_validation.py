import logging
import os
import numpy as np
import torch
import shutil
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.io import loadmat

import torch.nn as nn

from .preprocessing.preprocess import process_dataset
from .preprocessing.filter import bandpass_filter
from .preprocessing.normalize import normalize_ecg
from .preprocessing.resample import resample_ecg
from .preprocessing.extract_windows import extract_windows
from .feature_extraction.wide_features import extract_wide_features, preprocess_ecg
from .feature_extraction.determine_classes import get_unique_classes
from .models.cnn_gru_model import CNNGRUModel
from .models.wide_deep_transformer_model import create_model
from .models.train_cnn_gru_model import train_cnn_gru_model
from .models.train_wide_deep_transformer_model import train_wide_deep_transformer
from .utils.io_utils import load_header

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    def __init__(self, features, labels, ecg_paths, sequence_length=7500):
        self.features = features
        self.labels = labels
        self.ecg_paths = ecg_paths
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features)

    def pad_or_truncate(self, data):
        """Standardize all ECG sequences to the same length"""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        current_length = data.shape[1]
        if current_length > self.sequence_length:
            return data[:, : self.sequence_length]
        elif current_length < self.sequence_length:
            padding = np.zeros((data.shape[0], self.sequence_length - current_length))
            return np.concatenate([data, padding], axis=1)
        return data

    def __getitem__(self, idx):
        try:
            feature = torch.FloatTensor(self.features[idx])
            label = torch.FloatTensor(self.labels[idx])

            # Load and standardize ECG data
            ecg_data = np.load(self.ecg_paths[idx])
            ecg_data = self.pad_or_truncate(ecg_data)
            if ecg_data.shape[0] != 12:  # Ensure 12 leads
                ecg_data = np.pad(ecg_data, ((0, 12 - ecg_data.shape[0]), (0, 0)))
            ecg_data = torch.FloatTensor(ecg_data)

            return feature, label, ecg_data
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            print(f"Path: {self.ecg_paths[idx]}")
            raise


def prepare_single_hospital(
    input_dir, hospital_processed, window_size=15, sampling_rate=500
):
    """Create directory structure for a single hospital with consistent window sizes"""
    processed_files = []

    try:
        # Create hospital output directory
        os.makedirs(hospital_processed, exist_ok=True)

        # Look for .mat files directly in the hospital directory
        mat_files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.endswith(".mat"):
                    mat_files.append(os.path.join(input_dir, f))

        if not mat_files:
            logger.error(f"No .mat files found in {input_dir}")
            return []

        # Process each file
        for mat_file in tqdm(
            mat_files, desc=f"Processing {os.path.basename(input_dir)}"
        ):
            try:
                # Load the .mat file
                mat_data = loadmat(str(mat_file))
                if "val" not in mat_data:
                    logger.warning(f"No 'val' field in {mat_file}")
                    continue

                ecg_data = mat_data["val"]

                header_file = str(mat_file).replace(".mat", ".hea")
                if not os.path.exists(header_file):
                    logger.warning(f"No header file found for {mat_file}")
                    continue

                # Load and process header
                header = load_header(header_file)

                # Standardize ECG data
                ecg_resampled = resample_ecg(ecg_data, header["fs"], sampling_rate)
                ecg_filtered = bandpass_filter(
                    ecg_resampled, lowcut=3, highcut=45, fs=sampling_rate
                )
                ecg_normalized = normalize_ecg(ecg_filtered)

                # Extract fixed window
                ecg_windows = extract_windows(
                    ecg_normalized, window_size=window_size, target_fs=sampling_rate
                )

                # Save processed data
                output_base = os.path.splitext(os.path.basename(mat_file))[0]
                output_path = os.path.join(hospital_processed, f"{output_base}.npy")
                np.save(output_path, ecg_windows)
                processed_files.append(output_path)

                # Save header
                header_path = os.path.join(
                    hospital_processed, f"{output_base}_header.json"
                )
                with open(header_path, "w") as f:
                    json.dump(header, f)

            except Exception as e:
                logger.error(f"Error processing {mat_file}: {str(e)}")
                continue

        return processed_files

    except Exception as e:
        logger.error(f"Error in prepare_single_hospital: {str(e)}")
        return []


def prepare_hospital_data(hospital_dir, hospital_name, temp_dir, sequence_length=7500):
    """
    Prepare data for a specific hospital including preprocessing

    Args:
        hospital_dir (str): Path to the hospital's raw data directory
        hospital_name (str): Name of the hospital for logging purposes
        temp_dir (str): Path to store processed data
        sequence_length (int): Target sequence length for ECG data

    Returns:
        tuple: (features, labels, valid_files) or (None, None, None) on error
    """
    hospital_processed = os.path.join(temp_dir, hospital_name)

    logger.info(f"\nProcessing data for {hospital_name}")
    try:
        # Create processed directory
        os.makedirs(hospital_processed, exist_ok=True)

        # Get list of .mat files from hospital directory
        mat_files = []
        if os.path.exists(hospital_dir):
            mat_files = [f for f in os.listdir(hospital_dir) if f.endswith(".mat")]

        if not mat_files:
            logger.error(f"No .mat files found in {hospital_dir}")
            return None, None, None

        # Process each file
        processed_files = []
        window_size = sequence_length // 500  # Convert to seconds

        for mat_file in tqdm(mat_files, desc=f"Processing {hospital_name} files"):
            try:
                # Load ECG data
                input_path = os.path.join(hospital_dir, mat_file)
                mat_data = loadmat(str(input_path))
                if "val" not in mat_data:
                    logger.warning(f"No 'val' field in {mat_file}")
                    continue
                ecg_data = mat_data["val"]

                # Load header file
                header_file = input_path.replace(".mat", ".hea")
                if not os.path.exists(header_file):
                    logger.warning(f"No header file found for {mat_file}")
                    continue

                # Process header
                header = load_header(header_file)
                if header is None:
                    logger.warning(f"Could not load header for {mat_file}")
                    continue

                # Preprocess ECG data
                try:
                    # Resample to 500 Hz
                    ecg_resampled = resample_ecg(ecg_data, header["fs"], 500)

                    # Apply bandpass filter
                    ecg_filtered = bandpass_filter(
                        ecg_resampled, lowcut=3, highcut=45, fs=500
                    )

                    # Normalize
                    ecg_normalized = normalize_ecg(ecg_filtered)

                    # Extract fixed window
                    ecg_windows = extract_windows(
                        ecg_normalized, window_size=window_size, target_fs=500
                    )

                    # Save processed data
                    output_base = os.path.splitext(mat_file)[0]
                    output_path = os.path.join(hospital_processed, f"{output_base}.npy")
                    np.save(output_path, ecg_windows)
                    processed_files.append(output_path)

                    # Save processed header
                    header_out_path = output_path.replace(".npy", "_header.json")
                    with open(header_out_path, "w") as f:
                        json.dump(header, f)

                except Exception as e:
                    logger.error(f"Error in preprocessing {mat_file}: {str(e)}")
                    continue

            except Exception as e:
                logger.error(f"Error processing {mat_file}: {str(e)}")
                continue

        if not processed_files:
            logger.error(f"No files successfully processed for {hospital_name}")
            return None, None, None

        # Extract features and create labels
        features_list = []
        labels_list = []
        valid_files = []

        # Get classes for label encoding
        CLASSES = get_unique_classes(os.path.dirname(hospital_dir))

        for npy_path in tqdm(
            processed_files, desc=f"Extracting features for {hospital_name}"
        ):
            try:
                # Load processed ECG data and extract features
                ecg_data = np.load(npy_path)
                features = extract_wide_features(ecg_data)

                # Get labels from header
                header_path = npy_path.replace(".npy", "_header.json")
                if os.path.exists(header_path):
                    with open(header_path, "r") as f:
                        header = json.load(f)
                        dx = header.get("Dx", [])
                        if isinstance(dx, str):
                            dx = dx.split(",")

                        # Create one-hot encoding
                        label = np.zeros(len(CLASSES))
                        for d in dx:
                            if d in CLASSES:
                                label[CLASSES.index(d)] = 1

                        # Only add if features and labels are valid
                        if features is not None and len(features) > 0:
                            features_list.append(features)
                            labels_list.append(label)
                            valid_files.append(npy_path)

            except Exception as e:
                logger.error(f"Error extracting features from {npy_path}: {str(e)}")
                continue

        if not features_list:
            logger.error(f"No valid features extracted for {hospital_name}")
            return None, None, None

        # Convert to numpy arrays
        features = np.array(features_list)
        labels = np.array(labels_list)

        # Ensure consistent dimensions
        if features.shape[1] > 9:
            features = features[:, :9]
        elif features.shape[1] < 9:
            pad_width = ((0, 0), (0, 9 - features.shape[1]))
            features = np.pad(features, pad_width, mode="constant", constant_values=0)

        logger.info(
            f"Successfully processed {len(features)} files from {hospital_name}"
        )
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")

        return features, labels, valid_files

    except Exception as e:
        logger.error(f"Error processing hospital {hospital_name}: {str(e)}")
        return None, None, None


def prepare_data(
    features, labels, ecg_paths, batch_size=32, sequence_length=7500, shuffle=True
):
    """Prepare data loaders with standardized sequence length"""
    dataset = ECGDataset(features, labels, ecg_paths, sequence_length=sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def cross_database_validation(args):
    """Perform cross-database validation with consistent window sizes"""
    logger.info("Starting cross-database validation...")

    hospitals = [
        "cpsc_2018",
        "cpsc_2018_extra",
        "georgia",
        "ptb",
        "ptb-xl",
        "st_petersburg_incart",
    ]
    results = {}

    # Define standard window size
    WINDOW_SIZE = 15
    SAMPLING_RATE = 500
    SEQUENCE_LENGTH = WINDOW_SIZE * SAMPLING_RATE

    temp_dir = os.path.join(args.output_dir, "temp_cross_val")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    try:
        # First pass: Process all hospitals and collect data
        hospital_data = {}
        for hospital in hospitals:
            logger.info(f"Processing {hospital}")

            # Build correct paths
            hospital_dir = os.path.join(args.input_dir, hospital)
            if not os.path.exists(hospital_dir):
                logger.error(f"Hospital directory not found: {hospital_dir}")
                continue

            features, labels, ecg_paths = prepare_hospital_data(
                hospital_dir, hospital, temp_dir, sequence_length=SEQUENCE_LENGTH
            )

            if all(x is not None for x in [features, labels, ecg_paths]):
                if len(features) > 0 and len(labels) > 0 and len(ecg_paths) > 0:
                    hospital_data[hospital] = (features, labels, ecg_paths)
                    logger.info(
                        f"{hospital} processed: features {features.shape}, "
                        f"labels {labels.shape}, ecg_paths {len(ecg_paths)}"
                    )
                else:
                    logger.error(f"Empty data for {hospital}")
            else:
                logger.error(f"Failed to process {hospital}")

        if not hospital_data:
            logger.error("No hospitals processed successfully")
            return {}

        # Second pass: Cross-validation
        for test_hospital in hospitals:
            logger.info(f"\nValidating on {test_hospital}")
            logger.info("-" * 50)

            if test_hospital not in hospital_data:
                logger.warning(f"Skipping {test_hospital} - no data available")
                continue

            # Get test data
            test_features, test_labels, test_ecg_paths = hospital_data[test_hospital]

            # Combine other hospitals' data for training
            train_features_list = []
            train_labels_list = []
            train_ecg_paths_list = []

            for train_hospital in hospitals:
                if train_hospital != test_hospital and train_hospital in hospital_data:
                    features, labels, ecg_paths = hospital_data[train_hospital]
                    train_features_list.append(features)
                    train_labels_list.append(labels)
                    train_ecg_paths_list.extend(ecg_paths)
                    logger.info(
                        f"Added {train_hospital} data: {len(ecg_paths)} samples"
                    )

            if not train_features_list:
                logger.warning(f"No training data available for {test_hospital}")
                continue

            # Combine training data
            train_features = np.concatenate(train_features_list)
            train_labels = np.concatenate(train_labels_list)

            logger.info(f"Combined training data shape: {train_features.shape}")
            logger.info(f"Test data shape: {test_features.shape}")

            # Split training data into train/val
            train_idx, val_idx = train_test_split(
                range(len(train_features)), test_size=0.15, random_state=42
            )

            # Prepare training data
            train_features_split = train_features[train_idx]
            train_labels_split = train_labels[train_idx]
            train_ecg_paths_split = [train_ecg_paths_list[i] for i in train_idx]

            # Prepare validation data
            val_features_split = train_features[val_idx]
            val_labels_split = train_labels[val_idx]
            val_ecg_paths_split = [train_ecg_paths_list[i] for i in val_idx]

            # Create data loaders
            train_dataset, train_loader = prepare_data(
                train_features_split,
                train_labels_split,
                train_ecg_paths_split,
                batch_size=args.batch_size,
                sequence_length=SEQUENCE_LENGTH,
                shuffle=True,
            )

            val_dataset, val_loader = prepare_data(
                val_features_split,
                val_labels_split,
                val_ecg_paths_split,
                batch_size=args.batch_size,
                sequence_length=SEQUENCE_LENGTH,
                shuffle=False,
            )

            test_dataset, test_loader = prepare_data(
                test_features,
                test_labels,
                test_ecg_paths,
                batch_size=args.batch_size,
                sequence_length=SEQUENCE_LENGTH,
                shuffle=False,
            )

            hospital_results = {}

            # Train CNN-GRU model if selected
            if args.model in ["cnn_gru", "both"]:
                logger.info(f"Training CNN-GRU model for {test_hospital}")
                try:
                    model = CNNGRUModel().to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                    criterion = nn.BCELoss()
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", patience=args.patience // 2
                    )

                    cnn_gru_results = train_cnn_gru_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        scheduler=scheduler,
                        device=device,
                        save_dir=os.path.join(
                            args.model_dir, f"cv_{test_hospital}_cnn_gru"
                        ),
                        num_epochs=args.num_epochs,
                        patience=args.patience,
                    )
                    hospital_results["cnn_gru"] = {
                        "test_auc": cnn_gru_results[0],
                        "train_losses": cnn_gru_results[1],
                        "val_losses": cnn_gru_results[2],
                        "val_aucs": cnn_gru_results[3],
                    }
                except Exception as e:
                    logger.error(f"Error training CNN-GRU model: {str(e)}")

            # Train Transformer model if selected
            if args.model in ["transformer", "both"]:
                logger.info(f"Training Transformer model for {test_hospital}")
                try:
                    model = create_model(
                        wide_dim=train_features.shape[1],
                        n_classes=train_labels.shape[1],
                        sequence_length=SEQUENCE_LENGTH,
                    ).to(device)

                    transformer_results = train_wide_deep_transformer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        model_size=args.model_size,
                        warmup_steps=args.warmup_steps,
                        device=device,
                        save_dir=os.path.join(
                            args.model_dir, f"cv_{test_hospital}_transformer"
                        ),
                        num_epochs=args.num_epochs,
                        patience=args.patience,
                    )
                    hospital_results["transformer"] = {
                        "test_auc": transformer_results[0],
                        "train_losses": transformer_results[1],
                        "val_losses": transformer_results[2],
                        "val_aucs": transformer_results[3],
                    }
                except Exception as e:
                    logger.error(f"Error training Transformer model: {str(e)}")

            if hospital_results:
                results[test_hospital] = hospital_results
                logger.info(f"Completed validation on {test_hospital}")
            logger.info("-" * 50)

        # Calculate and log final results
        summary = {}
        for model_type in ["cnn_gru", "transformer"]:
            if args.model in [model_type, "both"]:
                scores = []
                for hospital, hospital_results in results.items():
                    if model_type in hospital_results:
                        scores.append(hospital_results[model_type]["test_auc"])

                if scores:
                    avg_score = np.mean(scores)
                    std_score = np.std(scores)
                    logger.info(f"\n{model_type.upper()}:")
                    logger.info(f"Average Test AUC: {avg_score:.4f} ± {std_score:.4f}")
                    logger.info("Individual Results:")
                    for hospital, score in zip(results.keys(), scores):
                        logger.info(f"{hospital}: {score:.4f}")

                    summary[model_type] = {
                        "average_auc": float(avg_score),
                        "std_auc": float(std_score),
                        "individual_scores": {
                            h: float(s) for h, s in zip(results.keys(), scores)
                        },
                    }

    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}", exc_info=True)
        return {}
    finally:
        # Cleanup temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")

    return summary


# Optional helper functions for data verification
def verify_data_consistency(features, labels, ecg_paths, sequence_length):
    """Verify data consistency before training"""
    try:
        # Check shapes
        assert len(features) == len(labels) == len(ecg_paths), "Mismatched data lengths"

        # Verify ECG data
        sample_ecg = np.load(ecg_paths[0])
        assert (
            sample_ecg.shape[1] == sequence_length
        ), f"Incorrect sequence length: {sample_ecg.shape[1]} vs {sequence_length}"

        # Check for NaN values
        assert not np.isnan(features).any(), "NaN values in features"
        assert not np.isnan(labels).any(), "NaN values in labels"

        return True
    except Exception as e:
        logger.error(f"Data verification failed: {str(e)}")
        return False


def print_data_summary(hospital_data):
    """Print summary of processed hospital data"""
    logger.info("\nData Summary:")
    logger.info("=" * 30)

    total_samples = 0
    for hospital, (features, labels, paths) in hospital_data.items():
        logger.info(f"\nHospital: {hospital}")
        logger.info(f"Number of samples: {len(features)}")
        logger.info(f"Feature shape: {features.shape}")
        logger.info(f"Label shape: {labels.shape}")
        logger.info(f"Positive labels: {np.sum(labels, axis=0)}")

        total_samples += len(features)

    logger.info(f"\nTotal samples across all hospitals: {total_samples}")
    logger.info("=" * 30)
