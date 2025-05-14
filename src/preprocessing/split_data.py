import os
import shutil
import random
import json
from collections import Counter
from tqdm import tqdm


def reorganize_data(source_dir, dest_dir, test_hospital):
    print(
        "\nReorganizing the data into train, validation and test folders for further steps"
    )

    # Create destination directories and ensure they're empty to avoid file overlap
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(dest_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)  # Remove existing directory to ensure clean slate
        os.makedirs(split_dir, exist_ok=True)

    # Get list of hospital directories
    hospitals = [
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    ]

    # Check if test_hospital exists
    if test_hospital not in hospitals:
        raise ValueError(
            f"Test hospital '{test_hospital}' not found! Available hospitals: {hospitals}"
        )

    # Filter hospitals for train/val
    train_val_hospitals = [h for h in hospitals if h != test_hospital]
    print(f"Using {test_hospital} for test set")
    print(f"Using {train_val_hospitals} for train/validation sets")

    # Move test hospital data
    test_source = os.path.join(source_dir, test_hospital)
    test_dest = os.path.join(dest_dir, "test")
    test_files_copied = 0

    for file in tqdm(os.listdir(test_source), desc="Moving test data"):
        if file.endswith(".npy") or file.endswith(".json"):
            shutil.copy(os.path.join(test_source, file), os.path.join(test_dest, file))
            test_files_copied += 1

    print(f"Copied {test_files_copied} files to test set")

    # Track files to ensure no duplication
    used_file_prefixes = set()

    # Split remaining data into train and validation
    train_labels = []
    val_labels = []
    total_files = sum(
        len(os.listdir(os.path.join(source_dir, h))) for h in train_val_hospitals
    )

    with tqdm(total=total_files, desc="Processing train/val data") as pbar:
        for hospital in train_val_hospitals:
            hospital_dir = os.path.join(source_dir, hospital)
            files = [
                f
                for f in os.listdir(hospital_dir)
                if f.endswith(".npy") or f.endswith(".json")
            ]

            # Pair .npy and .json files
            file_pairs = []
            for f in files:
                if f.endswith(".npy"):
                    json_f = f.replace(".npy", "_header.json")
                    if json_f in files:
                        file_prefix = f.split(".")[0]
                        # Skip files that have already been processed or might overlap with test set
                        if file_prefix not in used_file_prefixes:
                            file_pairs.append((f, json_f))
                            used_file_prefixes.add(file_prefix)

            # Randomly split pairs into train and validation
            random.shuffle(file_pairs)
            split_index = int(len(file_pairs) * 0.8)  # 80% train, 20% validation

            for i, (npy_f, json_f) in enumerate(file_pairs):
                source_npy = os.path.join(hospital_dir, npy_f)
                source_json = os.path.join(hospital_dir, json_f)

                if i < split_index:
                    dest_folder = os.path.join(dest_dir, "train")
                    labels = train_labels
                else:
                    dest_folder = os.path.join(dest_dir, "validation")
                    labels = val_labels

                shutil.copy(source_npy, os.path.join(dest_folder, npy_f))
                shutil.copy(source_json, os.path.join(dest_folder, json_f))

                # Collect labels
                with open(source_json, "r") as f:
                    header = json.load(f)
                    labels.extend(header.get("Dx", []))

                pbar.update(2)  # Update for both .npy and .json files

    # Verify no file overlap
    train_files = set(
        [
            f.split(".")[0]
            for f in os.listdir(os.path.join(dest_dir, "train"))
            if f.endswith(".npy")
        ]
    )
    val_files = set(
        [
            f.split(".")[0]
            for f in os.listdir(os.path.join(dest_dir, "validation"))
            if f.endswith(".npy")
        ]
    )
    test_files = set(
        [
            f.split(".")[0]
            for f in os.listdir(os.path.join(dest_dir, "test"))
            if f.endswith(".npy")
        ]
    )

    train_test_overlap = train_files.intersection(test_files)
    val_test_overlap = val_files.intersection(test_files)

    if train_test_overlap or val_test_overlap:
        print(f"WARNING: File overlap detected!")
        print(f"Train-test overlap: {len(train_test_overlap)} files")
        print(f"Val-test overlap: {len(val_test_overlap)} files")
    else:
        print("Verified: No file overlap between sets")

    # Print dataset sizes and label distributions
    print(f"\nTrain set size: {len(os.listdir(os.path.join(dest_dir, 'train'))) // 2}")
    print(
        f"Validation set size: {len(os.listdir(os.path.join(dest_dir, 'validation'))) // 2}"
    )
    print(f"Test set size: {len(os.listdir(test_dest)) // 2}")

    print("\nLabel distribution in train set:")
    print(Counter(train_labels))
    print("\nLabel distribution in validation set:")
    print(Counter(val_labels))

    print("Data reorganization complete.\n")
