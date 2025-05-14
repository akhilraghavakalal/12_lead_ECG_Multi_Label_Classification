import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import sys


def check_for_data_leakage(final_data_dir, test_hospital="ptb-xl"):
    """
    Comprehensive check for potential data leakage in the ECG classification pipeline.

    Args:
        final_data_dir (str): Path to the directory containing train, validation, and test data
        test_hospital (str): The hospital reserved for testing
    """
    print("Running comprehensive data leakage detection...")

    # 1. Check file separation between sets
    train_dir = os.path.join(final_data_dir, "train")
    val_dir = os.path.join(final_data_dir, "validation")
    test_dir = os.path.join(final_data_dir, "test")

    print("\n1. Checking dataset separation...")

    train_files = [f for f in os.listdir(train_dir) if f.endswith(".npy")]
    val_files = [f for f in os.listdir(val_dir) if f.endswith(".npy")]
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".npy")]

    print(f"  Number of training files: {len(train_files)}")
    print(f"  Number of validation files: {len(val_files)}")
    print(f"  Number of test files: {len(test_files)}")

    # Check for file name overlaps
    train_test_overlap = set([f.split(".")[0] for f in train_files]).intersection(
        set([f.split(".")[0] for f in test_files])
    )
    val_test_overlap = set([f.split(".")[0] for f in val_files]).intersection(
        set([f.split(".")[0] for f in test_files])
    )

    print(f"  File name overlap between train and test: {len(train_test_overlap)}")
    print(f"  File name overlap between validation and test: {len(val_test_overlap)}")

    # 2. Verify test set origin
    print("\n2. Verifying test set origin...")
    sample_size = min(10, len(test_files))
    sampled_test_files = random.sample(test_files, sample_size)

    hospital_counts = defaultdict(int)
    for test_file in sampled_test_files:
        json_file = os.path.join(test_dir, test_file.replace(".npy", "_header.json"))
        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    header = json.load(f)
                    # The field containing hospital info might vary based on your data structure
                    # Try different possible fields
                    hospital = None
                    for field in ["hospital", "origin", "source", "database"]:
                        if field in header:
                            hospital = header[field]
                            break

                    if hospital:
                        hospital_counts[hospital] += 1
                    else:
                        hospital_counts["unknown"] += 1
            except Exception as e:
                print(f"  Error reading {json_file}: {str(e)}")
                hospital_counts["error"] += 1

    print(f"  Test files hospital distribution (sample of {sample_size}):")
    for hospital, count in hospital_counts.items():
        print(f"    {hospital}: {count} files ({count / sample_size * 100:.1f}%)")

    non_test_hospital_count = sum(
        count
        for hospital, count in hospital_counts.items()
        if hospital != test_hospital and hospital != "unknown" and hospital != "error"
    )

    if non_test_hospital_count > 0:
        print(
            f"  WARNING: {non_test_hospital_count} test files appear to be from hospitals other than {test_hospital}"
        )
    else:
        print(f"   All identifiable test files appear to be from {test_hospital}")

    # 3. Check for feature extraction leakage
    print("\n3. Checking for feature extraction leakage...")

    try:
        train_features = np.load(
            os.path.join(final_data_dir, "train_wide_features.npy")
        )
        val_features = np.load(os.path.join(final_data_dir, "val_wide_features.npy"))

        print(f"  Train features shape: {train_features.shape}")
        print(f"  Validation features shape: {val_features.shape}")

        # Compare train and validation feature distributions
        train_mean = np.mean(train_features, axis=0)
        val_mean = np.mean(val_features, axis=0)
        train_std = np.std(train_features, axis=0)
        val_std = np.std(val_features, axis=0)

        # If normalized correctly, distributions should differ somewhat
        mean_diff = np.abs(train_mean - val_mean)
        std_diff = np.abs(train_std - val_std)

        print(
            f"  Mean absolute difference between train and val feature means: {np.mean(mean_diff):.4f}"
        )
        print(
            f"  Mean absolute difference between train and val feature stds: {np.mean(std_diff):.4f}"
        )

        if np.mean(mean_diff) < 0.01 and np.mean(std_diff) < 0.01:
            print(
                "  WARNING: Train and validation features are very similar, suggesting possible leakage"
            )
        else:
            print("   Train and validation features show expected differences")

        # Check if a single scaling factor is applied across all sets
        # If so, this could indicate that normalization was applied globally

    except Exception as e:
        print(f"  Error analyzing features: {str(e)}")

    # 4. Check preprocessing consistency
    print("\n4. Checking preprocessing consistency...")

    # Sample a few files from each set to check their properties
    sample_train = random.sample(train_files, min(5, len(train_files)))
    sample_val = random.sample(val_files, min(5, len(val_files)))
    sample_test = random.sample(test_files, min(5, len(test_files)))

    def analyze_samples(dir_path, files):
        results = []
        for file in files:
            file_path = os.path.join(dir_path, file)
            try:
                data = np.load(file_path)
                results.append(
                    {
                        "shape": data.shape,
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                    }
                )
            except Exception as e:
                print(f"  Error analyzing {file_path}: {str(e)}")
        return results

    train_analysis = analyze_samples(train_dir, sample_train)
    val_analysis = analyze_samples(val_dir, sample_val)
    test_analysis = analyze_samples(test_dir, sample_test)

    # Check if preprocessing is consistently applied
    train_means = [a["mean"] for a in train_analysis]
    val_means = [a["mean"] for a in val_analysis]
    test_means = [a["mean"] for a in test_analysis]

    train_stds = [a["std"] for a in train_analysis]
    val_stds = [a["std"] for a in val_analysis]
    test_stds = [a["std"] for a in test_analysis]

    print(
        f"  Train sample means range: {min(train_means):.3f} to {max(train_means):.3f}"
    )
    print(f"  Val sample means range: {min(val_means):.3f} to {max(val_means):.3f}")
    print(f"  Test sample means range: {min(test_means):.3f} to {max(test_means):.3f}")

    print(f"  Train sample stds range: {min(train_stds):.3f} to {max(train_stds):.3f}")
    print(f"  Val sample stds range: {min(val_stds):.3f} to {max(val_stds):.3f}")
    print(f"  Test sample stds range: {min(test_stds):.3f} to {max(test_stds):.3f}")

    # 5. Check for correlation between test performance and train similarity
    print("\n5. Summary and recommendations...")

    # Based on the collected data, provide a summary of findings
    issues_found = []

    if len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
        issues_found.append("File overlap between train/val and test sets")

    if non_test_hospital_count > 0:
        issues_found.append(f"Test files from hospitals other than {test_hospital}")

    if np.mean(mean_diff) < 0.01 and np.mean(std_diff) < 0.01:
        issues_found.append("Suspiciously similar feature distributions")

    # Check for preprocessing consistency
    train_val_means_gap = abs(np.mean(train_means) - np.mean(val_means))
    train_test_means_gap = abs(np.mean(train_means) - np.mean(test_means))

    if train_val_means_gap > 0.5 and train_test_means_gap < 0.1:
        issues_found.append(
            "Test data preprocessing appears more similar to training than validation"
        )

    if not issues_found:
        print("   No obvious signs of data leakage detected")
        print("   Your dataset separation appears to be working correctly")
    else:
        print(" Potential issues found:")
        for issue in issues_found:
            print(f"    - {issue}")

    print("\nRecommendations:")
    if issues_found:
        print(
            "  1. Review your split_data.py implementation to ensure test_hospital data is properly isolated"
        )
        print(
            "  2. Check normalization in normalize.py to ensure it's applied separately for each set"
        )
        print("  3. Verify that feature selection uses only training data")
        print("  4. Run a simple baseline model as a sanity check on your data split")
    else:
        print(
            "  1. The high performance might be due to other factors than data leakage:"
        )
        print(
            "     - Your model architecture might be particularly well-suited for this task"
        )
        print("     - The test set might be easier to classify than expected")
        print(
            "     - Consider validating on a completely external dataset if available"
        )
        print(
            "  2. Review your metrics calculation - ensure predictions and ground truth align correctly"
        )


if __name__ == "__main__":
    path = sys.argv[1]
    final_data_dir = path
    check_for_data_leakage(final_data_dir)
