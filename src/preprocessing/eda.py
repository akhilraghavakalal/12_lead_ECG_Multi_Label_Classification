import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import itertools


def load_data(data_dir):
    """
    Load ECG data and labels from directory.
    
    Args:
        data_dir: Directory containing ECG data files
        
    Returns:
        ecg_data: List of ECG recordings
        labels: List of label sets corresponding to each recording
    """
    ecg_data = []
    labels = []
    npy_files = []
    for root, dirs, files in os.walk(data_dir):
        npy_files.extend(
            [
                os.path.join(root, f)
                for f in files
                if f.endswith(".npy") and not f.endswith("_header.npy")
            ]
        )

    for npy_path in tqdm(npy_files, desc="Loading ECG data", unit="file"):
        json_path = npy_path.replace(".npy", "_header.json")

        if os.path.exists(json_path):
            ecg = np.load(npy_path)
            with open(json_path, "r") as f:
                header = json.load(f)

            ecg_data.append(ecg)
            labels.append(header.get("Dx", []))

    return ecg_data, labels


def calculate_signal_stats(ecg):
    """
    Calculate statistical features for ECG signal.
    
    Args:
        ecg: ECG recording with shape (leads, time_points)
        
    Returns:
        List of statistical measures for each lead
    """
    stats = []
    for lead in ecg:
        stats.append(
            {
                "mean": np.mean(lead),
                "std": np.std(lead),
                "max": np.max(lead),
                "min": np.min(lead),
                "skewness": skew(lead),
                "kurtosis": kurtosis(lead),
                "median": np.median(lead),
                "p2p": np.max(lead) - np.min(lead),  # peak-to-peak amplitude
                "rms": np.sqrt(np.mean(np.square(lead))),  # root mean square
                "energy": np.sum(np.square(lead)),  # signal energy
            }
        )
    return stats


def analyze_frequency_components(ecg, fs=500):
    """
    Analyze frequency components of ECG signal using Welch's method.
    
    Args:
        ecg: ECG recording with shape (leads, time_points)
        fs: Sampling frequency in Hz
        
    Returns:
        List of frequency-domain features for each lead
    """
    psd_stats = []
    for lead in ecg:
        f, psd = welch(lead, fs, nperseg=min(1024, len(lead)))
        
        # Find frequency bands
        # Delta: 0.5-4Hz, Theta: 4-8Hz, Alpha: 8-13Hz, Beta: 13-30Hz, Gamma: >30Hz
        delta_mask = (f >= 0.5) & (f < 4)
        theta_mask = (f >= 4) & (f < 8) 
        alpha_mask = (f >= 8) & (f < 13)
        beta_mask = (f >= 13) & (f < 30)
        gamma_mask = f >= 30
        
        psd_stats.append(
            {
                "total_power": np.sum(psd),
                "peak_frequency": f[np.argmax(psd)],
                "median_frequency": f[np.argmax(np.cumsum(psd) >= 0.5 * np.sum(psd))],
                "spectral_entropy": -np.sum((psd / np.sum(psd)) * np.log2(psd / np.sum(psd) + 1e-10)),
                "delta_power": np.sum(psd[delta_mask]),
                "theta_power": np.sum(psd[theta_mask]),
                "alpha_power": np.sum(psd[alpha_mask]),
                "beta_power": np.sum(psd[beta_mask]),
                "gamma_power": np.sum(psd[gamma_mask]),
            }
        )
    return psd_stats


def plot_ecg_sample(ecg, title, save_path):
    """
    Plot all leads of an ECG sample.
    
    Args:
        ecg: ECG recording with shape (leads, time_points)
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Define standard 12-lead labels
    lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    plt.figure(figsize=(20, 10))
    for i in range(min(len(lead_labels), ecg.shape[0])):
        plt.subplot(3, 4, i + 1)
        plt.plot(ecg[i])
        plt.title(f"Lead {lead_labels[i]}")
        plt.grid(True)
        
        # Add scale indicator (1mV, 200ms) assuming 500Hz sampling
        time_scale = 100  # 200ms at 500Hz = 100 samples
        amp_scale = 100   # Assuming 1mV = 100 units
        
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        # Add horizontal and vertical scale bars
        plt.plot([x_min + 50, x_min + 50 + time_scale], [y_min + 50, y_min + 50], 'k-', linewidth=2)
        plt.plot([x_min + 50, x_min + 50], [y_min + 50, y_min + 50 + amp_scale], 'k-', linewidth=2)
        
        # Add scale text
        plt.text(x_min + 50 + time_scale//2, y_min + 30, '200ms', ha='center')
        plt.text(x_min + 30, y_min + 50 + amp_scale//2, '1mV', va='center')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def calculate_lead_correlations(ecg):
    """
    Calculate correlation between ECG leads.
    
    Args:
        ecg: ECG recording with shape (leads, time_points)
        
    Returns:
        Correlation matrix
    """
    return np.corrcoef(ecg)


def detect_heartbeats(ecg, fs=500):
    """
    Simple R-peak detection for heart rate estimation.
    
    Args:
        ecg: ECG recording (lead II is used)
        fs: Sampling frequency in Hz
        
    Returns:
        heart_rate: Estimated heart rate in BPM
        rr_intervals: R-R intervals in seconds
    """
    from scipy.signal import find_peaks
    
    # Use lead II (index 1) for R-peak detection
    lead_ii = ecg[1] if ecg.shape[0] > 1 else ecg[0]
    
    # Normalize the signal
    normalized = (lead_ii - np.mean(lead_ii)) / np.std(lead_ii)
    
    # Find R-peaks (adjust parameters as needed)
    peaks, _ = find_peaks(normalized, height=1.5, distance=fs*0.5)  # At least 0.5s apart
    
    if len(peaks) < 2:
        return 0, []
    
    # Calculate RR intervals
    rr_intervals = np.diff(peaks) / fs  # in seconds
    
    # Calculate heart rate
    heart_rate = 60 / np.mean(rr_intervals)  # in BPM
    
    return heart_rate, rr_intervals


def analyze_ecg_data(data_dir, specific_file=None, visualization_dir=None):
    """
    Comprehensive ECG data analysis with enhanced visualizations and metrics.
    
    Args:
        data_dir: Directory containing ECG data
        specific_file: Optional specific file to analyze
        visualization_dir: Directory to save visualizations
    
    Returns:
        ecg_data: List of ECG recordings
        labels: List of label sets
    """
    print("\nStarting ECG data analysis...")
    
    # Set visualization directory or use current directory
    if visualization_dir is None:
        visualization_dir = os.path.join("src", "visualization", "eda_output")
    os.makedirs(visualization_dir, exist_ok=True)
    
    print(f"Visualizations will be saved to: {visualization_dir}")

    if specific_file:
        print(f"Analyzing specific file: {specific_file}")
        ecg_data = [np.load(specific_file)]
        header_file = specific_file.replace(".npy", "_header.json")
        if os.path.exists(header_file):
            with open(header_file, "r") as f:
                labels = [json.load(f).get("Dx", [])]
        else:
            labels = [[]]
    else:
        print("\nStarting data loading process...")
        ecg_data, labels = load_data(data_dir)

    print("Data loading complete.")

    # Basic dataset information
    print("\nDataset Information:")
    print(f"Total number of ECG samples: {len(ecg_data)}")
    
    if len(ecg_data) > 0:
        print(f"Shape of a single ECG sample: {ecg_data[0].shape}")
        print(f"Number of leads: {ecg_data[0].shape[0]}")
        print(f"Number of time points: {ecg_data[0].shape[1]}")
        
        # Calculate sampling rate (if available in metadata)
        fs = 500  # Default assumption
        print(f"Sampling rate (Hz): {fs}")
        print(f"Recording duration (seconds): {ecg_data[0].shape[1]/fs:.2f}")

    # Check for null/inf values and data quality
    null_count = sum(1 for ecg in ecg_data if np.isnan(ecg).any())
    inf_count = sum(1 for ecg in ecg_data if np.isinf(ecg).any())
    print(f"\nData Quality Checks:")
    print(f"- ECG samples with NaN values: {null_count}")
    print(f"- ECG samples with Inf values: {inf_count}")
    
    # Calculate value ranges for normalization insights
    if len(ecg_data) > 0:
        # Use a subset of data for statistics to avoid memory issues
        sample_size = min(100, len(ecg_data))
        sampled_ecgs = ecg_data[:sample_size]
        all_values = np.concatenate([ecg.flatten() for ecg in sampled_ecgs])
        
        print(f"- Value range: {np.min(all_values):.2f} to {np.max(all_values):.2f}")
        print(f"- 5th-95th percentile range: {np.percentile(all_values, 5):.2f} to {np.percentile(all_values, 95):.2f}")

    # Analyze ECG signal properties
    print("\nAnalyzing ECG signal properties...")
    if len(ecg_data) > 0:
        # Calculate aggregate properties for visualization
        durations = [ecg.shape[1] / 500 for ecg in ecg_data]  # Assuming 500 Hz sampling rate
        amplitudes = [np.max(np.abs(ecg)) for ecg in ecg_data]
        
        # Calculate detailed statistics for a subset of records
        sample_size = min(100, len(ecg_data))
        signal_stats = [calculate_signal_stats(ecg) for ecg in ecg_data[:sample_size]]
        frequency_stats = [analyze_frequency_components(ecg) for ecg in ecg_data[:sample_size]]
        
        # Estimate heart rates for a subset
        heart_rates = []
        for ecg in ecg_data[:sample_size]:
            hr, _ = detect_heartbeats(ecg)
            if hr > 0:  # Only include valid heart rates
                heart_rates.append(hr)

        # Create figure for signal properties
        plt.figure(figsize=(20, 15))
        
        # Plot durations
        plt.subplot(3, 3, 1)
        sns.histplot(durations, bins=50, kde=True)
        plt.title("ECG Recording Durations")
        plt.xlabel("Duration (seconds)")
        
        # Plot amplitudes
        plt.subplot(3, 3, 2)
        sns.histplot(amplitudes, bins=50, kde=True)
        plt.title("ECG Maximum Amplitudes")
        plt.xlabel("Max Amplitude")
        
        # Plot heart rates
        if heart_rates:
            plt.subplot(3, 3, 3)
            sns.histplot(heart_rates, bins=30, kde=True)
            plt.title("Estimated Heart Rates")
            plt.xlabel("Heart Rate (BPM)")
            plt.axvline(x=60, color='r', linestyle='--')  # Bradycardia threshold
            plt.axvline(x=100, color='r', linestyle='--')  # Tachycardia threshold
        
        # Plot lead statistics
        plt.subplot(3, 3, 4)
        sns.histplot([stat[0]["skewness"] for stat in signal_stats], bins=50, kde=True)
        plt.title("Lead I Skewness Distribution")
        plt.xlabel("Skewness")
        
        plt.subplot(3, 3, 5)
        sns.histplot([stat[0]["kurtosis"] for stat in signal_stats], bins=50, kde=True)
        plt.title("Lead I Kurtosis Distribution")
        plt.xlabel("Kurtosis")
        
        plt.subplot(3, 3, 6)
        sns.histplot([stat[0]["p2p"] for stat in signal_stats], bins=50, kde=True)
        plt.title("Lead I Peak-to-Peak Amplitude")
        plt.xlabel("Amplitude")
        
        # Plot frequency statistics
        plt.subplot(3, 3, 7)
        sns.histplot([stat[0]["total_power"] for stat in frequency_stats], bins=50, kde=True)
        plt.title("Lead I Total Power Distribution")
        plt.xlabel("Total Power")
        
        plt.subplot(3, 3, 8)
        sns.histplot([stat[0]["peak_frequency"] for stat in frequency_stats], bins=50, kde=True)
        plt.title("Lead I Peak Frequency Distribution")
        plt.xlabel("Peak Frequency (Hz)")
        
        plt.subplot(3, 3, 9)
        sns.histplot([stat[0]["spectral_entropy"] for stat in frequency_stats], bins=50, kde=True)
        plt.title("Lead I Spectral Entropy")
        plt.xlabel("Spectral Entropy")
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, "ecg_properties_distribution.png"))
        plt.close()
        
        # Create a correlation heatmap for leads on a sample ECG
        sample_ecg = ecg_data[0]
        corr_matrix = calculate_lead_correlations(sample_ecg)
        
        plt.figure(figsize=(12, 10))
        lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        used_labels = lead_labels[:min(len(lead_labels), sample_ecg.shape[0])]
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                   xticklabels=used_labels, yticklabels=used_labels)
        plt.title("Correlation Between ECG Leads")
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, "lead_correlation.png"))
        plt.close()
        
        # Plot temporal pattern of Lead II (rhythm analysis)
        plt.figure(figsize=(15, 6))
        lead_ii_idx = min(1, sample_ecg.shape[0]-1)  # Lead II is typically index 1
        time_seconds = np.arange(min(5000, sample_ecg.shape[1])) / fs
        plt.plot(time_seconds, sample_ecg[lead_ii_idx, :min(5000, sample_ecg.shape[1])])
        plt.grid(True)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title("Lead II Temporal Pattern (First 10 seconds)")
        
        # Mark R-peaks if detected
        hr, _ = detect_heartbeats(sample_ecg)
        if hr > 0:
            from scipy.signal import find_peaks
            lead_ii = sample_ecg[lead_ii_idx]
            normalized = (lead_ii - np.mean(lead_ii)) / np.std(lead_ii)
            peaks, _ = find_peaks(normalized[:min(5000, normalized.shape[0])], height=1.5, distance=fs*0.5)
            plt.plot(peaks/fs, normalized[peaks], 'ro')
            plt.text(0.02, 0.95, f"Est. Heart Rate: {hr:.1f} BPM", transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
            
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, "lead_ii_temporal.png"))
        plt.close()

    # Analyze label distribution
    print("\nAnalyzing label distribution...")
    if len(labels) > 0:
        all_labels = [label for sublist in labels for label in sublist]
        label_counts = Counter(all_labels)
        
        if label_counts:
            # Create sorted label distribution
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            
            plt.figure(figsize=(20, 10))
            sns.barplot(x=[item[0] for item in sorted_labels], 
                        y=[item[1] for item in sorted_labels])
            plt.title("Distribution of ECG Labels")
            plt.xticks(rotation=90)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, "label_distribution.png"))
            plt.close()
            
            # Also create a version with balanced axis for better comparison
            plt.figure(figsize=(20, 10))
            sns.barplot(x=[item[0] for item in sorted_labels], 
                        y=[item[1] for item in sorted_labels])
            plt.title("Distribution of ECG Labels (Balanced Axis)")
            plt.xticks(rotation=90)
            plt.ylabel("Count")
            plt.yscale('log')  # Log scale for better visualization of rare labels
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, "label_distribution_log_scale.png"))
            plt.close()

            # Pie chart for top 10 labels
            top_10_labels = dict(sorted_labels[:10])
            plt.figure(figsize=(12, 8))
            plt.pie(top_10_labels.values(), labels=top_10_labels.keys(), autopct="%1.1f%%")
            plt.title("Top 10 ECG Labels")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, "top_10_labels_pie_chart.png"))
            plt.close()
            
            # Create a horizontal bar chart for top 20 labels - often better for many categories
            top_20_labels = dict(sorted_labels[:20])
            plt.figure(figsize=(12, 10))
            sns.barplot(y=list(top_20_labels.keys()), x=list(top_20_labels.values()))
            plt.title("Top 20 ECG Labels")
            plt.xlabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, "top_20_labels_bar.png"))
            plt.close()

            # Calculate and print label statistics
            print("\nLabel Statistics:")
            print(f"Total number of unique labels: {len(label_counts)}")
            if label_counts:
                print(f"Most common label: {label_counts.most_common(1)[0][0]} (Count: {label_counts.most_common(1)[0][1]})")
                print(f"Least common label: {sorted_labels[-1][0]} (Count: {sorted_labels[-1][1]})")
                
                # Calculate imbalance ratio
                imbalance_ratio = label_counts.most_common(1)[0][1] / sorted_labels[-1][1]
                print(f"Imbalance ratio (Most common / Least common): {imbalance_ratio:.2f}")
                
                # Calculate class distribution statistics
                total_samples = sum(label_counts.values())
                majority_percentage = (label_counts.most_common(1)[0][1] / total_samples) * 100
                minority_percentage = (sorted_labels[-1][1] / total_samples) * 100
                print(f"Majority class percentage: {majority_percentage:.2f}%")
                print(f"Minority class percentage: {minority_percentage:.2f}%")

            # Analyze multi-label statistics
            label_count_per_sample = [len(label_set) for label_set in labels]
            
            plt.figure(figsize=(10, 5))
            sns.histplot(label_count_per_sample, kde=True, discrete=True)
            plt.title("Distribution of Number of Labels per Sample")
            plt.xlabel("Number of Labels")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(visualization_dir, "label_count_distribution.png"))
            plt.close()
            
            print(f"\nMulti-label Statistics:")
            print(f"Average labels per sample: {np.mean(label_count_per_sample):.2f}")
            print(f"Maximum labels per sample: {np.max(label_count_per_sample)}")
            print(f"Samples with multiple labels: {sum(count > 1 for count in label_count_per_sample)} ({sum(count > 1 for count in label_count_per_sample)/len(label_count_per_sample)*100:.2f}%)")

            # Calculate label co-occurrence
            co_occurrence = Counter()
            for label_set in labels:
                if len(label_set) > 1:  # Only consider multi-label instances
                    co_occurrence.update(frozenset(pair) for pair in itertools.combinations(label_set, 2))
            
            if co_occurrence:
                # Plot top co-occurrences
                top_co_occurrences = dict(sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:20])
                
                plt.figure(figsize=(15, 10))
                sns.barplot(
                    x=list(top_co_occurrences.values()),
                    y=[", ".join(pair) for pair in top_co_occurrences.keys()],
                )
                plt.title("Top 20 Label Co-occurrences")
                plt.xlabel("Co-occurrence Count")
                plt.tight_layout()
                plt.savefig(os.path.join(visualization_dir, "label_co_occurrences.png"))
                plt.close()
                
                # Create a co-occurrence matrix heatmap for top 15 labels
                top_15_labels = [item[0] for item in sorted_labels[:15]]
                co_matrix = np.zeros((len(top_15_labels), len(top_15_labels)))
                
                # Fill the co-occurrence matrix
                for i, label1 in enumerate(top_15_labels):
                    for j, label2 in enumerate(top_15_labels):
                        if i == j:  # Diagonal is occurrence count
                            co_matrix[i, j] = label_counts[label1]
                        else:
                            pair = frozenset([label1, label2])
                            co_matrix[i, j] = co_occurrence.get(pair, 0)
                
                # Plot the co-occurrence matrix
                plt.figure(figsize=(14, 12))
                sns.heatmap(co_matrix, annot=True, fmt='.0f', cmap='viridis',
                           xticklabels=top_15_labels, yticklabels=top_15_labels)
                plt.title("Co-occurrence Matrix for Top 15 Labels")
                plt.tight_layout()
                plt.savefig(os.path.join(visualization_dir, "label_co_occurrence_matrix.png"))
                plt.close()

    # Plot a sample ECG
    print("\nPlotting sample ECGs...")
    if len(ecg_data) > 0:
        # Plot a random sample
        sample_index = 0 if specific_file else np.random.randint(len(ecg_data))
        sample_ecg = ecg_data[sample_index]
        sample_labels = labels[sample_index] if sample_index < len(labels) else []
        
        plot_ecg_sample(
            sample_ecg, 
            f"Sample ECG (index: {sample_index}, labels: {', '.join(sample_labels)})",
            os.path.join(visualization_dir, "sample_ecg.png")
        )
        
        # If we have multiple samples, plot a few more with different labels
        if len(ecg_data) > 10 and not specific_file:
            # Find samples with different label counts
            single_label_idx = next((i for i, l in enumerate(labels) if len(l) == 1), None)
            multi_label_idx = next((i for i, l in enumerate(labels) if len(l) > 1), None)
            
            if single_label_idx is not None:
                plot_ecg_sample(
                    ecg_data[single_label_idx],
                    f"Single Label ECG (label: {labels[single_label_idx][0]})",
                    os.path.join(visualization_dir, "single_label_ecg.png")
                )
            
            if multi_label_idx is not None:
                plot_ecg_sample(
                    ecg_data[multi_label_idx],
                    f"Multi-Label ECG (labels: {', '.join(labels[multi_label_idx])})",
                    os.path.join(visualization_dir, "multi_label_ecg.png")
                )
    
    # Create a comprehensive visualization of a sample ECG with annotations
    if len(ecg_data) > 0:
        sample_ecg = ecg_data[0]
        lead_ii_idx = min(1, sample_ecg.shape[0]-1)  # Use Lead II for rhythm
        
        plt.figure(figsize=(20, 12))
        
        # Main plot - all 12 leads in a grid
        gs = plt.GridSpec(4, 3, height_ratios=[1, 1, 1, 2])
        
        # Plot each lead
        lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        used_labels = lead_labels[:min(len(lead_labels), sample_ecg.shape[0])]
        
        for i, label in enumerate(used_labels):
            if i < 9:  # First 9 leads in the grid (3x3)
                row, col = i // 3, i % 3
                ax = plt.subplot(gs[row, col])
                ax.plot(sample_ecg[i, :2500])  # Show 5 seconds at 500Hz
                ax.set_title(f"Lead {label}")
                ax.grid(True, alpha=0.3)
                # Remove axis labels for cleaner look
                ax.set_xticklabels([])
                if col != 0:
                    ax.set_yticklabels([])
            
        # Bottom row - extended Lead II for rhythm analysis
        ax_rhythm = plt.subplot(gs[3, :])
        ax_rhythm.plot(np.arange(5000)/fs, sample_ecg[lead_ii_idx, :5000])
        ax_rhythm.set_title("Lead II Rhythm Strip (10 seconds)")
        ax_rhythm.set_xlabel("Time (seconds)")
        ax_rhythm.grid(True, alpha=0.3)
        
        # Add heart rate estimation if possible
        hr, _ = detect_heartbeats(sample_ecg)
        if hr > 0:
            from scipy.signal import find_peaks
            lead_ii = sample_ecg[lead_ii_idx]
            normalized = (lead_ii - np.mean(lead_ii)) / np.std(lead_ii)
            peaks, _ = find_peaks(normalized[:5000], height=1.5, distance=fs*0.5)
            ax_rhythm.plot(peaks/fs, normalized[peaks], 'ro')
            ax_rhythm.text(0.02, 0.90, f"Est. Heart Rate: {hr:.1f} BPM", transform=ax_rhythm.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, "sample_ecg_visualization.png"))
        plt.close()
        
    print(f"\nECG data analysis complete. Visualizations saved to {visualization_dir}")
    return ecg_data, labels  # Return loaded data for further analysis