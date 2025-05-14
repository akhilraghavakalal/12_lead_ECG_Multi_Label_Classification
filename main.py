import matplotlib

matplotlib.use("Agg")

import argparse
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import logging
import json
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from collections import defaultdict

from src.preprocessing.preprocess import process_dataset
from src.preprocessing.eda import analyze_ecg_data
from src.visualization.visualize import visualize_sample
from src.preprocessing.split_data import reorganize_data
from src.feature_extraction.wide_features import process_wide_features
from src.models.ecg_dataset import prepare_data
from src.models.cnn_gru_model import CNNGRUModel
from src.models.optimizer import (
    create_cnn_gru_optimizer,
    create_cnn_gru_scheduler,
    create_noam_optimizer,
)
from src.models.train_cnn_gru_model import train_cnn_gru_model, setup_logging
from src.models.wide_deep_transformer_model import create_model
from src.models.train_wide_deep_transformer_model import train_wide_deep_transformer

# Enable GPU optimizations
if torch.cuda.is_available():
    cudnn.benchmark = True
    cudnn.deterministic = False

# Define ECG class names based on the paper
CLASS_NAMES = [
    "Normal sinus rhythm",
    "Atrial fibrillation",
    "First-degree AV block",
    "Left bundle branch block",
    "Right bundle branch block",
    "Premature atrial contraction",
    "Premature ventricular contraction",
    "ST-segment depression",
    "ST-segment elevation",
    "T-wave abnormal",
    "T-wave inversion",
    "Left axis deviation",
    "Right axis deviation",
    "Low QRS voltages",
    "Sinus bradycardia",
    "Sinus tachycardia",
    "Left ventricular hypertrophy",
    "Right ventricular hypertrophy",
    "Myocardial infarction",
    "Left atrial abnormality",
    "Right atrial abnormality",
    "Nonspecific intraventricular conduction delay",
    "Atrial flutter",
    "Wandering atrial pacemaker",
    "Supraventricular tachycardia",
    "Junctional rhythm",
    "Ventricular trigeminy",
]


def check_processed_data(final_data_dir):
    """Check if processed data already exists."""
    return (
        os.path.exists(os.path.join(final_data_dir, "train_wide_features.npy"))
        and os.path.exists(os.path.join(final_data_dir, "train_labels.npy"))
        and os.path.exists(os.path.join(final_data_dir, "val_wide_features.npy"))
        and os.path.exists(os.path.join(final_data_dir, "val_labels.npy"))
        and os.path.exists(os.path.join(final_data_dir, "feature_selector.joblib"))
    )


def process_and_save_data(args):
    """Process and save the data"""
    if not os.path.exists(args.output_dir) or args.force_preprocess:
        print("Processing dataset...")
        process_dataset(args.input_dir, args.output_dir)
    else:
        print("Preprocessed data already exists. Skipping preprocessing step.")

    if args.visualize:
        print("\nVisualizing data before reorganizing it.")
        visualize_sample(args.output_dir)

    print("\nAnalyzing ECG data...")
    analyze_ecg_data(args.output_dir)

    if not os.path.exists(args.final_data_dir) or args.force_preprocess:
        print("\nReorganizing data...")
        # reorganize_data(args.output_dir, args.final_data_dir, "st_petersburg_incart")
        reorganize_data(args.output_dir, args.final_data_dir, "ptb-xl")
    else:
        print("\nReorganized data already exists. Skipping reorganization step.")

    print("\nAnalyzing reorganized ECG data...")
    analyze_ecg_data(args.final_data_dir)
    if args.visualize:
        visualize_sample(args.final_data_dir)

    print("\nExtracting wide features...")
    (train_features, train_labels), (val_features, val_labels), feature_selector = (
        process_wide_features(args.final_data_dir)
    )

    # Save processed data
    np.save(
        os.path.join(args.final_data_dir, "train_wide_features.npy"), train_features
    )
    np.save(os.path.join(args.final_data_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(args.final_data_dir, "val_wide_features.npy"), val_features)
    np.save(os.path.join(args.final_data_dir, "val_labels.npy"), val_labels)
    joblib.dump(
        feature_selector, os.path.join(args.final_data_dir, "feature_selector.joblib")
    )

    return train_features, train_labels, val_features, val_labels, feature_selector


def load_processed_data(final_data_dir):
    """Load processed data from files"""
    train_features = np.load(os.path.join(final_data_dir, "train_wide_features.npy"))
    train_labels = np.load(os.path.join(final_data_dir, "train_labels.npy"))
    val_features = np.load(os.path.join(final_data_dir, "val_wide_features.npy"))
    val_labels = np.load(os.path.join(final_data_dir, "val_labels.npy"))
    feature_selector = joblib.load(
        os.path.join(final_data_dir, "feature_selector.joblib")
    )
    return train_features, train_labels, val_features, val_labels, feature_selector


def format_value(value):
    """Helper function to format values appropriately based on their type."""
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 1:
            return f"{float(value[0]):.4f}"
        return f"{np.mean(value):.4f}"  # Use mean for list values
    elif isinstance(value, (float, np.float32, np.float64)):
        return f"{float(value):.4f}"
    else:
        return str(value)


def compare_models(model_dir):
    """Compare and visualize the performance of both models with enhanced metrics."""
    cnn_path = os.path.join(model_dir, "cnn_gru_results.json")
    transformer_path = os.path.join(model_dir, "transformer_results.json")
    plt.rcParams.update({'font.size': 14}) # added this to handle the font size of the image labels
    results = {}
    if os.path.exists(cnn_path):
        with open(cnn_path, "r") as f:
            results["CNN-GRU"] = json.load(f)
    if os.path.exists(transformer_path):
        with open(transformer_path, "r") as f:
            results["Transformer"] = json.load(f)

    if not results:
        logging.info("No results found to compare")
        return

    # Create comparison plots directory
    comparison_dir = os.path.join(model_dir, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Plot training metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 16
    legend_fontsize = 16
    
    # Plot training loss
    ax = axes[0, 0]
    for model_name, model_results in results.items():
        ax.plot(model_results["train_losses"], label=f"{model_name}", linewidth=2)
    ax.set_title("Training Loss Comparison", fontsize=title_fontsize,  fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=label_fontsize)
    ax.set_ylabel("Loss", fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Plot validation loss
    ax = axes[0, 1]
    for model_name, model_results in results.items():
        ax.plot(model_results["val_losses"], label=f"{model_name}", linewidth=2)
    ax.set_title("Validation Loss Comparison", fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=label_fontsize)
    ax.set_ylabel("Loss", fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Plot validation AUC
    ax = axes[1, 0]
    for model_name, model_results in results.items():
        ax.plot(model_results["val_aucs"], label=f"{model_name}", linewidth=2)
    ax.set_title("Validation AUC Comparison", fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=label_fontsize)
    ax.set_ylabel("AUC", fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    fig.delaxes(axes[1,1])
    # Create comparison table
    # ax = axes[1, 1]
    # ax.axis("off")

    # table_data = []
    # table_columns = ["Model", "Best AUC", "Final Loss", "Macro F1", "Weighted F1"]

    # for model_name, model_results in results.items():
    #     metrics = model_results.get("final_metrics", {})
    #     macro_f1 = metrics.get("macro_avg", {}).get("f1", 0)
    #     weighted_f1 = metrics.get("weighted_avg", {}).get("f1", 0)

    #     table_data.append(
    #         [
    #             model_name,
    #             format_value(model_results["best_val_auc"]),
    #             format_value(model_results["val_losses"][-1]),
    #             format_value(macro_f1),
    #             format_value(weighted_f1),
    #         ]
    #     )

    # table = ax.table(
    #     cellText=table_data, colLabels=table_columns, loc="center", cellLoc="center"
    # )
    # table.auto_set_font_size(False)
    # table.set_fontsize(9)
    # table.scale(1.2, 1.5)
    plt.rcParams['font.size'] = 16
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "model_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed comparison report
    report = "Model Comparison Report\n"
    report += "=====================\n\n"

    for model_name, model_results in results.items():
        report += f"\n{model_name} Performance:\n"
        report += "-" * (len(model_name) + 12) + "\n"
        report += (
            f"Best Validation AUC: {format_value(model_results['best_val_auc'])}\n"
        )
        report += (
            f"Final Validation Loss: {format_value(model_results['val_losses'][-1])}\n"
        )

        if "final_metrics" in model_results:
            metrics = model_results["final_metrics"]
            report += "\nMacro Averages:\n"
            for metric, value in metrics.get("macro_avg", {}).items():
                report += f"  {metric}: {format_value(value)}\n"

            report += "\nWeighted Averages:\n"
            for metric, value in metrics.get("weighted_avg", {}).items():
                report += f"  {metric}: {format_value(value)}\n"

            report += "\nPer-Class Performance:\n"
            for i, class_name in enumerate(CLASS_NAMES):
                if f"class_{i}" in metrics:
                    class_metrics = metrics[f"class_{i}"]
                    report += f"\n{class_name}:\n"
                    for metric, value in class_metrics.items():
                        report += f"  {metric}: {format_value(value)}\n"

        report += "\n" + "=" * 50 + "\n"

    with open(os.path.join(comparison_dir, "comparison_report.txt"), "w") as f:
        f.write(report)

    logging.info("\nModel Comparison Results:")
    logging.info(report)


def convert_arrays_to_lists(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays_to_lists(item) for item in obj]
    return obj


def train_model(args, model_type, device_id):
    """Train a single model with comprehensive metrics tracking."""
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device_id)

    # Load and prepare data
    if not check_processed_data(args.final_data_dir) or args.force_preprocess:
        train_features, train_labels, val_features, val_labels, feature_selector = (
            process_and_save_data(args)
        )
    else:
        train_features, train_labels, val_features, val_labels, feature_selector = (
            load_processed_data(args.final_data_dir)
        )

    train_ecg_paths = [
        os.path.join(args.final_data_dir, "train", f)
        for f in os.listdir(os.path.join(args.final_data_dir, "train"))
        if f.endswith(".npy")
    ]
    val_ecg_paths = [
        os.path.join(args.final_data_dir, "validation", f)
        for f in os.listdir(os.path.join(args.final_data_dir, "validation"))
        if f.endswith(".npy")
    ]

    train_dataset, train_loader = prepare_data(
        train_features, train_labels, train_ecg_paths, batch_size=args.batch_size
    )
    val_dataset, val_loader = prepare_data(
        val_features, val_labels, val_ecg_paths, batch_size=args.batch_size
    )

    if model_type == "cnn_gru":
        model = CNNGRUModel().to(device)
        optimizer = create_cnn_gru_optimizer(model, args.lr)
        scheduler = create_cnn_gru_scheduler(optimizer, args.patience)
        criterion = nn.BCELoss()

        results = train_cnn_gru_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            save_dir=args.model_dir,
            num_epochs=args.num_epochs,
            patience=args.patience,
            class_names=CLASS_NAMES,
        )
    else:  # transformer
        sample_ecg = np.load(train_ecg_paths[0])
        sequence_length = sample_ecg.shape[1]

        model = create_model(
            wide_dim=train_features.shape[1],
            n_classes=train_labels.shape[1],
            sequence_length=sequence_length,
        ).to(device)

        results = train_wide_deep_transformer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_size=args.model_size,
            warmup_steps=args.warmup_steps,
            device=device,
            save_dir=args.model_dir,
            num_epochs=args.num_epochs,
            patience=args.patience,
            class_names=CLASS_NAMES,
        )

    # Organize and save results
    best_val_auc, train_losses, val_losses, val_aucs, class_wise_aucs, final_metrics = (
        results
    )

    return_results = {
        "best_val_auc": float(best_val_auc),  # Convert to float
        "train_losses": convert_arrays_to_lists(train_losses),
        "val_losses": convert_arrays_to_lists(val_losses),
        "val_aucs": convert_arrays_to_lists(val_aucs),
        "class_wise_aucs": convert_arrays_to_lists(class_wise_aucs),
        "final_metrics": convert_arrays_to_lists(final_metrics),
    }

    # Save results to JSON
    with open(os.path.join(args.model_dir, f"{model_type}_results.json"), "w") as f:
        json.dump(return_results, f, indent=4)

    logging.info(f"\n{model_type.upper()} Training completed!")
    logging.info(f"Best validation AUC: {float(best_val_auc):.4f}")

    # Log detailed final metrics
    logging.info("\nFinal Model Performance:")
    logging.info(f"Macro Averages:")
    for metric, value in final_metrics["macro_avg"].items():
        # Handle different types of values safely
        try:
            if isinstance(value, (np.ndarray, np.generic)):
                if value.size == 1:
                    value = float(value.item())
                else:
                    value = float(value.mean())  # or another appropriate reduction
            else:
                value = float(value)
            logging.info(f"  {metric}: {value:.4f}")
        except (ValueError, TypeError) as e:
            logging.warning(f"Could not convert {metric} value to float: {e}")

    logging.info(f"\nWeighted Averages:")
    for metric, value in final_metrics["weighted_avg"].items():
        try:
            if isinstance(value, (np.ndarray, np.generic)):
                if value.size == 1:
                    value = float(value.item())
                else:
                    value = float(value.mean())  # or another appropriate reduction
            else:
                value = float(value)
            logging.info(f"  {metric}: {value:.4f}")
        except (ValueError, TypeError) as e:
            logging.warning(f"Could not convert {metric} value to float: {e}")

    logging.info("\nPer-Class Performance Summary:")
    for i, class_name in enumerate(CLASS_NAMES):
        if f"class_{i}" in final_metrics:
            logging.info(f"\n{class_name}:")
            for metric, value in final_metrics[f"class_{i}"].items():
                try:
                    if isinstance(value, (np.ndarray, np.generic)):
                        if value.size == 1:
                            value = float(value.item())
                        else:
                            value = float(
                                value.mean()
                            )  # or another appropriate reduction
                    else:
                        value = float(value)
                    logging.info(f"  {metric}: {value:.4f}")
                except (ValueError, TypeError) as e:
                    logging.warning(f"Could not convert {metric} value to float: {e}")

    return return_results


def main():
    parser = argparse.ArgumentParser(
        description="ECG Classification with Enhanced Metrics"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to input directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--final_data_dir", type=str, required=True, help="Path to final data directory"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model_checkpoints",
        help="Directory to save models and metrics",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize samples after preprocessing",
    )
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Force preprocessing even if processed data exists",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn_gru", "transformer", "both"],
        default="both",
        help="Which model to train",
    )
    parser.add_argument(
        "--model_size", type=int, default=256, help="Transformer model size"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=4000, help="Warmup steps for Noam optimizer"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.model_dir)
    logging.info("Starting ECG Classification Training with Enhanced Metrics...")

    # Log GPU information
    if torch.cuda.is_available():
        logging.info(f"Using device: {args.device}")
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
        logging.info(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")
    else:
        logging.info("Using CPU for training")

    # Create necessary directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, "plots"), exist_ok=True)

    # Train models based on selection
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    try:
        if args.model == "both" and num_gpus >= 2:
            # Parallel training on multiple GPUs
            processes = []
            mp.spawn(train_model, args=(args, "transformer", 1), nprocs=1, join=True)
            mp.spawn(train_model, args=(args, "cnn_gru", 0), nprocs=1, join=False)
        else:
            # Sequential training
            if args.model in ["transformer", "both"]:
                logging.info("\n***************************************************\n\n Transformer model training...\n***************************************************\n")
                train_model(args, "transformer", 0)

            if args.model in ["cnn_gru", "both"]:
                logging.info("\n***************************************************\n\n Starting CNN-GRU model training...\n***************************************************\n")
                train_model(args, "cnn_gru", 0)

        # Compare models if both were trained
        if args.model == "both":
            logging.info("\n***************************************************\n\n Generating comprehensive model comparison...\n***************************************************\n")
            compare_models(args.model_dir)

        logging.info("\nTraining completed successfully!")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        plt.close("all")
        torch.cuda.empty_cache()
        logging.info("Cleanup completed")


if __name__ == "__main__":
    try:
        # Use spawn method for Windows compatibility
        if torch.cuda.is_available():
            mp.set_start_method("spawn", force=True)
        main()
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}", exc_info=True)
    finally:
        plt.close("all")
        torch.cuda.empty_cache()
