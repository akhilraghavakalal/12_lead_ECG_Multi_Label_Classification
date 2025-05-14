import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report,
)
import json
import os
import logging
from typing import Dict, List, Tuple
import torch


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> Dict:
    """Calculate comprehensive metrics for multi-label classification."""
    metrics = {}

    # Calculate metrics for each class
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        metrics[f"class_{i}"] = {
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "auc": roc_curve(y_true[:, i], y_scores[:, i])[2]
            if len(np.unique(y_true[:, i])) > 1
            else 0,
        }

    # Calculate macro and weighted averages
    metrics["macro_avg"] = {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    metrics["weighted_avg"] = {
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    return metrics


def evaluate_model(
    model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and return predictions, true labels, and scores."""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for wide_features, ecg_data, labels in val_loader:
            wide_features = wide_features.float().to(device)
            ecg_data = ecg_data.float().to(device)
            labels = labels.float().to(device)

            outputs = model(ecg_data, wide_features)
            predictions = (outputs > 0.5).float()

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_scores.append(outputs.cpu().numpy())

    return (np.vstack(all_preds), np.vstack(all_labels), np.vstack(all_scores))


def plot_confusion_matrices(results: Dict, save_dir: str, class_names: List[str]):
    """Plot confusion matrices for each model."""
    for model_name, model_results in results.items():
        if "confusion_matrix" not in model_results:
            continue

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            model_results["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{model_name.lower()}_confusion_matrix.png")
        )
        plt.close()


def plot_pr_curves(results: Dict, save_dir: str, class_names: List[str]):
    """Plot precision-recall curves for each model."""
    for model_name, model_results in results.items():
        plt.figure(figsize=(12, 8))

        for i, class_name in enumerate(class_names):
            if f"class_{i}_pr" in model_results:
                precision = model_results[f"class_{i}_pr"]["precision"]
                recall = model_results[f"class_{i}_pr"]["recall"]
                ap = model_results[f"class_{i}_pr"]["ap"]
                plt.plot(recall, precision, label=f"{class_name} (AP={ap:.2f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name} Precision-Recall Curves")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_name.lower()}_pr_curves.png"))
        plt.close()


def plot_roc_curves(results: Dict, save_dir: str, class_names: List[str]):
    """Plot ROC curves for each model."""
    for model_name, model_results in results.items():
        plt.figure(figsize=(12, 8))

        for i, class_name in enumerate(class_names):
            if f"class_{i}_roc" in model_results:
                fpr = model_results[f"class_{i}_roc"]["fpr"]
                tpr = model_results[f"class_{i}_roc"]["tpr"]
                roc_auc = model_results[f"class_{i}_roc"]["auc"]
                plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curves")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_name.lower()}_roc_curves.png"))
        plt.close()


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_dir: str,
    model_name: str,
):
    """Generate and save detailed classification report."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    # Save as JSON
    with open(
        os.path.join(save_dir, f"{model_name.lower()}_classification_report.json"), "w"
    ) as f:
        json.dump(report, f, indent=4)

    # Save as formatted text
    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    with open(
        os.path.join(save_dir, f"{model_name.lower()}_classification_report.txt"), "w"
    ) as f:
        f.write(f"Classification Report for {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(report_txt)


def generate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_dir: str,
) -> Dict:
    """Generate comprehensive evaluation metrics and visualizations."""
    results = {}

    # Calculate basic metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    results.update(metrics)

    # Calculate confusion matrix
    results["confusion_matrix"] = confusion_matrix(y_true.ravel(), y_pred.ravel())

    # Calculate ROC and PR curves for each class
    for i, class_name in enumerate(class_names):
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        results[f"class_{i}_roc"] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

        # PR curve
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        results[f"class_{i}_pr"] = {"precision": precision, "recall": recall, "ap": ap}

    # Generate and save visualizations
    plot_confusion_matrices({model_name: results}, save_dir, class_names)
    plot_pr_curves({model_name: results}, save_dir, class_names)
    plot_roc_curves({model_name: results}, save_dir, class_names)

    # Save classification report
    save_classification_report(y_true, y_pred, class_names, save_dir, model_name)

    return results
