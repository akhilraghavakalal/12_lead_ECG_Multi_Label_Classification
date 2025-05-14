import torch
import torch.nn as nn
import numpy as np
import logging
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from .evaluation import evaluate_model, generate_comprehensive_metrics


def setup_logging(model_dir, log_level=logging.INFO):
    """Set up logging configuration"""
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(model_dir, "training.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )

    logging.info(f"Logging configured. Log file: {log_file}")
    logging.info(f"Log level set to: {logging.getLevelName(log_level)}")


def save_training_plots(train_losses, val_losses, val_aucs, save_dir, model_type):
    """Save training metrics plots"""
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_type} Training and Validation Loss")
    plt.legend()

    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{model_type} Validation AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_type}_training_metrics.png"))
    plt.close()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []

    progress_bar = tqdm(loader, desc="Training")
    for wide_features, ecg, labels in progress_bar:
        # Move data to device
        wide_features = wide_features.to(device, non_blocking=True)
        ecg = ecg.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        outputs = model(ecg, wide_features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Store predictions and labels for metrics
        all_outputs.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate training metrics
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    predictions = (all_outputs > 0.5).astype(float)

    return total_loss / len(loader), all_outputs, all_labels, predictions


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for wide_features, ecg, labels in tqdm(loader, desc="Validating"):
            wide_features = wide_features.to(device)
            ecg = ecg.to(device)
            labels = labels.to(device)

            outputs = model(ecg, wide_features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    predictions = (all_outputs > 0.5).astype(float)

    # Calculate AUC for each class
    aucs = []
    for i in range(all_outputs.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            aucs.append(auc)

    return (
        total_loss / len(loader),
        np.mean(aucs),
        aucs,
        all_outputs,
        all_labels,
        predictions,
    )


def train_cnn_gru_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    save_dir,
    class_names,
    num_epochs=50,
    patience=5,
):
    """Train the CNN-GRU model with comprehensive metrics tracking"""
    best_val_auc = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_aucs = []
    class_wise_aucs = []

    # Create directories for saving results
    os.makedirs(os.path.join(save_dir, "cnn_gru_metrics"), exist_ok=True)
    metrics_dir = os.path.join(save_dir, "cnn_gru_metrics")
    
    print("\n\n CNN-GRU model training\n")
    logging.info("\nInitializing CNN-GRU model training...\n")

    for epoch in range(num_epochs):
        logging.info(f"\n\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_outputs, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        logging.info(f"Training Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_auc, class_aucs, val_outputs, val_labels, val_preds = validate(
            model, val_loader, criterion, device
        )
        logging.info(f"Validation Loss: {val_loss:.4f}")
        logging.info(f"Validation AUC: {val_auc:.4f}")

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        class_wise_aucs.append(class_aucs)

        # Generate comprehensive metrics for current epoch
        epoch_metrics = generate_comprehensive_metrics(
            val_labels,
            val_preds,
            val_outputs,
            class_names,
            f"CNN_GRU_Epoch_{epoch + 1}",
            metrics_dir,
        )

        # Log detailed metrics
        logging.info("\nDetailed Metrics:")
        logging.info(
            f"Macro Avg - Precision: {epoch_metrics['macro_avg']['precision']:.4f}"
        )
        logging.info(f"Macro Avg - Recall: {epoch_metrics['macro_avg']['recall']:.4f}")
        logging.info(f"Macro Avg - F1: {epoch_metrics['macro_avg']['f1']:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint if improved
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            # Save model checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auc": val_auc,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_aucs": val_aucs,
                "class_wise_aucs": class_wise_aucs,
                "metrics": epoch_metrics,
            }

            torch.save(checkpoint, os.path.join(save_dir, "best_cnn_gru_model.pth"))

            # Generate and save best model metrics
            best_metrics = generate_comprehensive_metrics(
                val_labels,
                val_preds,
                val_outputs,
                class_names,
                "CNN_GRU_Best_Model",
                save_dir,
            )

            logging.info("Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs!")
                break

        # Save training plots
        save_training_plots(train_losses, val_losses, val_aucs, save_dir, "cnn_gru")

    # Final evaluation using the best model
    model.load_state_dict(
        torch.load(os.path.join(save_dir, "best_cnn_gru_model.pth"))["model_state_dict"]
    )
    (
        final_val_loss,
        final_val_auc,
        final_class_aucs,
        final_outputs,
        final_labels,
        final_preds,
    ) = validate(model, val_loader, criterion, device)

    final_metrics = generate_comprehensive_metrics(
        final_labels, final_preds, final_outputs, class_names, "CNN_GRU_Final", save_dir
    )

    return (
        best_val_auc,
        train_losses,
        val_losses,
        val_aucs,
        class_wise_aucs,
        final_metrics,
    )
