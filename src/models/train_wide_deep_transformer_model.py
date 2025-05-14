import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os
from .optimizer import create_noam_optimizer
from .evaluation import generate_comprehensive_metrics


def train_wide_deep_transformer(
    model,
    train_loader,
    val_loader,
    model_size=256,
    warmup_steps=4000,
    num_epochs=50,
    patience=5,
    device="cuda",
    save_dir="model_checkpoints",
    class_names=None,
):
    """Train the Wide and Deep Transformer model with comprehensive metrics tracking."""
    logging.info("Initializing Wide Deep Transformer training...")

    # Create optimizer using the Noam implementation
    optimizer = create_noam_optimizer(
        model=model, model_size=model_size, factor=1.0, warmup_steps=warmup_steps
    )
    criterion = nn.BCELoss()

    # Initialize tracking variables
    best_auc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_aucs = []
    class_wise_aucs = []

    # Create save directories
    os.makedirs(save_dir, exist_ok=True)
    metrics_dir = os.path.join(save_dir, "transformer_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_losses = []
            all_train_outputs = []
            all_train_labels = []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, (wide_features, ecg_data, labels) in enumerate(progress_bar):
                # Debug shapes at first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    logging.info(f"Initial batch shapes:")
                    logging.info(f"Wide features: {wide_features.shape}")
                    logging.info(f"ECG data: {ecg_data.shape}")
                    logging.info(f"Labels: {labels.shape}")

                # Move data to device
                wide_features = wide_features.float().to(device)
                ecg_data = ecg_data.float().to(device)
                labels = labels.float().to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(ecg_data, wide_features)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Store predictions
                all_train_outputs.append(outputs.detach().cpu().numpy())
                all_train_labels.append(labels.cpu().numpy())

                # Update progress
                current_loss = loss.item()
                epoch_losses.append(current_loss)
                progress_bar.set_postfix({"train_loss": f"{current_loss:.4f}"})

            # Calculate training metrics
            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)

            all_train_outputs = np.vstack(all_train_outputs)
            all_train_labels = np.vstack(all_train_labels)
            train_predictions = (all_train_outputs > 0.5).astype(float)

            # Validation phase
            model.eval()
            val_epoch_losses = []
            all_val_outputs = []
            all_val_labels = []

            with torch.no_grad():
                for wide_features, ecg_data, labels in val_loader:
                    wide_features = wide_features.float().to(device)
                    ecg_data = ecg_data.float().to(device)
                    labels = labels.float().to(device)

                    outputs = model(ecg_data, wide_features)
                    loss = criterion(outputs, labels)

                    val_epoch_losses.append(loss.item())
                    all_val_outputs.append(outputs.cpu().numpy())
                    all_val_labels.append(labels.cpu().numpy())

            # Calculate validation metrics
            avg_val_loss = np.mean(val_epoch_losses)
            val_losses.append(avg_val_loss)

            all_val_outputs = np.vstack(all_val_outputs)
            all_val_labels = np.vstack(all_val_labels)
            val_predictions = (all_val_outputs > 0.5).astype(float)

            # Calculate AUC for each class
            current_aucs = []
            for i in range(all_val_outputs.shape[1]):
                if len(np.unique(all_val_labels[:, i])) > 1:
                    class_auc = roc_auc_score(
                        all_val_labels[:, i], all_val_outputs[:, i]
                    )
                    current_aucs.append(class_auc)

            mean_auc = np.mean(current_aucs)
            val_aucs.append(mean_auc)
            class_wise_aucs.append(current_aucs)

            # Generate comprehensive metrics for current epoch
            epoch_metrics = generate_comprehensive_metrics(
                all_val_labels,
                val_predictions,
                all_val_outputs,
                class_names,
                f"Transformer_Epoch_{epoch + 1}",
                metrics_dir,
            )

            # Log progress and metrics
            logging.info(f"\n\nEpoch {epoch + 1}/{num_epochs}:")
            logging.info(f"Train Loss: {avg_train_loss:.4f}")
            logging.info(f"Val Loss: {avg_val_loss:.4f}")
            logging.info(f"Val AUC: {mean_auc:.4f}")
            logging.info("\nDetailed Metrics:")
            logging.info(
                f"Macro Avg - Precision: {epoch_metrics['macro_avg']['precision']:.4f}"
            )
            logging.info(
                f"Macro Avg - Recall: {epoch_metrics['macro_avg']['recall']:.4f}"
            )
            logging.info(f"Macro Avg - F1: {epoch_metrics['macro_avg']['f1']:.4f}")

            # Save best model and handle early stopping
            if mean_auc > best_auc:
                best_auc = mean_auc

                # Save model checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_aucs": val_aucs,
                    "class_wise_aucs": class_wise_aucs,
                    "metrics": epoch_metrics,
                }

                model_save_path = os.path.join(
                    save_dir, "best_wide_deep_transformer.pth"
                )
                torch.save(checkpoint, model_save_path)

                # Generate and save best model metrics
                best_metrics = generate_comprehensive_metrics(
                    all_val_labels,
                    val_predictions,
                    all_val_outputs,
                    class_names,
                    "Transformer_Best_Model",
                    save_dir,
                )

                logging.info(f"New best model saved with AUC: {best_auc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

    # Final evaluation using the best model
    model.load_state_dict(
        torch.load(os.path.join(save_dir, "best_wide_deep_transformer.pth"))[
            "model_state_dict"
        ]
    )
    model.eval()

    final_val_outputs = []
    final_val_labels = []
    final_val_losses = []

    with torch.no_grad():
        for wide_features, ecg_data, labels in val_loader:
            wide_features = wide_features.float().to(device)
            ecg_data = ecg_data.float().to(device)
            labels = labels.float().to(device)

            outputs = model(ecg_data, wide_features)
            loss = criterion(outputs, labels)

            final_val_losses.append(loss.item())
            final_val_outputs.append(outputs.cpu().numpy())
            final_val_labels.append(labels.cpu().numpy())

    final_val_outputs = np.vstack(final_val_outputs)
    final_val_labels = np.vstack(final_val_labels)
    final_val_predictions = (final_val_outputs > 0.5).astype(float)

    final_metrics = generate_comprehensive_metrics(
        final_val_labels,
        final_val_predictions,
        final_val_outputs,
        class_names,
        "Transformer_Final",
        save_dir,
    )

    return best_auc, train_losses, val_losses, val_aucs, class_wise_aucs, final_metrics
