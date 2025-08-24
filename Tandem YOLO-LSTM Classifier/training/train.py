# integrapose_lstm/training/train.py
import os
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Import the model factory from within the 'training' package
from .models import get_model

logger = logging.getLogger(__name__)

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_params: Dict,
    training_params: Dict,
    class_names: List[str],
    output_dir: str,
    device: str,
    model_filename: str
) -> str:
    """
    Trains the LSTM classifier.

    This function is fully configured by parameters, ensuring no hard-coded values.

    Args:
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        model_params (Dict): Parameters for the LSTMClassifier model.
        training_params (Dict): Parameters for the training loop (epochs, lr, etc.).
        class_names (List[str]): Ordered list of class names for reporting.
        output_dir (str): Directory to save the trained model and stats.
        device (str): The device to train on ('cuda' or 'cpu').
        model_filename (str): The filename for the saved model.

    Returns:
        str: The full path to the saved best model file.
    """
    logger.info("Starting classifier training...")
    input_dim = X_train.shape[2]
    num_classes = len(class_names)

    # --- 1. Calculate Class Weights for Imbalanced Datasets ---
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    weights = np.ones(num_classes)
    if len(unique_classes) > 0:
        total_samples = len(y_train)
        calculated_weights = total_samples / (len(unique_classes) * class_counts)
        for class_idx, weight in zip(unique_classes, calculated_weights):
            if class_idx < num_classes:
                weights[class_idx] = weight
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    logger.info(f"Calculated class weights for loss function: {np.round(weights, 2)}")

    # --- 2. Conditionally Apply SMOTE ---
    use_smote = training_params.get("use_smote", False)
    if use_smote and len(np.unique(y_train)) > 1:
        logger.info("Applying SMOTE to the training data...")
        n_samples, seq_len, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(n_samples, seq_len * n_features)

        smote = SMOTE(random_state=42)
        X_train_resampled_flat, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
        
        X_train = X_train_resampled_flat.reshape(-1, seq_len, n_features)
        y_train = y_train_resampled

        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Training data distribution after SMOTE: {dict(zip(unique, counts))}")
    elif not use_smote:
        logger.info("SMOTE is disabled by config. Training on original data distribution.")
    else:
        logger.warning("Skipping SMOTE because there is only one class in the training data.")

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model_type = model_params.get('model_type', 'lstm')
    model = get_model(
        model_name=model_type,
        model_params=model_params,
        input_dim=input_dim,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    
    # --- 4. Create DataLoader objects ---
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)

    best_val_accuracy = 0.0
    model_save_path = os.path.join(output_dir, model_filename)

    # --- 5. Training Loop ---
    for epoch in range(training_params['epochs']):
        model.train()
        total_train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        
        logger.info(
            f"Epoch {epoch+1:02d}/{training_params['epochs']} | "
            f"Train Loss: {total_train_loss/len(train_loader):.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"New best model saved to {model_save_path} with accuracy: {best_val_accuracy:.2f}%")
            
    logger.info("--- Training Complete ---")

    # --- 6. Final Evaluation with the Best Model ---
    logger.info("--- Final Report (Best Model on Val Accuracy) ---")
    if not os.path.exists(model_save_path):
        logger.error("Best model file not found. Cannot generate final report.")
        return ""
    
    # Reload best model for final report
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    try:
        all_possible_labels = list(range(num_classes))
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=class_names, 
            zero_division=0,
            labels=all_possible_labels
        )
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=all_possible_labels)
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
    except Exception as e:
        logger.error(f"Could not generate classification report for best model: {e}")
    
    return model_save_path