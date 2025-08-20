# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
from models import LSTMClassifier
from sklearn.metrics import classification_report, confusion_matrix
# <-- NEW: Import SMOTE
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

def train_classifier(X_train, y_train, X_val, y_val, params, training_params, output_dir, device, model_filename):
    """
    Trains the LSTM classifier using a hybrid approach: SMOTE for oversampling
    and a weighted loss function.
    """
    logger.info("Starting classifier training with hybrid SMOTE + Weighted Loss approach...")
    input_dim = X_train.shape[2]

    # --- (Unchanged) Calculate class weights from ORIGINAL training data ---
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    weights = total_samples / (len(unique_classes) * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    logger.info(f"Calculated class weights for loss function: {class_weights.cpu().numpy()}")
    
    # --- NEW: Apply SMOTE to balance the training data ---
    logger.info("Applying SMOTE to the training data...")
    # SMOTE works on 2D data, so we need to reshape our 3D sequence data
    n_samples, seq_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples, seq_len * n_features)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled_flat, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
    
    # Reshape the data back to the original 3D sequence format
    X_train = X_train_resampled_flat.reshape(-1, seq_len, n_features)
    y_train = y_train_resampled # y_train is now the resampled labels

    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Training data distribution after SMOTE: {dict(zip(unique, counts))}")
    # --- End of SMOTE section ---

    # --- Initialize Model, Loss, and Optimizer ---
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        dropout_rate=params['dropout_rate']
    ).to(device)

    # Use the weights calculated from the original imbalanced data
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    
    # --- Create DataLoader objects ---
    # Use the NEW resampled (X_train, y_train) data
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # Validation data remains original and imbalanced
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)

    best_val_accuracy = 0.0
    model_save_path = os.path.join(output_dir, model_filename)

    for epoch in range(training_params['epochs']):
        # --- Training Phase ---
        model.train()
        total_train_loss, train_correct, train_total = 0, 0, 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        total_val_loss, val_correct, val_total = 0, 0, 0
        last_epoch_labels = []
        last_epoch_predictions = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long)
                
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                if epoch == training_params['epochs'] - 1:
                    last_epoch_labels.extend(labels.cpu().numpy())
                    last_epoch_predictions.extend(predicted.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{training_params['epochs']} | "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"New best model saved to {model_save_path} with accuracy: {best_val_accuracy:.2f}%")
            
    logger.info("--- Training Complete ---")

    # --- Report for the model from the LAST epoch ---
    logger.info("--- Validation Report (Final Epoch Model) ---")
    class_names = ["Walking", "Wall-Rearing", "Rearing", "Grooming"]
    try:
        report = classification_report(last_epoch_labels, last_epoch_predictions, target_names=class_names, zero_division=0)
        conf_matrix = confusion_matrix(last_epoch_labels, last_epoch_predictions)
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
    except Exception as e:
        logger.error(f"Could not generate classification report for final epoch model: {e}")
    logger.info("-------------------------------------------")

    # --- Report for the BEST saved model ---
    logger.info("--- Validation Report (Best Model on Val Accuracy) ---")
    if not os.path.exists(model_save_path):
        logger.warning("Best model file not found. Skipping its evaluation.")
    else:
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        best_model_labels = []
        best_model_predictions = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long)
                outputs = model(sequences, lengths)
                _, predicted = torch.max(outputs.data, 1)
                best_model_labels.extend(labels.cpu().numpy())
                best_model_predictions.extend(predicted.cpu().numpy())

        try:
            report = classification_report(best_model_labels, best_model_predictions, target_names=class_names, zero_division=0)
            conf_matrix = confusion_matrix(best_model_labels, best_model_predictions)
            logger.info(f"\nClassification Report:\n{report}")
            logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        except Exception as e:
            logger.error(f"Could not generate classification report for best model: {e}")
    logger.info("------------------------------------------------------")

    return model_save_path