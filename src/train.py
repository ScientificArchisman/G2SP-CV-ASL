import torch 
from torch import nn
from tqdm import tqdm
import os
import time
import pickle
from src.model import GCN
from src.load_data import load_dataloader, load_data
import yaml 
from typing import Tuple
from pprint import pprint
from datetime import datetime

def calculate_metrics_multiclass(y_pred, y_true, num_classes):
    """
    Calculate accuracy, precision, recall, and F1 score for multi-class classification.

    Args:
        y_pred (torch.Tensor): Raw logits or probabilities from the model.
        y_true (torch.Tensor): Ground truth class labels (integers 0 to num_classes - 1).
        num_classes (int): Number of classes.

    Returns:
        dict: Metrics - accuracy, precision, recall, F1 score.
    """
    # Convert logits to predicted classes
    y_pred_classes = torch.argmax(y_pred, dim=1)
    
    # Initialize metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    # Calculate accuracy
    accuracy = (y_pred_classes == y_true).float().mean().item()

    for class_idx in range(num_classes):
        # True Positives, False Positives, False Negatives
        TP = ((y_pred_classes == class_idx) & (y_true == class_idx)).sum().item()
        FP = ((y_pred_classes == class_idx) & (y_true != class_idx)).sum().item()
        FN = ((y_pred_classes != class_idx) & (y_true == class_idx)).sum().item()

        # Precision, Recall, F1 Score for the class
        precision = TP / (TP + FP + 1e-7) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN + 1e-7) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-7) if (precision + recall) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    # Macro-Averaged Metrics
    macro_precision = sum(precision_per_class) / num_classes
    macro_recall = sum(recall_per_class) / num_classes
    macro_f1 = sum(f1_per_class) / num_classes

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1
    }


def train_model(model: nn.Module, optimiser: torch.optim.Optimizer,
          criterion: torch.nn.modules.loss._Loss, 
          train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
          config: dict = None, **kwargs) -> Tuple[nn.Module, dict]:
    """ Train a model based on the provided configuration in the config file
    Args:
        model (nn.Module): PyTorch model to train
        optimiser (torch.optim.Optimizer): Optimiser to use for training
        criterion (torch.nn.modules.loss._Loss): Loss function to use for training
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data
        config (dict): Configuration dictionary (default: None)"""
    
    # Set up the hyperparameters
    if config:
        NUM_EPOCHS = config.get("training").get("epochs", 100)
        EARLY_STOPPING_PATIENCE = config.get("training").get("early_stopping").get("patience", 15)
        MODEL_SAVE_DIR = config.get("logging").get("checkpoint_path", "Models") 
        LOG_DIR = config.get("logging").get("log_dir", "logs")
    else:
        NUM_EPOCHS = kwargs.get("epochs", 100)  
        EARLY_STOPPING_PATIENCE = kwargs.get("early_stopping_patience", 15)
        MODEL_SAVE_DIR = kwargs.get("model_save_dir", "Models")
        LOG_DIR = kwargs.get("log_dir", "logs")

    WEIGHT_DECAY = optimiser.param_groups[0].get("weight_decay", 0)
    LEARNING_RATE = optimiser.param_groups[0].get("lr", 0.01)
    N_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    start_time = time.strftime("%Y-%m-%d, %H:%M:%S")
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"{session_id}.pth")
    logs_save_path = os.path.join(LOG_DIR, f"{session_id}.pkl")


    training_info = {
        "📅 Training Start Time": start_time,
        "📈 Total Number of Epochs": NUM_EPOCHS,
        "💻 Device Used for Training": device,
        "🆔 Session ID": session_id,

        "🔢 Number of Trainable Parameters": N_PARAMS,
        "🚦 Early Stopping Patience": EARLY_STOPPING_PATIENCE,
        "📉 Weight Decay": WEIGHT_DECAY,
        "📈 Initial Learning Rate": LEARNING_RATE,

        "📂 Model Save Path": model_save_path,
        "📄 Logs Save Path": logs_save_path
    }

    print("\n\n════════════════════════════════════════════")
    print("TRAINING SESSION START")
    print("════════════════════════════════════════════")
    pprint(training_info)
    print("════════════════════════════════════════════")


    train_losses, val_losses = [], [] # Lists to store the training and validation losses
    best_val_loss = float("inf") # Variable to store the best validation loss
    patience = 0 # Variable to store the patience
    
    model.to(device) # Move the model to the device
    loop = tqdm(range(NUM_EPOCHS), desc="Training", position=0, leave=True)
    for epoch in loop:
        model.train()
        epoch_loss = 0.0
        model.to(device)

         # Wrap train_loader with tqdm for a single-line progress bar
        for adj_matrix, features, labels in train_loader:
            # Move tensors to the specified device
            features, adj_matrix, labels = features.to(device), adj_matrix.to(device), labels.to(device)
    
            # Forward pass
            optimiser.zero_grad()  # Zero out the gradients
            print(f"Features Shape: {features.shape}")
            print(f"Adj Matrix Shape: {adj_matrix.shape}")
            output = model(features, adj_matrix)  # Forward pass
            loss = criterion(output, labels)  # Calculate the loss
            loss.backward()  # Backward pass
            optimiser.step()  # Update the weights
    
            epoch_loss += loss.item()


        avg_epoch_loss = epoch_loss / len(train_loader) # Calculate the average loss for the epoch
        train_losses.append(avg_epoch_loss)
        

        # Validation loop
        model.eval()
        val_epoch_loss = 0.0

        with torch.no_grad():
            for adj_matrix, features, labels in val_loader:
                features, adj_matrix, labels = features.to(device), adj_matrix.to(device), labels.to(device)
                output = model(features, adj_matrix)
                loss = criterion(output, labels)
                val_epoch_loss += loss.item()

        avg_val_epoch_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_epoch_loss)

        loop.set_description(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_epoch_loss:.4f}, Patience: {patience}")

        if avg_val_epoch_loss < best_val_loss:
            best_val_loss = avg_val_epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            patience = 0
        else:
            patience += 1
            if patience > EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    end_time = time.strftime("%Y-%m-%d, %H:%M:%S")
    training_info = {
        "Training Session Summary": {
            "📅 Training Start Time": start_time,
            "⏰ Estimated Training End": end_time,
            "🆔 Session ID": session_id, 
        },
        "File Paths": {
            "📂 Model Saved at": model_save_path,
            "📄 Logs Saved at": logs_save_path
        },
        "Best Performance Metrics": {
            "🔍 Best Validation Loss": round(best_val_loss, 4),
            "🔍 Best Training Loss": round(min(train_losses), 4),
            "🏆 Best Epoch": best_epoch
        }
    }
        
    print("\n\n════════════════════════════════════════════")
    print("TRAINING SESSION END")
    print("════════════════════════════════════════════")
    print(training_info)
    print("════════════════════════════════════════════")


    results = {
        "description": training_info,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss, 
        "model_path": model_save_path, 
    }

    # Save the results dictionary in logs 
    with open(logs_save_path, "wb") as file:
        pickle.dump(results, file)

    
    return model, results


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)


    # Setup the data loaders
    BASE_ADJ_MATRIX_PATH = config.get("dataset").get("adj_dir", "data/processed_data/adj_matrix")
    BASE_FEATURES_PATH = config.get("dataset").get("features_dir", "data/processed_data/extended_features")

    adj_matrices, features, labels = load_data(features_dir = BASE_FEATURES_PATH, 
                                          adj_matrix_dir = BASE_ADJ_MATRIX_PATH)


    train_loader, val_loader, test_loader = load_dataloader(adj_matrices, features, labels, config)

    # Setup the model
    INPUT_DIM = config.get("model").get("input_dim", 4)
    HIDDEN_DIMS = config.get("model").get("hidden_dims", [64, 128, 32])
    NUM_CLASSES = config.get("model").get("num_classes", 32)
        
    LEARNING_RATE = config.get("training").get("learning_rate", 4e-2)
    WEIGHT_DECAY = config.get("training").get("weight_decay", 0.001)

    model = GCN(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, num_classes=NUM_CLASSES, num_landmarks = 21)

    # Setup the optimizer
    optimizer_name = config.get("training").get("optimizer", "optimiser").get("type", "Adam")
    if optimizer_name not in ["Adam", "SGD"]:
        raise ValueError(f"Optimizer {optimizer_name} not supported. Supported optimizers are Adam and SGD")

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=WEIGHT_DECAY)


    # Setup the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    print(f"Hidden Dimensions: {HIDDEN_DIMS}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # for adj_matrix, features, labels in train_loader:
    #             # Move tensors to the specified device
    #             features, adj_matrix, labels = features.to(device), adj_matrix.to(device), labels.to(device)
        
    #             # Forward pass
    #             output = model(features, adj_matrix)  # Forward pass
    #             break

    # print(f"Output Shape: {output.shape}")
    # print(f"Labels Shape: {labels.shape}")
    # print(f"Features Shape: {features.shape}")
    model, results = train_model(model=model, 
                           optimiser=optimizer, 
                           criterion=loss_fn, 
                           train_loader=train_loader, 
                           val_loader=val_loader, 
                           config=config, 
                           device=device)


