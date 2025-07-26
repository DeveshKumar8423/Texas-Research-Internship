import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader
from models import MotionCode, TimesNet, iTransformer, CrossFormer, Mamba

def get_model(model_name, config):
    if model_name == 'motion_code':
        return MotionCode(input_dim=config['num_features'], num_classes=config['num_classes'])
    elif model_name == 'timesnet':
        # The config dictionary is now passed directly to the model
        return TimesNet(config)
    elif model_name == 'itransformer':
        return iTransformer(seq_len=config['seq_len'], num_features=config['num_features'], num_classes=config['num_classes'])
    elif model_name == 'crossformer':
        return CrossFormer(seq_len=config['seq_len'], num_features=config['num_features'], num_classes=config['num_classes'])
    elif model_name == 'mamba':
        return Mamba(seq_len=config['seq_len'], num_features=config['num_features'], num_classes=config['num_classes'])
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_experiment(model_name):
    print(f"--- Starting Experiment for: {model_name.upper()} ---")
    
    try:
        X = np.load('data/X_processed.npy')
        y = np.load('data/y_processed.npy')
    except FileNotFoundError:
        print("Error: Processed data not found. Run preprocess.py first.")
        return

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42, stratify=y_tensor)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

    # --- MODIFIED STEP ---
    # This config dictionary now correctly sets top_k=1 for TimesNet
    config = {
        'seq_len': X_train.shape[1],
        'num_features': X_train.shape[2],
        'num_classes': len(torch.unique(y_tensor)),
        'd_model': 64, 'n_heads': 4, 'e_layers': 2, 'd_layers': 1, 'd_ff': 128,
        'dropout': 0.2, 'top_k': 1, 'num_kernels': 6, 'activation': 'gelu',
        'task_name': 'classification', 'pred_len': 0,
    }
    model = get_model(model_name, config)
    print(f"{model_name.upper()} model initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    print(f"Training on {device}...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    print("Training complete.")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    print("\n--- Evaluation Results ---")
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['PD', 'SWEDD'], zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate time-series models for Parkinson's classification.")
    parser.add_argument('--model', type=str, required=True, choices=['motion_code', 'timesnet', 'itransformer', 'crossformer', 'mamba'])
    args = parser.parse_args()
    run_experiment(args.model)