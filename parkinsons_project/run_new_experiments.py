import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from models import MotionCode

def run_experiment(task, data_type, qoi_target=None):
    """
    Main function to run an experiment.
    - task: 'classification' or 'prediction'
    - data_type: 'gait' or 'swing'
    - qoi_target: The name of the QOI to predict (e.g., 'asymmetry')
    """
    print(f"- Starting Experiment: TASK={task.upper()}, DATA={data_type.upper()} -")
    
    # 1. Load the correct dataset
    X = np.load(f'data/X_{data_type}.npy')
    
    if task == 'classification':
        y = np.load('data/y_processed.npy')
        print(f"Loaded classification data: X shape {X.shape}, y shape {y.shape}")
    elif task == 'prediction':
        y = np.load(f'data/y_qoi_{qoi_target}.npy')
        print(f"Loaded QOI prediction data: X shape {X.shape}, y shape {y.shape}")
    else:
        raise ValueError("Task must be 'classification' or 'prediction'")

    # 2. Setup DataLoaders
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long) if task == 'classification' else torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Stratify for classification, not for regression
    stratify_option = y_tensor if task == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42, stratify=stratify_option)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

    # 3. Initialize Motion Code model
    num_features = X_train.shape[2]
    num_classes = len(torch.unique(y_tensor)) if task == 'classification' else 1
    
    model = MotionCode(input_dim=num_features, num_classes=num_classes)
    print(f"Motion Code model initialized for {task}.")

    # 4. Train model
    device = torch.device("cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
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

    # 5. Evaluate model
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            if task == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
            else: # Regression
                all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    print("\n- Evaluation Results -")
    if task == 'classification':
        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Test Accuracy: {accuracy:.4f}')
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['PD', 'SWEDD'], zero_division=0))
    else: # Regresssion
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        print(f"Mean Absolute Error (MAE) for {qoi_target}: {mae:.4f}")
        print(f"R-squared (R2 Score) for {qoi_target}: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run new experiments for Parkinson's analysis.")
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'prediction'], help="Task to perform.")
    parser.add_argument('--data', type=str, required=True, choices=['gait', 'swing'], help="Dataset to use.")
    parser.add_argument('--qoi', type=str, default=None, choices=['asymmetry', 'speed'], help="QOI to predict for regression tasks.")
    args = parser.parse_args()
    
    if args.task == 'prediction' and args.qoi is None:
        raise ValueError("For prediction task, you must specify a --qoi target.")
        
    run_experiment(args.task, args.data, args.qoi)