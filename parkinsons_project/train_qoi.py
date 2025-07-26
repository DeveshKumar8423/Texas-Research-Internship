import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

# Import the models. We'll use iTransformer as our example.
from models import MotionCode, TimesNet, iTransformer, CrossFormer, Mamba

def get_model_for_qoi(model_name, config):
    """
    Initializes a model, ensuring its final layer is adapted for regression (outputting 1 value).
    """
    # Set num_classes to 1 for all models for this regression task
    config['num_classes'] = 1

    if model_name == 'itransformer':
        return iTransformer(seq_len=config['seq_len'], num_features=config['num_features'], num_classes=config['num_classes'])
    # Add other models here using the same pattern if you want to compare them
    # elif model_name == 'timesnet':
    #     return TimesNet(config) # Assuming TimesNet config is passed as a dict
    else:
        # For simplicity, we'll focus on iTransformer for this example.
        # You can expand this logic for all models.
        print(f"Model '{model_name}' not fully configured for QOI yet. Using iTransformer as default.")
        return iTransformer(seq_len=config['seq_len'], num_features=config['num_features'], num_classes=config['num_classes'])


def run_qoi_experiment(model_name):
    """
    Main function to run a regression experiment for a given model to predict a QOI.
    """
    print(f"--- Starting QOI Regression Experiment for: {model_name.upper()} ---")
    
    # 1. Load the processed feature data and the new regression target data
    try:
        X = np.load('data/X_processed.npy')
        y = np.load('data/y_severity_scores.npy')
        print(f"Loaded processed data: X shape {X.shape}, y shape {y.shape}")
    except FileNotFoundError:
        print("Error: Processed data not found. Run preprocess.py and create_dummy_scores.py first.")
        return

    # 2. Setup DataLoaders
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Target is a float with an added dimension

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

    # 3. Initialize the selected model for REGRESSION
    config = {
        'seq_len': X_train.shape[1],
        'num_features': X_train.shape[2],
    }
    model = get_model_for_qoi(model_name, config)
    print(f"{model_name.upper()} model initialized for regression.")

    # 4. Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss() # MODIFICATION: Using Mean Squared Error loss for regression
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

    # 5. Evaluate model with REGRESSION metrics
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print("\n--- Regression Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")

if __name__ == "__main__":
    # You can add argument parsing here to select different models if you expand the script
    # For now, we will default to the best model, iTransformer.
    parser = argparse.ArgumentParser(description="Train models for QOI regression.")
    parser.add_argument('--model', type=str, default='itransformer',
                        choices=['itransformer'], # Add other models here as you adapt them
                        help='The name of the model to run for QOI prediction.')
    args = parser.parse_args()
    run_qoi_experiment(args.model)