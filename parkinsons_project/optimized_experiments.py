import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from models.motion_code import MotionCode
import itertools
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataAugmentation:
    """Time series data augmentation techniques"""
    
    @staticmethod
    def add_noise(X, noise_factor=0.05):
        """Add Gaussian noise to time series"""
        noise = torch.randn_like(X) * noise_factor
        return X + noise
    
    @staticmethod
    def time_scaling(X, scale_factor=0.1):
        """Scale time series values"""
        scale = 1 + (torch.rand(X.shape[0], 1, 1) - 0.5) * 2 * scale_factor
        return X * scale
    
    @staticmethod
    def magnitude_warping(X, sigma=0.2):
        """Apply magnitude warping to time series"""
        warping = torch.normal(1, sigma, size=(X.shape[0], 1, X.shape[2]))
        return X * warping

class EnhancedMotionCode(nn.Module):
    """Enhanced Motion Code with regularization"""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.3, hidden_dims=[128, 64]):
        super(EnhancedMotionCode, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Enhanced architecture with batch normalization and dropout
        layers = []
        prev_dim = input_dim * 2  # 2 conditions (base, dual-task)
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten time series: (batch, 2, features) -> (batch, 2*features)
        x = x.view(x.size(0), -1)
        return self.network(x)

def create_feature_interactions(X):
    """Create polynomial and interaction features"""
    batch_size, seq_len, features = X.shape
    
    # Flatten for easier processing
    X_flat = X.view(batch_size, seq_len * features)
    
    # Add polynomial features (squared terms)
    X_squared = X_flat ** 2
    
    # Add statistical features
    X_mean = X.mean(dim=1)  # Mean across time
    X_std = X.std(dim=1)    # Std across time
    X_max = X.max(dim=1)[0] # Max across time
    X_min = X.min(dim=1)[0] # Min across time
    
    # Combine all features
    enhanced_features = torch.cat([
        X_flat, X_squared, X_mean, X_std, X_max, X_min
    ], dim=1)
    
    return enhanced_features

def optimize_hyperparameters(X_train, y_train, X_val, y_val, input_dim, num_classes):
    """Grid search for optimal hyperparameters"""
    
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [8, 16, 32],
        'dropout_rate': [0.2, 0.3, 0.5],
        'hidden_dims': [[64, 32], [128, 64], [256, 128, 64]]
    }
    
    best_score = 0
    best_params = None
    
    print("Starting hyperparameter optimization...")
    
    # Get all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    for i, combination in enumerate(combinations):
        params = dict(zip(keys, combination))
        
        print(f"Testing combination {i+1}/{len(combinations)}: {params}")
        
        # Create model with current parameters
        model = EnhancedMotionCode(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout_rate=params['dropout_rate'],
            hidden_dims=params['hidden_dims']
        )
        
        # Train model
        score = train_and_evaluate_model(
            model, X_train, y_train, X_val, y_val,
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            epochs=30  # Reduced for grid search
        )
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"New best score: {best_score:.4f}")
    
    return best_params, best_score

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, 
                           learning_rate=0.001, batch_size=16, epochs=50,
                           use_class_weights=True):
    """Train model with given parameters and return validation accuracy"""
    
    device = torch.device("cpu")
    model.to(device)
    
    # Compute class weights for imbalanced data
    if use_class_weights:
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train.numpy()), 
            y=y_train.numpy()
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Data augmentation during training
            if np.random.random() > 0.5:  # 50% chance
                batch_X = DataAugmentation.add_noise(batch_X)
            if np.random.random() > 0.7:  # 30% chance
                batch_X = DataAugmentation.time_scaling(batch_X)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_predictions)
        scheduler.step(1 - val_acc)  # Use 1-accuracy as loss for scheduler
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc

def stratified_k_fold_evaluation(X, y, input_dim, num_classes, k=5, best_params=None):
    """Perform k-fold cross-validation"""
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = []
    
    print(f"Starting {k}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{k}")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Create model with best parameters
        if best_params:
            model = EnhancedMotionCode(
                input_dim=input_dim,
                num_classes=num_classes,
                dropout_rate=best_params['dropout_rate'],
                hidden_dims=best_params['hidden_dims']
            )
            
            score = train_and_evaluate_model(
                model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                learning_rate=best_params['learning_rate'],
                batch_size=best_params['batch_size'],
                epochs=50
            )
        else:
            # Use default parameters
            model = EnhancedMotionCode(input_dim=input_dim, num_classes=num_classes)
            score = train_and_evaluate_model(
                model, X_train_fold, y_train_fold, X_val_fold, y_val_fold
            )
        
        cv_scores.append(score)
        print(f"Fold {fold + 1} accuracy: {score:.4f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"\nCross-validation results:")
    print(f"Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Individual fold scores: {cv_scores}")
    
    return mean_score, std_score, cv_scores

def run_optimized_experiment(data_type='gait'):
    """Run complete optimized experiment"""
    
    print(f"=== Optimized Experiment: {data_type.upper()} Data ===")
    
    # Load data
    X = np.load(f'data/X_{data_type}.npy')
    y = np.load('data/y_processed.npy')
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    num_features = X_tensor.shape[2]
    num_classes = len(torch.unique(y_tensor))
    
    print(f"Features: {num_features}, Classes: {num_classes}")
    
    # Initial train/validation split for hyperparameter optimization
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
    )
    
    # Split training data for hyperparameter validation
    X_train_hp, X_val_hp, y_train_hp, y_val_hp = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Step 1: Hyperparameter optimization
    print("\n1. Hyperparameter Optimization")
    best_params, best_score = optimize_hyperparameters(
        X_train_hp, y_train_hp, X_val_hp, y_val_hp, num_features, num_classes
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best validation score: {best_score:.4f}")
    
    # Step 2: K-fold cross-validation with best parameters
    print("\n2. K-Fold Cross-Validation")
    mean_score, std_score, cv_scores = stratified_k_fold_evaluation(
        X_train, y_train, num_features, num_classes, k=5, best_params=best_params
    )
    
    # Step 3: Final model training and testing
    print("\n3. Final Model Evaluation")
    final_model = EnhancedMotionCode(
        input_dim=num_features,
        num_classes=num_classes,
        dropout_rate=best_params['dropout_rate'],
        hidden_dims=best_params['hidden_dims']
    )
    
    final_score = train_and_evaluate_model(
        final_model, X_train, y_train, X_test, y_test,
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        epochs=100  # More epochs for final training
    )
    
    print(f"\nFinal test accuracy: {final_score:.4f}")
    
    # Save results
    results = {
        'data_type': data_type,
        'best_params': best_params,
        'cv_mean': mean_score,
        'cv_std': std_score,
        'cv_scores': cv_scores,
        'final_test_accuracy': final_score
    }
    
    np.save(f'results_optimized_{data_type}.npy', results)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized Motion Code Experiments')
    parser.add_argument('--data', choices=['gait', 'swing'], default='gait',
                       help='Data type to use')
    
    args = parser.parse_args()
    
    # Run optimized experiment
    results = run_optimized_experiment(args.data)
    
    print(f"\n=== FINAL RESULTS FOR {args.data.upper()} ===")
    print(f"Cross-validation accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    print(f"Final test accuracy: {results['final_test_accuracy']:.4f}")
    print(f"Best hyperparameters: {results['best_params']}")
