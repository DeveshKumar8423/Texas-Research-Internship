import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import the original MotionCode
import sys
sys.path.append('/Users/a1/Documents/GitHub/Texas-Research-Internship/parkinsons_project')
from models.motion_code import MotionCode

class ImprovedMotionCode(nn.Module):
    """Improved Motion Code with key optimizations"""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.4):
        super(ImprovedMotionCode, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Enhanced architecture
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Flatten: (batch, 2, features) -> (batch, 2*features)
        x = x.view(x.size(0), -1)
        return self.feature_extractor(x)

def add_noise_augmentation(X, noise_factor=0.03):
    """Add noise augmentation to training data"""
    noise = torch.randn_like(X) * noise_factor
    return X + noise

def improved_train_model(model, X_train, y_train, X_val, y_val, 
                        learning_rate=0.003, batch_size=32, epochs=100):
    """Enhanced training with multiple optimizations"""
    
    device = torch.device("cpu")
    model.to(device)
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train.numpy()), 
        y=y_train.numpy()
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-3,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, verbose=True
    )
    
    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 15
    
    print(f"Training with lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Data augmentation (50% chance)
            if np.random.random() > 0.5:
                batch_X = add_noise_augmentation(batch_X, noise_factor=0.03)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_acc = train_correct / train_total
        
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
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model weights
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            # Restore best model weights
            model.load_state_dict(best_model_state)
            break
    
    return best_val_acc

def run_improved_experiment(data_type='gait'):
    """Run experiment with all improvements applied"""
    
    print(f"=== IMPROVED EXPERIMENT: {data_type.upper()} DATA ===")
    print("Optimizations applied:")
    print("- Enhanced model architecture")
    print("- Class balancing with weights")
    print("- Data augmentation") 
    print("- Advanced optimization (AdamW + scheduler)")
    print("- Early stopping with best model restoration")
    print("- Gradient clipping")
    print("- Proper weight initialization")
    
    # Load data
    X = np.load(f'data/X_{data_type}.npy')
    y = np.load('data/y_processed.npy')
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    num_features = X_tensor.shape[2]
    num_classes = len(torch.unique(y_tensor))
    
    print(f"Features: {num_features}, Classes: {num_classes}")
    
    # Hyperparameter grid (simplified)
    hyperparams = [
        {'lr': 0.003, 'batch_size': 32, 'epochs': 100},
        {'lr': 0.001, 'batch_size': 16, 'epochs': 120}, 
        {'lr': 0.005, 'batch_size': 64, 'epochs': 80},
    ]
    
    best_overall_score = 0
    best_config = None
    all_results = []
    
    for i, params in enumerate(hyperparams):
        print(f"\n--- Configuration {i+1}/{len(hyperparams)} ---")
        print(f"Parameters: {params}")
        
        # 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tensor.numpy(), y_tensor.numpy())):
            print(f"  Fold {fold+1}/5")
            
            X_train_fold = X_tensor[train_idx]
            y_train_fold = y_tensor[train_idx]
            X_val_fold = X_tensor[val_idx]
            y_val_fold = y_tensor[val_idx]
            
            # Create fresh model for each fold
            model = ImprovedMotionCode(
                input_dim=num_features,
                num_classes=num_classes,
                dropout_rate=0.4
            )
            
            # Train and evaluate
            score = improved_train_model(
                model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                learning_rate=params['lr'],
                batch_size=params['batch_size'],
                epochs=params['epochs']
            )
            
            fold_scores.append(score)
            print(f"    Fold {fold+1} score: {score:.4f}")
        
        # Calculate mean and std for this configuration
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"  Configuration {i+1} results:")
        print(f"    Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")
        print(f"    Individual scores: {fold_scores}")
        
        all_results.append({
            'config': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores
        })
        
        if mean_score > best_overall_score:
            best_overall_score = mean_score
            best_config = params
    
    # Final evaluation on test set with best configuration
    print(f"\n=== FINAL EVALUATION ===")
    print(f"Best configuration: {best_config}")
    print(f"Best CV score: {best_overall_score:.4f}")
    
    # Train final model on full training set
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
    )
    
    final_model = ImprovedMotionCode(
        input_dim=num_features,
        num_classes=num_classes,
        dropout_rate=0.4
    )
    
    # Use validation split for early stopping
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Use best configuration, fallback to first if none found
    if best_config is None:
        best_config = hyperparams[0]
        print("Warning: No best config found, using first configuration")
    
    print("\nTraining final model...")
    improved_train_model(
        final_model, X_train_final, y_train_final, X_val_final, y_val_final,
        learning_rate=best_config['lr'],
        batch_size=best_config['batch_size'],
        epochs=best_config['epochs']
    )
    
    # Test set evaluation
    final_model.eval()
    device = torch.device("cpu")
    
    with torch.no_grad():
        X_test = X_test.to(device)
        outputs = final_model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        
    final_test_accuracy = accuracy_score(y_test.numpy(), predicted.cpu().numpy())
    
    print(f"\n=== FINAL RESULTS FOR {data_type.upper()} ===")
    print(f"Cross-validation accuracy: {best_overall_score:.4f}")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    print(f"Improvement target: 72.6%")
    print(f"Achievement: {'✓' if final_test_accuracy > 0.726 else '✗'}")
    
    if final_test_accuracy > 0.726:
        improvement = (final_test_accuracy - 0.726) * 100
        print(f"Improvement: +{improvement:.2f} percentage points!")
    else:
        needed = (0.726 - final_test_accuracy) * 100
        print(f"Still need: +{needed:.2f} percentage points")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        y_test.numpy(), 
        predicted.cpu().numpy(), 
        target_names=['PD', 'SWEDD']
    ))
    
    return {
        'data_type': data_type,
        'best_cv_score': best_overall_score,
        'final_test_accuracy': final_test_accuracy,
        'best_config': best_config,
        'all_results': all_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved Motion Code Experiments')
    parser.add_argument('--data', choices=['gait', 'swing'], default='gait',
                       help='Data type to use')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run improved experiment
    results = run_improved_experiment(args.data)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETED")
    print(f"Target achieved: {'YES' if results['final_test_accuracy'] > 0.726 else 'NO'}")
    print(f"Final accuracy: {results['final_test_accuracy']:.4f}")
    print(f"{'='*60}")
