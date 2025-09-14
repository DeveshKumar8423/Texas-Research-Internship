import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import argparse
from models.motion_code import MotionCode
from optimized_experiments import EnhancedMotionCode, DataAugmentation
import warnings
warnings.filterwarnings('ignore')

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper to make PyTorch models compatible with sklearn ensemble methods"""
    
    def __init__(self, model_class, model_params, training_params):
        self.model_class = model_class
        self.model_params = model_params
        self.training_params = training_params
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
            
        self.classes_ = np.unique(y.numpy())
        
        # Initialize model
        self.model = self.model_class(**self.model_params)
        
        # Training setup
        device = torch.device("cpu")
        self.model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.training_params.get('learning_rate', 0.001),
            weight_decay=1e-4
        )
        
        # Create data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.training_params.get('batch_size', 16), 
            shuffle=True
        )
        
        # Training loop
        epochs = self.training_params.get('epochs', 50)
        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Random data augmentation
                if np.random.random() > 0.5:
                    batch_X = DataAugmentation.add_noise(batch_X, noise_factor=0.03)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
            
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        self.model.eval()
        device = torch.device("cpu")
        X = X.to(device)
        
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
            
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        self.model.eval()
        device = torch.device("cpu")
        X = X.to(device)
        
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()

class AttentionMotionCode(nn.Module):
    """Motion Code with attention mechanism"""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(AttentionMotionCode, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=min(4, input_dim), 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 32),  # 2 for two time steps
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len=2, features)
        
        # Apply attention
        attended, _ = self.attention(x, x, x)
        
        # Process each time step
        batch_size, seq_len, features = attended.shape
        
        # Flatten and process through feature extractor
        attended_flat = attended.view(batch_size * seq_len, features)
        features_extracted = self.feature_extractor(attended_flat)
        
        # Reshape back and flatten for classification
        features_reshaped = features_extracted.view(batch_size, seq_len, -1)
        features_final = features_reshaped.view(batch_size, -1)
        
        # Classification
        output = self.classifier(features_final)
        
        return output

class LSTMMotionCode(nn.Module):
    """Motion Code with LSTM layers"""
    
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2, dropout_rate=0.3):
        super(LSTMMotionCode, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len=2, features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = hidden[-1]  # Shape: (batch, hidden_dim)
        
        # Classification
        output = self.classifier(last_hidden)
        
        return output

def create_ensemble_models(input_dim, num_classes):
    """Create different model architectures for ensemble"""
    
    models = []
    
    # Model 1: Enhanced Motion Code with different architecture
    model1 = PyTorchClassifier(
        model_class=EnhancedMotionCode,
        model_params={
            'input_dim': input_dim,
            'num_classes': num_classes,
            'dropout_rate': 0.3,
            'hidden_dims': [128, 64]
        },
        training_params={
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 60
        }
    )
    
    # Model 2: Enhanced Motion Code with different parameters
    model2 = PyTorchClassifier(
        model_class=EnhancedMotionCode,
        model_params={
            'input_dim': input_dim,
            'num_classes': num_classes,
            'dropout_rate': 0.2,
            'hidden_dims': [256, 128, 64]
        },
        training_params={
            'learning_rate': 0.005,
            'batch_size': 32,
            'epochs': 50
        }
    )
    
    # Model 3: Attention-based Motion Code
    model3 = PyTorchClassifier(
        model_class=AttentionMotionCode,
        model_params={
            'input_dim': input_dim,
            'num_classes': num_classes,
            'dropout_rate': 0.3
        },
        training_params={
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 60
        }
    )
    
    # Model 4: LSTM-based Motion Code
    model4 = PyTorchClassifier(
        model_class=LSTMMotionCode,
        model_params={
            'input_dim': input_dim,
            'num_classes': num_classes,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout_rate': 0.3
        },
        training_params={
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 55
        }
    )
    
    # Model 5: Another variant with different parameters
    model5 = PyTorchClassifier(
        model_class=EnhancedMotionCode,
        model_params={
            'input_dim': input_dim,
            'num_classes': num_classes,
            'dropout_rate': 0.4,
            'hidden_dims': [64, 32]
        },
        training_params={
            'learning_rate': 0.01,
            'batch_size': 8,
            'epochs': 70
        }
    )
    
    models = [
        ('enhanced_1', model1),
        ('enhanced_2', model2),
        ('attention', model3),
        ('lstm', model4),
        ('enhanced_3', model5)
    ]
    
    return models

def run_ensemble_experiment(data_type='gait'):
    """Run ensemble experiment with multiple model architectures"""
    
    print(f"=== Ensemble Experiment: {data_type.upper()} Data ===")
    
    # Load data
    X = np.load(f'data/X_{data_type}.npy')
    y = np.load('data/y_processed.npy')
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Convert to numpy for sklearn compatibility
    X_reshaped = X.reshape(X.shape[0], -1)  # Flatten for sklearn
    
    num_features = X.shape[2]
    num_classes = len(np.unique(y))
    
    print(f"Features: {num_features}, Classes: {num_classes}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Convert back to original shape for training
    X_train_original = X_train.reshape(-1, 2, num_features)
    X_test_original = X_test.reshape(-1, 2, num_features)
    
    # Create individual models
    models = create_ensemble_models(num_features, num_classes)
    
    # Train individual models and get their accuracies
    individual_accuracies = {}
    trained_models = []
    
    print("\nTraining individual models...")
    for name, model in models:
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train_original, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_original)
        accuracy = accuracy_score(y_test, y_pred)
        individual_accuracies[name] = accuracy
        trained_models.append((name, model))
        
        print(f"{name} accuracy: {accuracy:.4f}")
    
    # Create ensemble with voting
    print("\nCreating ensemble...")
    
    # Voting classifier (hard voting)
    ensemble_hard = VotingClassifier(
        estimators=trained_models,
        voting='hard'
    )
    
    # Note: For soft voting, we need to modify our PyTorchClassifier
    # to properly implement predict_proba
    ensemble_soft = VotingClassifier(
        estimators=trained_models,
        voting='soft'
    )
    
    # Train ensembles (actually just fits the voting mechanism)
    print("Training hard voting ensemble...")
    ensemble_hard.fit(X_train_original, y_train)
    
    print("Training soft voting ensemble...")
    ensemble_soft.fit(X_train_original, y_train)
    
    # Evaluate ensembles
    y_pred_hard = ensemble_hard.predict(X_test_original)
    y_pred_soft = ensemble_soft.predict(X_test_original)
    
    accuracy_hard = accuracy_score(y_test, y_pred_hard)
    accuracy_soft = accuracy_score(y_test, y_pred_soft)
    
    print(f"\n=== ENSEMBLE RESULTS ===")
    print(f"Individual model accuracies:")
    for name, acc in individual_accuracies.items():
        print(f"  {name}: {acc:.4f}")
    
    print(f"\nEnsemble accuracies:")
    print(f"  Hard voting: {accuracy_hard:.4f}")
    print(f"  Soft voting: {accuracy_soft:.4f}")
    
    print(f"\nBest individual model: {max(individual_accuracies.values()):.4f}")
    print(f"Best ensemble: {max(accuracy_hard, accuracy_soft):.4f}")
    
    # Detailed classification report for best ensemble
    best_ensemble = ensemble_soft if accuracy_soft > accuracy_hard else ensemble_hard
    best_pred = y_pred_soft if accuracy_soft > accuracy_hard else y_pred_hard
    
    print(f"\nDetailed classification report (best ensemble):")
    print(classification_report(y_test, best_pred, target_names=['PD', 'SWEDD']))
    
    # Save results
    results = {
        'data_type': data_type,
        'individual_accuracies': individual_accuracies,
        'ensemble_hard_accuracy': accuracy_hard,
        'ensemble_soft_accuracy': accuracy_soft,
        'best_accuracy': max(accuracy_hard, accuracy_soft)
    }
    
    # Save with pickle instead of numpy for dict
    import pickle
    with open(f'ensemble_results_{data_type}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble Motion Code Experiments')
    parser.add_argument('--data', choices=['gait', 'swing'], default='gait',
                       help='Data type to use')
    
    args = parser.parse_args()
    
    # Run ensemble experiment
    results = run_ensemble_experiment(args.data)
    
    print(f"\n=== FINAL ENSEMBLE RESULTS FOR {args.data.upper()} ===")
    print(f"Best individual model: {max(results['individual_accuracies'].values()):.4f}")
    print(f"Best ensemble accuracy: {results['best_accuracy']:.4f}")
    
    improvement = results['best_accuracy'] - max(results['individual_accuracies'].values())
    print(f"Ensemble improvement: {improvement:.4f} ({improvement*100:.2f}%)")
