import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import argparse
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_features(X):
    """Create enhanced features from the time series data"""
    
    # X shape: (samples, 2, features) -> (samples, 2*features)
    X_flattened = X.reshape(X.shape[0], -1)
    
    # Statistical features
    features_list = [X_flattened]
    
    # Add interaction features (base * dual-task)
    base_features = X[:, 0, :]  # Base condition
    dual_features = X[:, 1, :]  # Dual-task condition
    
    # Feature interactions
    interaction_features = base_features * dual_features
    features_list.append(interaction_features)
    
    # Difference features (dual-task - base)
    diff_features = dual_features - base_features
    features_list.append(diff_features)
    
    # Ratio features (dual-task / base), avoid division by zero
    ratio_features = np.divide(dual_features, base_features, 
                              out=np.zeros_like(dual_features), 
                              where=base_features!=0)
    features_list.append(ratio_features)
    
    # Statistical moments
    # Mean across conditions
    mean_features = np.mean(X, axis=1)
    features_list.append(mean_features)
    
    # Standard deviation across conditions
    std_features = np.std(X, axis=1)
    features_list.append(std_features)
    
    # Min and max across conditions
    min_features = np.min(X, axis=1)
    max_features = np.max(X, axis=1)
    features_list.append(min_features)
    features_list.append(max_features)
    
    # Combine all features
    enhanced_X = np.concatenate(features_list, axis=1)
    
    return enhanced_X

def optimize_sklearn_models(X_train, y_train, X_val, y_val):
    """Optimize multiple sklearn models and return the best one"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    models_to_test = []
    
    # 1. Random Forest with hyperparameter tuning
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    }
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    models_to_test.append(('Random Forest', rf_grid.best_estimator_))
    
    # 2. Gradient Boosting
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='accuracy', n_jobs=-1)
    gb_grid.fit(X_train_scaled, y_train)
    models_to_test.append(('Gradient Boosting', gb_grid.best_estimator_))
    
    # 3. SVM
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced']
    }
    svm = SVC(random_state=42, probability=True)
    svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)
    models_to_test.append(('SVM', svm_grid.best_estimator_))
    
    # 4. Logistic Regression
    lr_params = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced']
    }
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    models_to_test.append(('Logistic Regression', lr_grid.best_estimator_))
    
    # 5. Neural Network
    mlp_params = {
        'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
        'alpha': [0.001, 0.01, 0.1],
        'learning_rate': ['adaptive'],
        'max_iter': [500]
    }
    mlp = MLPClassifier(random_state=42)
    mlp_grid = GridSearchCV(mlp, mlp_params, cv=3, scoring='accuracy', n_jobs=-1)
    mlp_grid.fit(X_train_scaled, y_train)
    models_to_test.append(('Neural Network', mlp_grid.best_estimator_))
    
    # Evaluate all models
    best_score = 0
    best_model = None
    best_name = None
    results = {}
    
    print("Individual model performances:")
    for name, model in models_to_test:
        y_pred = model.predict(X_val_scaled)
        score = accuracy_score(y_val, y_pred)
        results[name] = score
        print(f"{name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=models_to_test,
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble.predict(X_val_scaled)
    ensemble_score = accuracy_score(y_val, ensemble_pred)
    
    print(f"Ensemble (Soft Voting): {ensemble_score:.4f}")
    
    # Return best performing model/ensemble
    if ensemble_score > best_score:
        return ensemble, scaler, ensemble_score, 'Ensemble'
    else:
        return best_model, scaler, best_score, best_name

def run_sklearn_optimization(data_type='gait'):
    """Run optimized experiment using sklearn models"""
    
    print(f"=== SKLEARN OPTIMIZATION: {data_type.upper()} DATA ===")
    print("Optimizations applied:")
    print("- Enhanced feature engineering (interactions, differences, ratios)")
    print("- Multiple model architectures (RF, GB, SVM, LR, MLP)")
    print("- Hyperparameter grid search")
    print("- Feature scaling")
    print("- Class balancing")
    print("- Ensemble methods")
    print("- Cross-validation")
    
    # Load data (path relative to this script)
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / 'data'
    X_file = data_dir / f'X_{data_type}.npy'
    y_file = data_dir / 'y_processed.npy'
    if not X_file.exists():
        raise FileNotFoundError(f"Expected data file not found: {X_file}")
    if not y_file.exists():
        raise FileNotFoundError(f"Expected label file not found: {y_file}")
    X = np.load(X_file)
    y = np.load(y_file)
    
    print(f"\nOriginal data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create enhanced features
    print("Creating enhanced features...")
    X_enhanced = create_enhanced_features(X)
    print(f"Enhanced feature shape: {X_enhanced.shape}")
    
    # Cross-validation evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    print(f"\nRunning 5-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_enhanced, y)):
        print(f"\nFold {fold+1}/5")
        
        X_train_fold = X_enhanced[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X_enhanced[val_idx]
        y_val_fold = y[val_idx]
        
        # Optimize models for this fold
        best_model, scaler, score, model_name = optimize_sklearn_models(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold
        )
        
        cv_scores.append(score)
        print(f"Fold {fold+1} best model: {model_name}, Score: {score:.4f}")
    
    # Calculate CV statistics
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Mean CV accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    print(f"Individual fold scores: {cv_scores}")
    
    # Final model training on full dataset
    print(f"\n=== FINAL MODEL TRAINING ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train final model
    final_model, final_scaler, val_score, final_model_name = optimize_sklearn_models(
        X_train_opt, y_train_opt, X_val_opt, y_val_opt
    )
    
    # Check if final_model is None
    if final_model is None:
        print("Error: No final model was trained successfully")
        return None
    
    # Test set evaluation
    X_test_scaled = final_scaler.transform(X_test)
    y_pred_test = final_model.predict(X_test_scaled)
    final_test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n=== FINAL RESULTS FOR {data_type.upper()} ===")
    print(f"Cross-validation accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    print(f"Final model: {final_model_name}")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    print(f"Previous baseline: 0.549 (54.9%)")
    print(f"Previous best: 0.726 (72.6%)")
    
    if final_test_accuracy > 0.726:
        improvement = (final_test_accuracy - 0.726) * 100
        print(f"✓ TARGET ACHIEVED! Improvement: +{improvement:.2f} percentage points")
    elif final_test_accuracy > 0.549:
        improvement = (final_test_accuracy - 0.549) * 100
        print(f"✓ Improvement over baseline: +{improvement:.2f} percentage points")
    else:
        needed = (0.726 - final_test_accuracy) * 100
        print(f"✗ Still need: +{needed:.2f} percentage points to reach 72.6%")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['PD', 'SWEDD']))
    
    # Feature importance (if available)
    if hasattr(final_model, 'feature_importances_') and final_model_name != 'Ensemble':
        print(f"\nTop 10 Most Important Features:")
        feature_importance = final_model.feature_importances_
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    else:
        print(f"\nFeature importance not available for {final_model_name}")
    
    return {
        'data_type': data_type,
        'cv_mean': mean_cv_score,
        'cv_std': std_cv_score,
        'final_test_accuracy': final_test_accuracy,
        'final_model_name': final_model_name,
        'improvement_over_baseline': final_test_accuracy - 0.549,
        'achieved_target': final_test_accuracy > 0.726
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sklearn Optimization Experiments')
    parser.add_argument('--data', choices=['gait', 'swing'], default='gait',
                       help='Data type to use')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(42)
    
    # Run experiment
    results = run_sklearn_optimization(args.data)
    
    if results is not None:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION EXPERIMENT COMPLETED")
        print(f"Target (>72.6%): {'ACHIEVED' if results['achieved_target'] else 'NOT ACHIEVED'}")
        print(f"Final accuracy: {results['final_test_accuracy']:.4f}")
        print(f"Improvement: +{results['improvement_over_baseline']*100:.2f}% over baseline")
        print(f"{'='*60}")
    else:
        print("Experiment failed to complete successfully")
