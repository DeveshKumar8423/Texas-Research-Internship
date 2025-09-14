import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import argparse

def create_enhanced_features(X):
    """Create enhanced features from the time series data"""
    # X shape: (samples, 2, features) -> flatten and add interactions
    X_flattened = X.reshape(X.shape[0], -1)
    
    # Feature interactions between base and dual-task
    base_features = X[:, 0, :]  # Base condition
    dual_features = X[:, 1, :]  # Dual-task condition
    
    # Difference features (dual-task - base)
    diff_features = dual_features - base_features
    
    # Ratio features (dual-task / base), avoid division by zero
    ratio_features = np.divide(dual_features, base_features, 
                              out=np.ones_like(dual_features), 
                              where=base_features!=0)
    
    # Statistical features
    mean_features = np.mean(X, axis=1)
    std_features = np.std(X, axis=1)
    
    # Combine all features
    enhanced_X = np.concatenate([
        X_flattened, diff_features, ratio_features, mean_features, std_features
    ], axis=1)
    
    return enhanced_X

def quick_optimization_test(data_type='gait'):
    """Quick test of optimization strategies"""
    
    print(f"=== QUICK OPTIMIZATION TEST: {data_type.upper()} DATA ===")
    
    # Load data
    try:
        X = np.load(f'data/X_{data_type}.npy')
        y = np.load('data/y_processed.npy')
        print(f"Data loaded successfully: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Class distribution: {np.bincount(y)}")
    
    # Test 1: Baseline (flattened features)
    print("\n--- Test 1: Baseline (Flattened Features) ---")
    X_baseline = X.reshape(X.shape[0], -1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, 
                               class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    baseline_accuracy = accuracy_score(y_test, y_pred)
    print(f"Baseline RF accuracy: {baseline_accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    print(f"Baseline CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Test 2: Enhanced features
    print("\n--- Test 2: Enhanced Features ---")
    X_enhanced = create_enhanced_features(X)
    print(f"Enhanced feature shape: {X_enhanced.shape}")
    
    # Split enhanced data
    X_train_enh, X_test_enh, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale enhanced features
    scaler_enh = StandardScaler()
    X_train_enh_scaled = scaler_enh.fit_transform(X_train_enh)
    X_test_enh_scaled = scaler_enh.transform(X_test_enh)
    
    # Train enhanced RF
    rf_enh = RandomForestClassifier(n_estimators=300, max_depth=25, 
                                   min_samples_split=3, class_weight='balanced', 
                                   random_state=42)
    rf_enh.fit(X_train_enh_scaled, y_train)
    
    # Evaluate enhanced
    y_pred_enh = rf_enh.predict(X_test_enh_scaled)
    enhanced_accuracy = accuracy_score(y_test, y_pred_enh)
    print(f"Enhanced RF accuracy: {enhanced_accuracy:.4f}")
    
    # Enhanced CV
    cv_scores_enh = cross_val_score(rf_enh, X_train_enh_scaled, y_train, cv=5)
    print(f"Enhanced CV accuracy: {cv_scores_enh.mean():.4f} ± {cv_scores_enh.std():.4f}")
    
    # Test 3: Gradient Boosting on enhanced features
    print("\n--- Test 3: Gradient Boosting ---")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, 
                                   max_depth=5, random_state=42)
    gb.fit(X_train_enh_scaled, y_train)
    
    y_pred_gb = gb.predict(X_test_enh_scaled)
    gb_accuracy = accuracy_score(y_test, y_pred_gb)
    print(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
    
    # GB CV
    cv_scores_gb = cross_val_score(gb, X_train_enh_scaled, y_train, cv=5)
    print(f"GB CV accuracy: {cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")
    
    # Test 4: Simple ensemble
    print("\n--- Test 4: Simple Ensemble ---")
    y_pred_ensemble = (rf_enh.predict_proba(X_test_enh_scaled)[:, 1] + 
                      gb.predict_proba(X_test_enh_scaled)[:, 1]) / 2
    y_pred_ensemble_binary = (y_pred_ensemble > 0.5).astype(int)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble_binary)
    print(f"Simple ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Results summary
    results = {
        'baseline': baseline_accuracy,
        'enhanced': enhanced_accuracy,
        'gradient_boosting': gb_accuracy,
        'ensemble': ensemble_accuracy
    }
    
    best_method = max(results.keys(), key=lambda k: results[k])
    best_accuracy = results[best_method]
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Baseline (flattened): {baseline_accuracy:.4f}")
    print(f"Enhanced features: {enhanced_accuracy:.4f}")
    print(f"Gradient Boosting: {gb_accuracy:.4f}")
    print(f"Simple Ensemble: {ensemble_accuracy:.4f}")
    print(f"\nBest method: {best_method} ({best_accuracy:.4f})")
    
    # Compare with targets
    print(f"\nComparison with targets:")
    print(f"Previous baseline (54.9%): {0.549:.3f}")
    print(f"Previous best (72.6%): {0.726:.3f}")
    print(f"Current best: {best_accuracy:.3f}")
    
    if best_accuracy > 0.726:
        improvement = (best_accuracy - 0.726) * 100
        print(f"✅ TARGET ACHIEVED! +{improvement:.2f} percentage points above 72.6%")
    elif best_accuracy > 0.549:
        improvement = (best_accuracy - 0.549) * 100
        print(f"✅ Improvement: +{improvement:.2f} percentage points above baseline")
    else:
        print(f"❌ Below baseline")
    
    # Detailed report for best model
    if best_method == 'ensemble':
        print(f"\nDetailed Classification Report (Ensemble):")
        print(classification_report(y_test, y_pred_ensemble_binary, target_names=['PD', 'SWEDD']))
    elif best_method == 'gradient_boosting':
        print(f"\nDetailed Classification Report (Gradient Boosting):")
        print(classification_report(y_test, y_pred_gb, target_names=['PD', 'SWEDD']))
    else:
        print(f"\nDetailed Classification Report ({best_method}):")
        if best_method == 'enhanced':
            print(classification_report(y_test, y_pred_enh, target_names=['PD', 'SWEDD']))
        else:
            print(classification_report(y_test, y_pred, target_names=['PD', 'SWEDD']))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick Optimization Test')
    parser.add_argument('--data', choices=['gait', 'swing'], default='gait',
                       help='Data type to use')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(42)
    
    # Run test
    results = quick_optimization_test(args.data)
    
    print(f"\n{'='*50}")
    print(f"QUICK OPTIMIZATION TEST COMPLETED")
    print(f"{'='*50}")
