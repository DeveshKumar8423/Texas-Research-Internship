import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def test_accuracy_improvements():
    """Test actual accuracy improvements on gait data"""
    
    print("=== ACCURACY IMPROVEMENT TEST ===")
    print("Baseline reference: 54.9% (from previous runs)")
    print("Target: > 72.6%")
    print()
    
    # Load gait data (robust path resolution relative to this script)
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / 'data'
    X_path = data_dir / 'X_gait.npy'
    y_path = data_dir / 'y_processed.npy'
    if not X_path.exists():
        raise FileNotFoundError(f"Could not find gait feature file at {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Could not find label file at {y_path}")
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    print(f"Class distribution: PD={np.sum(y==0)}, SWEDD={np.sum(y==1)}")
    print()
    
    # Test 1: Simple baseline (just flattened features)
    print("--- Test 1: Simple Baseline ---")
    X_flat = X.reshape(X.shape[0], -1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Simple Random Forest
    rf_simple = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_simple.fit(X_train, y_train)
    y_pred_simple = rf_simple.predict(X_test)
    acc_simple = accuracy_score(y_test, y_pred_simple)
    
    print(f"Simple RF accuracy: {acc_simple:.4f} ({acc_simple*100:.1f}%)")
    
    # Test 2: Enhanced features
    print("\n--- Test 2: Enhanced Features ---")
    
    # Create enhanced features
    base_features = X[:, 0, :]  # Base condition
    dual_features = X[:, 1, :]  # Dual-task condition
    
    # Feature engineering
    diff_features = dual_features - base_features
    ratio_features = np.divide(dual_features, base_features, 
                              out=np.ones_like(dual_features), 
                              where=base_features!=0)
    mean_features = np.mean(X, axis=1)
    std_features = np.std(X, axis=1)
    
    # Combine features
    X_enhanced = np.concatenate([
        X_flat, diff_features, ratio_features, mean_features, std_features
    ], axis=1)
    
    print(f"Enhanced features shape: {X_enhanced.shape}")
    
    # Split enhanced data
    X_train_enh, X_test_enh, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enh)
    X_test_scaled = scaler.transform(X_test_enh)
    
    # Enhanced Random Forest
    rf_enhanced = RandomForestClassifier(
        n_estimators=300, 
        max_depth=20, 
        min_samples_split=3,
        class_weight='balanced',
        random_state=42
    )
    rf_enhanced.fit(X_train_scaled, y_train)
    y_pred_enhanced = rf_enhanced.predict(X_test_scaled)
    acc_enhanced = accuracy_score(y_test, y_pred_enhanced)
    
    print(f"Enhanced RF accuracy: {acc_enhanced:.4f} ({acc_enhanced*100:.1f}%)")
    
    # Test 3: Multiple Random Forest ensemble
    print("\n--- Test 3: Simple Ensemble ---")
    
    # Train multiple RF models with different parameters
    rf1 = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
    rf2 = RandomForestClassifier(n_estimators=400, max_depth=25, random_state=123, class_weight='balanced')
    rf3 = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=456, class_weight='balanced')
    
    rf1.fit(X_train_scaled, y_train)
    rf2.fit(X_train_scaled, y_train)
    rf3.fit(X_train_scaled, y_train)
    
    # Ensemble predictions (majority voting)
    pred1 = rf1.predict(X_test_scaled)
    pred2 = rf2.predict(X_test_scaled)
    pred3 = rf3.predict(X_test_scaled)
    
    # Majority vote
    ensemble_pred = np.array([
        1 if (p1 + p2 + p3) >= 2 else 0 
        for p1, p2, p3 in zip(pred1, pred2, pred3)
    ])
    
    acc_ensemble = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble accuracy: {acc_ensemble:.4f} ({acc_ensemble*100:.1f}%)")
    
    # Results summary
    print(f"\n{'='*50}")
    print("ACCURACY RESULTS SUMMARY")
    print(f"{'='*50}")
    
    baseline_ref = 0.549  # 54.9% from previous runs
    target = 0.726       # 72.6% target
    
    results = {
        'Previous baseline': baseline_ref,
        'Simple RF': acc_simple,
        'Enhanced RF': acc_enhanced, 
        'Ensemble': acc_ensemble
    }
    
    best_accuracy = max(acc_simple, acc_enhanced, acc_ensemble)
    best_method = 'Simple RF' if best_accuracy == acc_simple else ('Enhanced RF' if best_accuracy == acc_enhanced else 'Ensemble')
    
    for method, acc in results.items():
        status = ""
        if acc > target:
            status = "TARGET ACHIEVED"
        elif acc > baseline_ref:
            status = "IMPROVED"
        else:
            status = ""
        print(f"{method:<20}: {acc:.4f} ({acc*100:.1f}%) {status}")
    
    print(f"\nBest result: {best_method} = {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")
    
    # Calculate improvements
    improvement_over_baseline = (best_accuracy - baseline_ref) * 100
    if best_accuracy > target:
        improvement_over_target = (best_accuracy - target) * 100
        print(f"TARGET EXCEEDED by {improvement_over_target:.1f} percentage points!")
    
    print(f"Improvement over baseline: +{improvement_over_baseline:.1f} percentage points")
    
    # Show detailed classification report for best model
    print(f"\nDetailed Classification Report ({best_method}):")
    if best_method == 'Ensemble':
        print(classification_report(y_test, ensemble_pred, target_names=['PD', 'SWEDD']))
    elif best_method == 'Enhanced RF':
        print(classification_report(y_test, y_pred_enhanced, target_names=['PD', 'SWEDD']))
    else:
        print(classification_report(y_test, y_pred_simple, target_names=['PD', 'SWEDD']))
    
    return {
        'simple_rf': acc_simple,
        'enhanced_rf': acc_enhanced,
        'ensemble': acc_ensemble,
        'best_accuracy': best_accuracy,
        'improvement': improvement_over_baseline,
        'target_achieved': best_accuracy > target
    }

if __name__ == "__main__":
    try:
        results = test_accuracy_improvements()
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.1f}%)")
        print(f"IMPROVEMENT: +{results['improvement']:.1f} percentage points")
        print(f"TARGET (72.6%): {'ACHIEVED' if results['target_achieved'] else 'NOT ACHIEVED'}")
        print(f"{'='*50}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
