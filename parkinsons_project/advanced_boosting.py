import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Optional imports guarded
try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover
    SMOTE = None
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

np.random.seed(42)


def engineer_features(X):
    base = X[:, 0, :]
    dual = X[:, 1, :]
    flat = X.reshape(X.shape[0], -1)
    diff = dual - base
    ratio = np.divide(dual, base, out=np.ones_like(dual), where=base != 0)
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    return np.concatenate([flat, diff, ratio, mean, std], axis=1)


def build_models(n_features):
    models = {}

    rf = RandomForestClassifier(n_estimators=400, max_depth=None, class_weight='balanced', random_state=42)
    models['rf'] = rf

    if lgb is not None:
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight='balanced',
            random_state=42
        )
    if xgb is not None:
        models['xgb'] = xgb.XGBClassifier(
            n_estimators=700,
            learning_rate=0.035,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1
        )

    return models


def cross_validate(models, X, y, use_smote=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {name: [] for name in models}
    aucs = {name: [] for name in models}

    feature_selector = RandomForestClassifier(n_estimators=500, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Feature selection
        feature_selector.fit(X_train, y_train)
        importances = feature_selector.feature_importances_
        threshold = np.percentile(importances, 40)  # keep top 60%
        mask = importances >= threshold
        X_train_sel = X_train[:, mask]
        X_val_sel = X_val[:, mask]

        # SMOTE (if enabled and available)
        if use_smote and SMOTE is not None:
            try:
                sm = SMOTE(random_state=42, k_neighbors=3)
                X_train_sel, y_train = sm.fit_resample(X_train_sel, y_train)
            except Exception:
                pass

        for name, model in models.items():
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_val_sel)
            acc = accuracy_score(y_val, y_pred)
            results[name].append(acc)
            # AUC if probs available
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_val_sel)[:, 1]
                    auc = roc_auc_score(y_val, proba)
                    aucs[name].append(auc)
                except Exception:
                    pass
        print(f"Fold {fold} complete")

    summary = {}
    for name in models:
        summary[name] = {
            'cv_mean': float(np.mean(results[name])) if results[name] else None,
            'cv_std': float(np.std(results[name])) if results[name] else None,
            'auc_mean': float(np.mean(aucs[name])) if aucs[name] else None,
            'auc_std': float(np.std(aucs[name])) if aucs[name] else None,
            'fold_scores': [float(x) for x in results[name]]
        }
    return summary, mask  # return last mask as heuristic


def build_stacking(models, X_train, y_train):
    base_estimators = []
    for name, model in models.items():
        base_estimators.append((name, model))
    meta = LogisticRegression(max_iter=1000, class_weight='balanced')
    stack = StackingClassifier(estimators=base_estimators, final_estimator=meta, stack_method='auto', passthrough=False, n_jobs=-1)
    stack.fit(X_train, y_train)
    return stack


def main():
    parser = argparse.ArgumentParser(description='Advanced boosting & stacking for PD vs SWEDD')
    parser.add_argument('--data', choices=['gait'], default='gait')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE oversampling')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / 'data'
    X = np.load(data_dir / f'X_{args.data}.npy')
    y = np.load(data_dir / 'y_processed.npy')

    print(f"Loaded X={X.shape}, y={y.shape}")
    X_feat = engineer_features(X)
    print(f"Engineered feature matrix: {X_feat.shape}")

    models = build_models(X_feat.shape[1])
    print(f"Models: {list(models.keys())}")

    cv_summary, feature_mask = cross_validate(models, X_feat, y, use_smote=not args.no_smote)
    print("\nCross-validation summary:")
    for name, stats in cv_summary.items():
        print(f"{name}: acc={stats['cv_mean']:.4f}±{stats['cv_std']:.4f}")

    # Train/test split for final eval
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train[:, feature_mask]
    X_test = X_test[:, feature_mask]

    # Class weights for meta models
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {c: w for c, w in zip(np.unique(y_train), class_weights)}

    # Refit base models on selected features
    for model in models.values():
        model.fit(X_train, y_train)

    # Stacking
    stack = build_stacking(models, X_train, y_train)

    # Calibration (if possible)
    calibrated = None
    try:
        calibrated = CalibratedClassifierCV(stack, method='isotonic', cv=3)
        calibrated.fit(X_train, y_train)
    except Exception:
        calibrated = None

    def evaluate(model, label):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = None
        if hasattr(model, 'predict_proba'):
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except Exception:
                pass
        print(f"{label}: acc={acc:.4f} f1={f1:.4f} auc={auc if auc else 'NA'}")
        return acc, f1, auc

    final_results = {}
    for name, m in models.items():
        acc, f1, auc = evaluate(m, f"Final {name}")
        final_results[name] = {'accuracy': acc, 'f1': f1, 'auc': auc}

    acc_stack, f1_stack, auc_stack = evaluate(stack, 'Stacking')
    final_results['stacking'] = {'accuracy': acc_stack, 'f1': f1_stack, 'auc': auc_stack}

    if calibrated is not None:
        acc_cal, f1_cal, auc_cal = evaluate(calibrated, 'Calibrated Stacking')
        final_results['calibrated_stacking'] = {'accuracy': acc_cal, 'f1': f1_cal, 'auc': auc_cal}

    best_model_name = max(final_results, key=lambda k: final_results[k]['accuracy'])
    best_acc = final_results[best_model_name]['accuracy']

    print("\nBest model:", best_model_name, f"accuracy={best_acc:.4f}")

    out = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'n_samples': int(X.shape[0]),
        'feature_dim_engineered': int(X_feat.shape[1]),
        'cv_summary': cv_summary,
        'final_results': final_results,
        'best_model': best_model_name,
        'best_accuracy': float(best_acc)
    }
    out_path = script_dir / 'results_boosting.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved boosting results to {out_path}")

    print("\nClassification report (best model):")
    best_model = stack if best_model_name == 'stacking' else (
        calibrated if best_model_name == 'calibrated_stacking' and calibrated is not None else models.get(best_model_name)
    )
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=['PD', 'SWEDD']))

if __name__ == '__main__':
    main()
