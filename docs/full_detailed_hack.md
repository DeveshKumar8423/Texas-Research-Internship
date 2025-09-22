# Parkinson's IMU Multi-Level Pipeline – Comprehensive Technical Hack Document

**Maximum Achieved Accuracy (PD vs SWEDD, Gait Data): 80.4%**  
Baseline (combined / weaker config): 54.9% → MotionCode gait: 72.6% → Engineered RandomForest: 80.4%

---
## 1. Objectives
1. Build a reproducible multi-level pipeline from raw IMU-derived kinematic CSV to domain-separated model-ready tensors.
2. Classify Parkinson's Disease (PD) vs SWEDD.
3. Explore cross-modality regression of quantitative outcomes (QOIs) – asymmetry & speed.
4. Improve accuracy beyond 72.6% through principled optimization.
5. Document feature engineering, scripts, architectures, and observed performance ceilings.

---
## 2. Data Inventory
| File | Description | Shape / Notes |
|------|-------------|---------------|
| `parkinsons_project/data/PPMI_Gait_Data.csv` | Source processed kinematic feature CSV | Raw tabular feature space |
| `parkinsons_project/data/MDS_UPDRS_Part_III.csv` | Clinical motor scores (optional) | Not yet fully integrated |
| `X_processed.npy` | Intermediate pseudo time-series (all features) | (167, 2, F) |
| `y_processed.npy` | Binary labels (0=PD, 1=SWEDD) | (167,) |
| `X_gait.npy` | Domain-separated gait features | (167, 2, 10) |
| `X_swing.npy` | Domain-separated arm swing features | (167, 2, 8) |
| `y_qoi_asymmetry.npy` | Arm swing asymmetry values | (167,) |
| `y_qoi_speed.npy` | Gait speed values | (167,) |
| `y_severity_scores.npy` | (If generated) severity surrogate | (167,) |
| `results_test_accuracy.json` | 80.4% classification record | JSON artifact |
| `results_boosting.json` | Boosting/stacking outcomes | JSON artifact |

Conditions ordering: `[Base, Dual-task]`  
Each feature array axis meaning: `(samples, 2 conditions, features)`

---
## 3. Domain Feature Sets
| Domain | Features | Count |
|--------|----------|-------|
| Gait | `CAD, JERK_T, SP_, STEP_REG, STEP_SYM, STR_CV, STR_T, SYM, T_AMP, TRA` | 10 |
| Swing | `ASA, ASYM_IND, LA_AMP, LA_STD, L_JERK, RA_AMP, RA_STD, R_JERK` | 8 |

---
## 4. Transformation Levels
```
Level 1  Raw / CSV (kinematic summaries)
Level 2  Selected kinematic columns → structured table
Level 3  Pseudo time-series tensor (N, 2, F)
Level 4  Domain separation (gait vs swing arrays)
Level 5  Task outputs (classification / regression)
```
Scripts implementing transitions:
| Stage | Script | Purpose |
|-------|--------|---------|
| L2→L3 | `preprocess.py` | Build `X_processed.npy`, `y_processed.npy` |
| L3→L4 | `prepare_separated_data.py` | Split into `X_gait.npy`, `X_swing.npy` & QOIs |
| L4→L5 | `run_new_experiments.py` | Train MotionCode classification/regression |

---
## 5. Core Scripts Overview
| Script | Role | Key Functions / Notes |
|--------|------|-----------------------|
| `run_new_experiments.py` | Baseline MotionCode experiments | Classification & regression modes |
| `improved_motion_code.py` | Deeper architecture + regularization | Early stopping, LR sched, dropout |
| `optimized_experiments.py` | Hyperparameter search scaffold | Grid combinations over structural params |
| `ensemble_experiments.py` | Multiple neural variants + ensembling | LSTM, attention, enhanced models |
| `sklearn_optimization.py` | Systematic sklearn model search | RF, GB, SVM, LR, MLP + engineered features |
| `test_accuracy.py` | Fast feature-engineered RandomForest benchmark | Achieved 80.4% accuracy |
| `simple_test.py` | Minimal RandomForest sanity | Flatten vs diff augmentation |
| `quick_test.py` | Rapid multi-model comparison | For exploratory iteration |
| `advanced_boosting.py` | LightGBM/XGBoost + stacking + calibration | Did not exceed 80.4% |
| `train_qoi.py` | (If used) QOI regression specialized training | Predict asymmetry/speed |
| `create_real_scores.py` | Auxiliary metric generation (optional) | Clinical alignment |

---
## 6. Model Families
| Family | Implementation | Notes |
|--------|---------------|-------|
| MotionCode | `models/motion_code.py` | Custom feed-forward temporal abstraction |
| CrossFormer | `models/crossformer.py` | Cross-dimension attention transformer |
| iTransformer | `models/itransformer.py` | Inverted channel/time modeling |
| Mamba | `models/mamba.py` | State-space sequential modeling |
| TimesNet | `models/timesnet.py` | Basis periodic temporal modeling |
| Classical (RF, GB, SVM, LR, MLP) | `sklearn_optimization.py` | Tabular with feature engineering |
| Boosting (LGBM, XGBoost) | `advanced_boosting.py` | Gradient boosting ensemble |

---
## 7. Feature Engineering (Driving 80.4%)
Applied in `test_accuracy.py` & `sklearn_optimization.py`:
1. Flatten conditions → base feature vector (20 for gait)
2. Difference (dual − base)
3. Ratio (dual / base, safe divide)
4. Mean across conditions
5. Standard deviation across conditions
→ Concatenated dimensionality (gait): 60

Interpretation: Diff & ratio encode dual-task interference; mean/std stabilize representation; redundancy aids tree ensembles.

---
## 8. Optimization & Regularization Tactics
| Tactic | Applied In | Effect |
|--------|-----------|--------|
| Class weights | RF, LR, SVM, deep models | Balance PD recall |
| Dropout (0.3–0.5) | `improved_motion_code.py` | Reduce overfit |
| BatchNorm | `improved_motion_code.py` | Stabilize layer activations |
| Early stopping | `improved_motion_code.py` | Prevent late-epoch drift |
| LR scheduling | `improved_motion_code.py` | Adapt learning plateau |
| Hyperparameter grids | `optimized_experiments.py`, `sklearn_optimization.py` | Structured search |
| Ensembling (soft vote) | `ensemble_experiments.py` | Variance reduction |
| SMOTE (5-fold CV) | `advanced_boosting.py` | Class balance in limited data |
| Feature selection (RF importance mask) | `advanced_boosting.py` | Noise pruning |
| Calibration (Isotonic) | `advanced_boosting.py` | Probability refinement |

---
## 9. Results Timeline
| Stage | Method | Accuracy | Notes |
|-------|--------|----------|------|
| Baseline combined | Early run / mixed features | 54.9% | Low discriminative alignment |
| Domain-specific MotionCode | Gait only | 72.6% | Domain isolation gain |
| Initial classical + FE (expected) | Pre-optimization | ~0.74–0.78 (projected) | Before measurement |
| Engineered RandomForest (`test_accuracy.py`) | Gait FE | **80.4%** | Best accuracy (identical across RF variants) |
| Boosting / Stacking (`advanced_boosting.py`) | LGBM/XGB + stack | 76.5% (max) | Did not surpass RF |

Maximum confirmed accuracy: **80.4%** (JSON: `results_test_accuracy.json`).

---
## 10. Detailed Best Run (80.4%)
| Aspect | Value |
|--------|-------|
| Split | Stratified 70/30 (random_state=42) |
| Samples (test) | 51 |
| Model | RandomForestClassifier(n_estimators=100) |
| Parallel RF variant | n_estimators=300, depth=20, class_weight='balanced' (same accuracy) |
| Feature count | 60 engineered |
| PD Precision / Recall | 0.84 / 0.70 |
| SWEDD Precision / Recall | 0.78 / 0.89 |
| JSON Artifact | `parkinsons_project/results_test_accuracy.json` |

---
## 11. Regression (Cross-Modality) Findings
| Task | Source → Target | MAE | R² | Interpretation |
|------|-----------------|-----|----|---------------|
| Gait → Asymmetry | Gait predicts swing asymmetry | 13.83 | -2.14 | Mismatch / weak mapping |
| Swing → Speed | Swing predicts gait speed | 0.73 | -17.91 | Strong negative generalization (noise) |

Conclusion: Cross-modality regression ineffective; pursue within-modality QOIs next.

---
## 12. Reproduction Guide
```bash
# 1. Navigate
cd parkinsons_project

# 2. (Optional) Regenerate processed arrays
python preprocess.py
python prepare_separated_data.py

# 3. Baseline MotionCode
python run_new_experiments.py --task classification --data gait

# 4. Achieve 80.4% tabular benchmark
python test_accuracy.py

# 5. (Optional) Sklearn optimization suite
python sklearn_optimization.py --data gait

# 6. (Optional) Boosting / stacking (will NOT beat 80.4% in current form)
python advanced_boosting.py --data gait
```
Artifacts appear in project root or `parkinsons_project/` depending on script.

---
## 13. Saved Artifacts Summary
| File | Content |
|------|---------|
| `results_test_accuracy.json` | Best accuracy metadata (80.4%) |
| `results_boosting.json` | Boosting/stacking metrics & CV summary |
| `OPTIMIZATION_SUMMARY.md` | Strategy rationale overview |
| `docs/detailed_hack.md` | Initial deep-dive + incremental updates |
| `docs/full_detailed_hack.md` | This comprehensive document |

---
## 14. Environment & Dependencies
Core libs: `numpy`, `pandas`, `scikit-learn`, `torch` (for MotionCode), `lightgbm`, `xgboost`, `imbalanced-learn`, `einops`.

Recommendation: Freeze environment with `pip freeze > requirements_lock.txt` once experimentation stabilizes.

---
## 15. Limitations & Risks
| Category | Concern | Mitigation |
|----------|---------|------------|
| Sample Size | Only 167 subjects | Use cross-validation, report CI |
| Class Balance | Mild imbalance PD vs SWEDD | Class weights, SMOTE (careful) |
| Overfitting Risk | High-dimensional engineered features | Feature selection masks (importance) |
| External Validity | Single-source dataset | Seek external cohort replication |
| Cross-Modal Instability | Negative R² regression | Restrict to within-modality prediction |

---
## 16. Next High-Value Directions
1. Temporal reconstruction: Recover raw sequence windows (if accessible) → CNN/transformer time models.
2. Add frequency-domain gait harmonics / stride variability spectra.
3. Threshold tuning for PD recall (optimize sensitivity-specificity trade-off clinically).
4. Permutation importance & SHAP for interpretability of gait ratios/diffs.
5. Calibrated probability deployment (Brier score tracking).
6. Multi-task learning (classification + asymmetry regression jointly) if positive transfer emerges.

---
## 17. Executive Recap
A principled feature engineering pipeline plus a well-tuned yet *simple* RandomForest surpassed more complex boosting and deep ensembling approaches on this dataset, reaching **80.4%** accuracy. Additional architectural complexity yielded diminishing returns under current sample constraints; future gains depend on richer temporal or biomechanical feature sources rather than further tabular stacking.

---
*Document generated to consolidate the complete technical pathway from raw data to maximum achieved accuracy (80.4%).*
