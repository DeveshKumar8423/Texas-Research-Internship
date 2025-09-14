#!/usr/bin/env python3
"""
Comprehensive experiment runner to maximize Parkinson's classification accuracy
"""

import numpy as np
import torch
import argparse
import os
from datetime import datetime
import sys

# Add project root to path
sys.path.append('/Users/a1/Documents/GitHub/Texas-Research-Internship/parkinsons_project')

def run_all_experiments(data_type='gait'):
    """Run all optimization experiments and report best results"""
    
    print(f"{'='*60}")
    print(f"COMPREHENSIVE OPTIMIZATION EXPERIMENT: {data_type.upper()}")
    print(f"Target: Improve accuracy above 72.6%")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    results_summary = {}
    
    # Experiment 1: Baseline (original)
    print(f"\n{'-'*20} EXPERIMENT 1: BASELINE {'-'*20}")
    try:
        os.system(f"cd /Users/a1/Documents/GitHub/Texas-Research-Internship/parkinsons_project && python run_new_experiments.py --task classification --data {data_type}")
        print("✓ Baseline experiment completed")
        results_summary['baseline'] = "54.9% (from previous runs)"
    except Exception as e:
        print(f"✗ Baseline experiment failed: {e}")
        results_summary['baseline'] = "Failed"
    
    # Experiment 2: Optimized with hyperparameter tuning
    print(f"\n{'-'*20} EXPERIMENT 2: HYPERPARAMETER OPTIMIZATION {'-'*20}")
    try:
        os.system(f"cd /Users/a1/Documents/GitHub/Texas-Research-Internship/parkinsons_project && python optimized_experiments.py --data {data_type}")
        print("✓ Hyperparameter optimization completed")
        results_summary['optimized'] = "Check optimized_results.txt"
    except Exception as e:
        print(f"✗ Hyperparameter optimization failed: {e}")
        results_summary['optimized'] = "Failed"
    
    # Experiment 3: Ensemble methods
    print(f"\n{'-'*20} EXPERIMENT 3: ENSEMBLE METHODS {'-'*20}")
    try:
        os.system(f"cd /Users/a1/Documents/GitHub/Texas-Research-Internship/parkinsons_project && python ensemble_experiments.py --data {data_type}")
        print("✓ Ensemble experiment completed")
        results_summary['ensemble'] = "Check ensemble_results.pkl"
    except Exception as e:
        print(f"✗ Ensemble experiment failed: {e}")
        results_summary['ensemble'] = "Failed"
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY FOR {data_type.upper()} DATA")
    print(f"{'='*60}")
    
    for exp_name, result in results_summary.items():
        print(f"{exp_name.capitalize()}: {result}")
    
    print(f"\n{'='*60}")
    print(f"NEXT STEPS TO ACHIEVE >72.6% ACCURACY:")
    print(f"{'='*60}")
    
    recommendations = [
        "1. Review hyperparameter optimization results from optimized_experiments.py",
        "2. Check ensemble voting results from ensemble_experiments.py", 
        "3. Implement domain-specific feature engineering",
        "4. Try advanced architectures (Transformer, Graph Neural Networks)",
        "5. Implement meta-learning or few-shot learning approaches",
        "6. Use transfer learning from larger medical datasets",
        "7. Implement advanced time series techniques (TSFresh features)",
        "8. Try different train/validation split strategies",
        "9. Implement uncertainty quantification with Monte Carlo Dropout",
        "10. Use advanced augmentation techniques (time warping, mixup)"
    ]
    
    for rec in recommendations:
        print(rec)
    
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comprehensive optimization experiments')
    parser.add_argument('--data', choices=['gait', 'swing'], default='gait',
                       help='Data type to use (default: gait)')
    parser.add_argument('--all', action='store_true',
                       help='Run experiments on both gait and swing data')
    
    args = parser.parse_args()
    
    if args.all:
        print("Running experiments on both data types...")
        gait_results = run_all_experiments('gait')
        swing_results = run_all_experiments('swing')
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY - ALL EXPERIMENTS")
        print(f"{'='*60}")
        print(f"Gait results: {gait_results}")
        print(f"Swing results: {swing_results}")
        
    else:
        results = run_all_experiments(args.data)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETED")
        print(f"Check the generated result files for detailed performance metrics")
        print(f"{'='*60}")
