import pandas as pd
import numpy as np

def prepare_data():
    """
    Splits the processed feature set into Gait-only and Arm Swing-only datasets.
    Also creates target files for specific QOI prediction tasks.
    """
    print("--- Preparing Separated Datasets and QOI Targets ---")
    
    # Part 1: Split the Feature Sets
    
    gait_df = pd.read_csv('data/PPMI_Gait_Data.csv', na_values=['', ' '])
    base_features = sorted([f for f in gait_df.columns if f.endswith('_U')])
    dt_features = sorted([f for f in gait_df.columns if f.endswith('_DT')])
    base_stems = [f.replace('_U', '') for f in base_features]
    dt_stems = [f.replace('_DT', '').replace('__', '_') for f in dt_features]
    common_stems = sorted(list(set(base_stems) & set(dt_stems)))
    
    arm_swing_features = [
        'ASA', 'ASYM_IND', 'LA_AMP', 'LA_STD', 
        'L_JERK', 'RA_AMP', 'RA_STD', 'R_JERK'
    ]
    
    gait_features = [
        'CAD', 'JERK_T', 'SP_', 'STEP_REG', 'STEP_SYM', 
        'STR_CV', 'STR_T', 'SYM', 'T_AMP', 'TRA'
    ]

    swing_indices = [i for i, stem in enumerate(common_stems) if any(f in stem for f in arm_swing_features)]
    gait_indices = [i for i, stem in enumerate(common_stems) if any(f in stem for f in gait_features)]
    
    print(f"Found {len(swing_indices)} Arm Swing features and {len(gait_indices)} Gait features.")

    X_processed = np.load('data/X_processed.npy') 
    
    X_swing = X_processed[:, :, swing_indices]
    X_gait = X_processed[:, :, gait_indices]
    
    np.save('data/X_swing.npy', X_swing)
    np.save('data/X_gait.npy', X_gait)
    
    print(f"Saved Arm Swing data with shape: {X_swing.shape}")
    print(f"Saved Gait data with shape: {X_gait.shape}")

    # Part 2: Create the QOI Target Files 
    
    df_filtered = gait_df[gait_df['COHORT'].isin([1, 3])].dropna(subset=['PATNO', 'INFODT']).copy()
    
    # QOI 1: Arm Swing Asymmetry (ASA_U) - a Level 2 Kinematic Feature
    y_qoi_asymmetry = pd.to_numeric(df_filtered['ASA_U'], errors='coerce').fillna(df_filtered['ASA_U'].median()).values
    np.save('data/y_qoi_asymmetry.npy', y_qoi_asymmetry)
    print("Saved QOI target for Arm Swing Asymmetry.")

    # QOI 2: Walking Speed (SP_U) - another Level 2 Kinematic Feature
    y_qoi_speed = pd.to_numeric(df_filtered['SP_U'], errors='coerce').fillna(df_filtered['SP_U'].median()).values
    np.save('data/y_qoi_speed.npy', y_qoi_speed)
    print("Saved QOI target for Walking Speed.")
    
    print("\nData Preparation Complete")

if __name__ == "__main__":
    prepare_data()