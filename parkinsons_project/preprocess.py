# In file: preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def preprocess_data():
    """
    Loads raw Parkinson's data, cleans it, creates the pseudo-time-series
    format, and saves processed data for training scripts.
    This version is adapted to compare PD (1) vs. SWEDD (3).
    """
    print("--- Starting Data Preprocessing ---")
    
    # 1. Load the raw dataset
    try:
        df = pd.read_csv('data/PPMI_Gait_Data.csv', na_values=['', ' '])
        print("Raw dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'data/PPMI_Gait_Data.csv' not found. Please place your CSV file in the 'data' folder.")
        return

    # --- MODIFIED STEP ---
    # 2. Filter for binary classification (PD vs. SWEDD)
    df_filtered = df[df['COHORT'].isin([1, 3])].copy()
    
    # Validation Step: Check if we have more than one class for classification
    if df_filtered['COHORT'].nunique() < 2:
        print("\nCRITICAL ERROR: The dataset contains only one class after filtering for PD (1) and SWEDD (3).")
        print("A meaningful classification requires at least two classes.")
        print("Please check your 'PPMI_Gait_Data.csv' file.")
        return # Stop the script
        
    df_filtered['COHORT'] = df_filtered['COHORT'].astype(int)
    print(f"Filtered for PD (Cohort 1) vs SWEDD (Cohort 3). Using {len(df_filtered)} records.")

    # 3. Separate features (X) from target (y)
    drop_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'COHORT']
    features = [col for col in df_filtered.columns if col not in drop_cols]
    X = df_filtered[features].copy()
    y_raw = df_filtered['COHORT']

    # 4. Clean features
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    print("Features cleaned and missing values filled.")

    # 5. Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Target labels encoded. Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 6. Create the 3D pseudo-time-series array
    base_features = sorted([f for f in features if f.endswith('_U')])
    dualtask_features = sorted([f for f in features if f.endswith('_DT')])
    
    base_stems = [f.replace('_U', '') for f in base_features]
    dt_stems = [f.replace('_DT', '').replace('__', '_') for f in dualtask_features]
    common_stems = sorted(list(set(base_stems) & set(dt_stems)))
    print(f"Found {len(common_stems)} common features for the time series.")

    base_features_common = [f + '_U' for f in common_stems]
    dt_features_common = [(f + '__DT' if f == 'SP_' else f + '_DT') for f in common_stems]
    
    X_base = X[base_features_common].values
    X_dt = X[dt_features_common].values
    
    X_pseudo_ts = np.stack([X_base, X_dt], axis=1)
    print(f"Created 3D data with shape: {X_pseudo_ts.shape}")

    # 7. Save the processed data arrays
    np.save('data/X_processed.npy', X_pseudo_ts)
    np.save('data/y_processed.npy', y)
    print("\n--- Preprocessing Complete ---")
    print("Processed data saved to 'data/X_processed.npy' and 'data/y_processed.npy'.")

if __name__ == "__main__":
    preprocess_data()