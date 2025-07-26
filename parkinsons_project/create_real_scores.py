# In file: create_real_scores.py
import pandas as pd
import numpy as np

def create_real_qoi_target():
    """
    Loads the gait and clinical data, handles potential duplicates,
    and saves the real severity scores as the target file.
    """
    print("--- Creating Real Severity Score Target File ---")

    updrs_filename = 'MDS_UPDRS_Part_III.csv'

    try:
        gait_df = pd.read_csv('data/PPMI_Gait_Data.csv', na_values=['', ' '])
        print("Original gait data loaded.")
        
        updrs_df = pd.read_csv(f'data/{updrs_filename}')
        print(f"Clinical data '{updrs_filename}' loaded.")

    except FileNotFoundError as e:
        print(f"Error: Could not find a data file: {e.filename}")
        return

    filtered_gait_df = gait_df[gait_df['COHORT'].isin([1, 3])].dropna(subset=['PATNO', 'INFODT']).copy()

    if 'NP3TOT' not in updrs_df.columns:
        print("Error: The clinical data file must contain the 'NP3TOT' column.")
        return
        
    updrs_scores = updrs_df[['PATNO', 'INFODT', 'NP3TOT']].copy()

    # --- NEW FIX ---
    # Remove any duplicate entries for the same patient on the same day
    updrs_scores.drop_duplicates(subset=['PATNO', 'INFODT'], inplace=True)
    print("Handled potential duplicate entries in the clinical data.")

    # Merge the two dataframes
    final_df = pd.merge(filtered_gait_df, updrs_scores, on=['PATNO', 'INFODT'], how='left')

    y_severity_scores = final_df['NP3TOT'].values

    if np.isnan(y_severity_scores).any():
        print("Warning: Some gait samples did not have a matching severity score and were filled with the median.")
        y_severity_scores = pd.Series(y_severity_scores).fillna(pd.Series(y_severity_scores).median()).values

    np.save('data/y_severity_scores.npy', y_severity_scores)

    print(f"\nSuccessfully processed and saved {len(y_severity_scores)} real severity scores.")
    print("File saved to 'data/y_severity_scores.npy'.")

if __name__ == "__main__":
    create_real_qoi_target()