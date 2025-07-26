# Parkinson's Disease Analysis with Time-Series Models

## Overview
This project aims to analyze Parkinson's Disease (PD) using gait and arm swing data from the PPMI dataset. It contains a complete pipeline to preprocess the data and evaluate five different advanced time-series models on two key tasks:

Classification: Distinguishing between patients with Parkinson's Disease (PD) and those with Scans Without Evidence of Dopaminergic Deficit (SWEDD).

Regression: Predicting a clinical Quantity of Interest (QOI), specifically the MDS-UPDRS Part III score, to estimate disease severity.

## Project Structure
The project is organized into the following directories and files:

parkinsons_project/
├── data/
│   ├── PPMI_Gait_Data.csv              # Raw gait/arm swing data
│   └── MDS_UPDRS_Part_III.csv        # Clinical severity scores
├── models/
│   ├── __init__.py
│   ├── motion_code.py
│   ├── timesnet.py
│   ├── itransformer.py
│   ├── crossformer.py
│   └── mamba.py
├── preprocess.py                       # Script to process gait data for classification
├── create_real_scores.py               # Script to process clinical data for regression
├── main.py                             # Main script to run classification experiments
├── train_qoi.py                        # Script to run the QOI regression experiment
└── requirements.txt                    # Required Python packages

## Setup

To set up the project, follow these steps:

Create a Virtual Environment: It is highly recommended to use a virtual environment to manage dependencies.

Install Packages: Install all required packages using the requirements.txt file.

pip install -r requirements.txt

## Data

You will need two data files, which should be placed inside the data/ folder:

PPMI_Gait_Data.csv: The primary dataset containing the gait and arm swing measurements.

MDS_UPDRS_Part_III.csv: The clinical data file containing the NP3TOT column for disease severity scores.

Workflow: How to Run the Experiments
The analysis is divided into four main steps. Run them in the following order.

# Step 1: Preprocess the Gait Data

This script cleans the PPMI_Gait_Data.csv file and creates the processed X_processed.npy and y_processed.npy files needed for the classification task. You only need to run this once.

python preprocess.py

# Step 2: Run the Classification Experiments

Use the main.py script to train and evaluate the five models on the PD vs. SWEDD classification task. Run each model with the --model argument.

# Run Motion Code
python main.py --model motion_code

# Run TimesNet
python main.py --model timesnet

# Run iTransformer
python main.py --model itransformer

# Run CrossFormer
python main.py --model crossformer

# Run Mamba
python main.py --model mamba

# Step 3: Prepare the Severity Score Data

This script merges the clinical data from MDS_UPDRS_Part_III.csv with the gait data and creates the target file (y_severity_scores.npy) for the regression task. You only need to run this once.

python create_real_scores.py

# Step 4: Run the QOI Regression Experiment

This script trains and evaluates a model to predict the disease severity score.

python train_qoi.py

Models Included
This project contains self-contained, error-corrected implementations for the following models:

Motion Code
TimesNet
iTransformer
CrossFormer
Mamba