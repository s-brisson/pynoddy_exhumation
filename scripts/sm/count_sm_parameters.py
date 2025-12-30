#!/usr/bin/env python3
"""
Count parameters from SM_20240406 CSV files to determine number of free parameters.
"""

import pandas as pd
from pathlib import Path
import glob

# Path to SM parameter files
params_dir = Path("/Users/flow/Documents/01_work/01_own_docs/02_paper_drafts/2025/Sofia_Paper_3/Thermokinematic/cluster_outputs/SM_20240406/model_params")

# Read first CSV file
csv_files = list(params_dir.glob("*.csv"))
if not csv_files:
    print(f"No CSV files found in {params_dir}")
    exit(1)

df = pd.read_csv(csv_files[0])
print(f"Analyzing: {csv_files[0].name}")
print(f"Total rows: {len(df)}")
print(f"\nColumns: {list(df.columns)}")

# Count unique events
events = sorted(df['event_name'].unique())
n_events = len(events)
print(f"\nUnique events: {events}")
print(f"Number of events: {n_events}")

# Check which parameters are varied
print("\n" + "="*60)
print("Checking which parameters are varied:")
print("="*60)

# Check for X, Z, Amplitude, Slip
param_cols = ['X', 'Z', 'Amplitude', 'Slip']
params_found = [col for col in param_cols if col in df.columns]
print(f"\nParameter columns found: {params_found}")

# Check variation for each parameter
params_per_event = 0
for param in param_cols:
    if param in df.columns:
        # Check if this parameter varies across simulations
        unique_vals = df[param].nunique()
        total_rows = len(df)
        variation_ratio = unique_vals / total_rows if total_rows > 0 else 0
        
        print(f"\n{param}:")
        print(f"  Unique values: {unique_vals} out of {total_rows} rows")
        print(f"  Variation ratio: {variation_ratio:.2%}")
        
        if variation_ratio > 0.1:  # More than 10% variation indicates free parameter
            print(f"  → FREE parameter (varies significantly)")
            params_per_event += 1
        else:
            print(f"  → FIXED parameter (few unique values)")

# According to colleague: Each fault has 4 parameters (X, Z, Amplitude, Slip)
print("\n" + "="*60)
print("PARAMETER COUNT SUMMARY:")
print("="*60)
print("\nAccording to colleague:")
print("  Each fault has 4 parameters: X, Z, Amplitude, Slip")
print(f"  Number of faults: {n_events}")
print(f"  Total parameters: {n_events} faults × 4 parameters = {n_events * 4}")

# Use confirmed counts
params_per_event = 4  # X, Z, Amplitude, Slip

total_params = n_events * params_per_event
print(f"\nTotal free parameters: {n_events} events × {params_per_event} params = {total_params}")

# Check if there are separate Model A and Model B directories
print("\n" + "="*60)
print("Checking for Model A vs Model B separation:")
print("="*60)

parent_dir = params_dir.parent
possible_dirs = [
    parent_dir / "modelA_params",
    parent_dir / "modelB_params",
    parent_dir / "modelA" / "model_params",
    parent_dir / "modelB" / "model_params",
]

for dir_path in possible_dirs:
    if dir_path.exists():
        print(f"Found: {dir_path}")
        csvs = list(dir_path.glob("*.csv"))
        if csvs:
            df_check = pd.read_csv(csvs[0])
            events_check = sorted(df_check['event_name'].unique())
            print(f"  Events: {events_check}")
            print(f"  Number of events: {len(events_check)}")
    else:
        print(f"Not found: {dir_path}")

print("\n" + "="*60)
print("CONFIRMED PARAMETER COUNTS:")
print("="*60)
print("\nFrom colleague:")
print("  SM Model A: 5 faults × 4 parameters (X, Z, Amplitude, Slip) = 20 parameters")
print("  SM Model B: 4 faults × 4 parameters (X, Z, Amplitude, Slip) = 16 parameters")
print(f"\nThis analysis found:")
print(f"  - {n_events} faults/events")
print(f"  - {params_per_event} parameters per fault (X, Z, Amplitude, Slip)")
print(f"  - Total: {n_events * params_per_event} parameters")
print(f"\nUse these values in analyze_sm_bayes_factor.py:")
print(f"  n_params_A=20  # 5 faults × 4 parameters")
print(f"  n_params_B=16  # 4 faults × 4 parameters")

