"""
BATTERY HEALTH PREDICTION SYSTEM
Step 1: Data Preparation
NASA Battery Dataset — 34 batteries, 2500+ discharge cycles

Run this first before training.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def parse_capacity(val):
    """Parse capacity values from NASA metadata format."""
    try:
        v = float(str(val).strip())
        return v if v > 0.5 else np.nan   # Filter out bad/zero readings
    except:
        return np.nan

def build_features(discharge_df):
    """
    Build physics-inspired features from discharge metadata.
    
    Features:
    - Cycle index and transforms (log, sqrt, squared) to capture nonlinear aging
    - Lagged capacities (previous cycles) — key for degradation trend
    - Degradation rate (dQ/dN) — the rate batteries are dying
    - Initial capacity and drop % — battery-specific normalization
    - Ambient temperature — Arrhenius aging factor
    """
    df = discharge_df.sort_values(['battery_id', 'cycle_idx']).copy()

    # Lag features
    for lag in [1, 2, 5]:
        df[f'cap_lag{lag}'] = df.groupby('battery_id')['Capacity'].shift(lag)

    # Degradation rate: how fast capacity is dropping
    df['deg_rate']  = df['Capacity'] - df['cap_lag1']
    df['deg_rate5'] = (df['Capacity'] - df['cap_lag5']) / 5

    # Cycle transforms (nonlinear aging models)
    df['log_cycle']      = np.log1p(df['cycle_idx'])
    df['cycle_squared']  = df['cycle_idx'] ** 2
    df['cycle_sqrt']     = np.sqrt(df['cycle_idx'])

    # Battery-normalized features
    df['initial_cap']  = df.groupby('battery_id')['Capacity'].transform('first')
    df['cap_drop_pct'] = (df['initial_cap'] - df['Capacity']) / df['initial_cap']

    # Target labels
    df['soh'] = df['Capacity'] / df['initial_cap']
    df['rul'] = df.groupby('battery_id')['cycle_idx'].transform(lambda x: x.max() - x)

    return df.dropna()


def main():
    print("=" * 60)
    print("BATTERY HEALTH PREDICTION — DATA PREPARATION")
    print("=" * 60)

    # Load NASA metadata (already has capacity per discharge cycle)
    meta_path = "metadata.csv"
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found. Place it in the same folder.")
        return

    df = pd.read_csv(meta_path)
    print(f"\nMetadata loaded: {len(df)} total records")
    print(f"Record types: {dict(df['type'].value_counts())}")

    # Filter to discharge cycles only — these have capacity measurements
    discharge = df[df['type'] == 'discharge'].copy()
    discharge['Capacity'] = discharge['Capacity'].apply(parse_capacity)
    discharge = discharge.dropna(subset=['Capacity']).reset_index(drop=True)

    # Add per-battery cycle index
    discharge['cycle_idx'] = discharge.groupby('battery_id').cumcount()

    # Parse ambient temperature
    discharge['ambient_temp'] = pd.to_numeric(discharge['ambient_temperature'], errors='coerce')

    print(f"\nClean discharge cycles: {len(discharge)}")
    print(f"Batteries: {discharge['battery_id'].nunique()}")
    print(f"Capacity range: {discharge['Capacity'].min():.3f} – {discharge['Capacity'].max():.3f} Ah")

    # Build features
    features_df = build_features(discharge)

    print(f"\nFeature matrix: {features_df.shape}")
    print(f"Features: {[c for c in features_df.columns if c not in ['battery_id','Capacity','soh','rul']]}")

    # Save
    Path("data").mkdir(exist_ok=True)
    features_df.to_csv("data/features.csv", index=False)
    discharge.to_csv("data/discharge_clean.csv", index=False)
    print(f"\n✓ Saved: data/features.csv")
    print(f"✓ Saved: data/discharge_clean.csv")
    print("\nReady! Run 02_train_model.py next.")


if __name__ == "__main__":
    main()
