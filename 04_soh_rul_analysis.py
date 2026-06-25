"""
BATTERY HEALTH PREDICTION SYSTEM
Step 4: SoH and RUL Analysis
Plots State of Health and Remaining Useful Life for all batteries.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


def plot_soh_rul():
    print("=" * 60)
    print("BATTERY SoH AND RUL ANALYSIS")
    print("=" * 60)

    df = pd.read_csv("data/features.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Battery State of Health (SoH) and RUL Analysis',
                 fontsize=15, fontweight='bold')

    colors = plt.cm.tab20(np.linspace(0, 1, df['battery_id'].nunique()))

    # ── Plot 1: SoH vs Cycle for all batteries ────────────────────────────────
    ax = axes[0, 0]
    for i, batt in enumerate(df['battery_id'].unique()):
        b = df[df['battery_id'] == batt].sort_values('cycle_idx')
        ax.plot(b['cycle_idx'], b['soh'] * 100, alpha=0.6,
                color=colors[i], linewidth=1.2)
    ax.axhline(80, color='red', linestyle='--', lw=2, label='EOL threshold (80%)')
    ax.set_xlabel('Cycle Number'); ax.set_ylabel('State of Health (%)')
    ax.set_title('SoH Degradation — All 34 Batteries')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(50, 105)

    # ── Plot 2: RUL distribution ──────────────────────────────────────────────
    ax = axes[0, 1]
    # RUL at the midpoint of each battery's life
    mid_rul = []
    for batt in df['battery_id'].unique():
        b = df[df['battery_id'] == batt]
        mid = b.iloc[len(b) // 2]['rul']
        mid_rul.append(mid)
    ax.hist(mid_rul, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Remaining Useful Life (cycles)')
    ax.set_ylabel('Number of Batteries')
    ax.set_title('RUL Distribution at Mid-Life')
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Degradation rate distribution ─────────────────────────────────
    ax = axes[1, 0]
    ax.hist(df['deg_rate'].dropna() * 1000, bins=50,
            color='coral', edgecolor='white', alpha=0.85)
    ax.axvline(0, color='red', lw=2, linestyle='--')
    ax.set_xlabel('Capacity Change per Cycle (mAh)')
    ax.set_ylabel('Frequency')
    ax.set_title('Degradation Rate Distribution')
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Initial vs Final capacity scatter ──────────────────────────────
    ax = axes[1, 1]
    init_caps, final_caps, total_cycles = [], [], []
    for batt in df['battery_id'].unique():
        b = df[df['battery_id'] == batt].sort_values('cycle_idx')
        if len(b) >= 10:
            init_caps.append(b['initial_cap'].iloc[0])
            final_caps.append(b['Capacity'].iloc[-1])
            total_cycles.append(b['cycle_idx'].iloc[-1])

    sc = ax.scatter(init_caps, final_caps, c=total_cycles, cmap='RdYlGn',
                    s=80, edgecolors='black', linewidths=0.5, zorder=3)
    ax.plot([1.0, 2.5], [1.0, 2.5], 'k--', alpha=0.3, label='No degradation')
    plt.colorbar(sc, ax=ax, label='Cycles lived')
    ax.set_xlabel('Initial Capacity (Ah)')
    ax.set_ylabel('Final Capacity (Ah)')
    ax.set_title('Initial vs Final Capacity (colored by longevity)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path("models").mkdir(exist_ok=True)
    plt.savefig('models/soh_rul_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: models/soh_rul_analysis.png")

    # Print summary stats
    print("\n── Battery Fleet Summary ─────────────────────────────────────────")
    print(f"  Total batteries        : {df['battery_id'].nunique()}")
    print(f"  Total discharge cycles : {len(df)}")
    print(f"  Avg initial capacity   : {df['initial_cap'].mean():.3f} Ah")
    print(f"  Avg SoH (dataset-wide) : {df['soh'].mean()*100:.1f}%")
    below_eol = (df['soh'] < 0.8).mean() * 100
    print(f"  Fraction below EOL     : {below_eol:.1f}%")


if __name__ == "__main__":
    plot_soh_rul()
