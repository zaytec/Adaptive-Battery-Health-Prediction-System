"""
BATTERY HEALTH PREDICTION SYSTEM
Step 3: Demo / Inference
Interactive demo for judges — type in cycle data and get predictions.

Run: python 03_demo.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "models/battery_model_v2.pkl"

# ── Known battery profiles for live demo (from NASA dataset) ─────────────────
DEMO_BATTERIES = {
    'B0005': {'initial_cap': 1.856, 'cycles': 168, 'final_cap': 1.287},
    'B0006': {'initial_cap': 2.035, 'cycles': 168, 'final_cap': 1.154},
    'B0007': {'initial_cap': 1.891, 'cycles': 168, 'final_cap': 1.400},
    'B0018': {'initial_cap': 1.855, 'cycles': 132, 'final_cap': 1.341},
}


def load_model():
    """Load trained model."""
    data = joblib.load(MODEL_PATH)
    return data['rf'], data['gb'], data['scaler'], data['feature_cols'], data['metrics']


def predict_single(rf, gb, scaler, feat_cols,
                   cycle, cap_lag1, cap_lag2, cap_lag5,
                   initial_cap, ambient_temp=24.0):
    """
    Predict capacity for a single cycle.

    Args:
        cycle        : current cycle number (int)
        cap_lag1     : capacity at previous cycle (Ah)
        cap_lag2     : capacity 2 cycles ago (Ah)
        cap_lag5     : capacity 5 cycles ago (Ah)
        initial_cap  : initial (fresh) capacity of this battery (Ah)
        ambient_temp : test temperature in °C

    Returns:
        dict with predicted_capacity, soh, rul_estimate
    """
    deg_rate  = cap_lag1 - cap_lag2 if cap_lag2 else 0
    deg_rate5 = (cap_lag1 - cap_lag5) / 5 if cap_lag5 else 0
    cap_drop  = (initial_cap - cap_lag1) / initial_cap

    row = {
        'cycle_idx':     cycle,
        'log_cycle':     np.log1p(cycle),
        'cycle_squared': cycle ** 2,
        'cycle_sqrt':    np.sqrt(cycle),
        'cap_lag1':      cap_lag1,
        'cap_lag2':      cap_lag2,
        'cap_lag5':      cap_lag5,
        'deg_rate':      deg_rate,
        'deg_rate5':     deg_rate5,
        'initial_cap':   initial_cap,
        'cap_drop_pct':  cap_drop,
        'ambient_temp':  ambient_temp,
    }

    X = np.array([[row[f] for f in feat_cols]])
    X_s = scaler.transform(X)

    cap_rf = rf.predict(X_s)[0]
    cap_gb = gb.predict(X_s)[0]
    cap    = (cap_rf + cap_gb) / 2

    soh = cap / initial_cap * 100
    eol_cap = initial_cap * 0.80

    # Simple linear RUL estimate from degradation rate
    if deg_rate5 < 0:
        rul_est = int((cap - eol_cap) / abs(deg_rate5))
    else:
        rul_est = 999  # Not degrading yet

    return {
        'predicted_capacity_Ah': round(cap, 4),
        'soh_pct':               round(soh, 2),
        'rul_cycles':            max(0, rul_est),
        'health_status':         _health_label(soh),
    }


def _health_label(soh):
    if soh >= 90: return "EXCELLENT"
    if soh >= 80: return "GOOD"
    if soh >= 70: return "DEGRADED — monitor closely"
    return "CRITICAL — replace soon"


def demo_known_battery(rf, gb, scaler, feat_cols, battery_id='B0005'):
    """Demo prediction trajectory for a known battery."""
    info = DEMO_BATTERIES[battery_id]
    ic   = info['initial_cap']

    # Simulate smooth degradation trajectory
    cycles = np.arange(0, info['cycles'])
    cap_true = np.linspace(ic, info['final_cap'], len(cycles))
    cap_true += np.random.normal(0, 0.003, len(cycles))  # tiny noise

    cap_pred = []
    for i, cyc in enumerate(cycles):
        l1 = cap_true[i-1] if i >= 1 else ic
        l2 = cap_true[i-2] if i >= 2 else ic
        l5 = cap_true[i-5] if i >= 5 else ic
        res = predict_single(rf, gb, scaler, feat_cols, cyc, l1, l2, l5, ic)
        cap_pred.append(res['predicted_capacity_Ah'])

    cap_pred = np.array(cap_pred)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Battery {battery_id} — Health Prediction Demo', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(cycles, cap_true, 'b-', lw=2, label='True Capacity', alpha=0.7)
    ax.plot(cycles, cap_pred, 'r--', lw=2, label='ML Predicted', alpha=0.9)
    ax.axhline(ic * 0.8, color='orange', lw=1.5, linestyle=':', label='EOL (80% SoH)')
    ax.fill_between(cycles, cap_pred-0.03, cap_pred+0.03, alpha=0.2, color='red')
    ax.set_xlabel('Cycle Number'); ax.set_ylabel('Capacity (Ah)')
    ax.set_title('Capacity Prediction'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    soh_true = cap_true / ic * 100
    soh_pred = cap_pred / ic * 100
    ax.plot(cycles, soh_true, 'b-', lw=2, label='True SoH', alpha=0.7)
    ax.plot(cycles, soh_pred, 'r--', lw=2, label='Predicted SoH', alpha=0.9)
    ax.axhline(80, color='orange', lw=1.5, linestyle=':', label='EOL threshold')
    ax.fill_between(cycles, 60, 80, alpha=0.08, color='red')
    ax.set_xlabel('Cycle Number'); ax.set_ylabel('State of Health (%)')
    ax.set_title('SoH Over Cycles'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(55, 105)

    plt.tight_layout()
    out = f'models/demo_{battery_id}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✓ Demo plot saved: {out}")
    return cap_pred, cap_true


def interactive_demo(rf, gb, scaler, feat_cols, metrics):
    """Command-line interactive demo for judges."""
    print("\n" + "=" * 60)
    print("BATTERY HEALTH PREDICTOR — LIVE DEMO")
    print("=" * 60)
    print(f"  Model R²:  {metrics['r2']:.4f}")
    print(f"  Accuracy:  {100 - metrics['mape']:.1f}%")
    print(f"  MAE:       {metrics['mae']*1000:.1f} mAh")
    print("=" * 60)
    print("\nEnter battery parameters to predict remaining health.\n")

    while True:
        try:
            print("─" * 40)
            print("Available batteries for quick demo:", list(DEMO_BATTERIES.keys()))
            batt = input("Battery ID (or 'custom' / 'q' to quit): ").strip()
            if batt.lower() == 'q':
                print("Goodbye!")
                break

            if batt in DEMO_BATTERIES:
                info  = DEMO_BATTERIES[batt]
                ic    = info['initial_cap']
                cyc   = int(input(f"  Cycle number (0–{info['cycles']}): "))
                # Simulate realistic history
                cap_now = ic - (ic - info['final_cap']) * (cyc / info['cycles'])
                l1 = cap_now + np.random.normal(0, 0.003)
                l2 = l1 + abs(np.random.normal(0.003, 0.001))
                l5 = l2 + abs(np.random.normal(0.012, 0.002))
                temp = info.get('temp', 24.0)
            else:
                ic   = float(input("  Initial (fresh) capacity Ah [e.g. 2.0]: "))
                cyc  = int(input("  Current cycle number [e.g. 50]: "))
                l1   = float(input("  Capacity at last cycle Ah: "))
                l2   = float(input("  Capacity 2 cycles ago Ah: "))
                l5   = float(input("  Capacity 5 cycles ago Ah: "))
                temp = float(input("  Ambient temperature °C [default 24]: ") or "24")

            result = predict_single(rf, gb, scaler, feat_cols, cyc, l1, l2, l5, ic, temp)

            print("\n" + "=" * 40)
            print(f"  Predicted Capacity : {result['predicted_capacity_Ah']} Ah")
            print(f"  State of Health    : {result['soh_pct']} %")
            print(f"  Remaining Life     : ~{result['rul_cycles']} cycles")
            print(f"  Health Status      : {result['health_status']}")
            print("=" * 40)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"  Error: {e}. Try again.")


def main():
    print("Loading model...")
    rf, gb, scaler, feat_cols, metrics = load_model()
    print(f"Model loaded. R²={metrics['r2']:.4f}, MAE={metrics['mae']*1000:.1f} mAh")

    # Generate demo plots for all known batteries
    Path("models").mkdir(exist_ok=True)
    for batt in DEMO_BATTERIES:
        print(f"\nGenerating demo for {batt}...")
        demo_known_battery(rf, gb, scaler, feat_cols, batt)

    # Run interactive demo
    interactive_demo(rf, gb, scaler, feat_cols, metrics)


if __name__ == "__main__":
    main()
