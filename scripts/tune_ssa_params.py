"""Quick SSA parameter sweep to find values that strengthen SSA-reset advantage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from analysis.thalamocortical_model import ThalamocorticalEnsemble, compute_swa, compute_swa_enhancement
from analysis.protocol_comparison import EPOCH_SEC

# Load first 5 subjects
DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
subjects = {}
for csv_path in sorted(DATA_DIR.glob('*_processed.csv'))[:5]:
    sid = csv_path.stem.replace('_processed', '')
    df = pd.read_csv(csv_path)
    if all(c in df.columns for c in ['delta_power','theta_power','alpha_power','beta_power']):
        subjects[sid] = df

print(f"Loaded {len(subjects)} subjects")

# Parameter combinations to test
param_sets = [
    # (eta_fast, eta_slow, f_scale, label)
    (0.4, 0.3, 2.0, "current (0.4/0.3/2.0)"),
    (0.5, 0.2, 2.0, "shift fast (0.5/0.2/2.0)"),
    (0.5, 0.15, 2.0, "more fast (0.5/0.15/2.0)"),
    (0.5, 0.15, 4.0, "more fast+scale (0.5/0.15/4.0)"),
    (0.55, 0.1, 3.0, "strong fast (0.55/0.1/3.0)"),
    (0.5, 0.2, 3.0, "balanced (0.5/0.2/3.0)"),
]

SESSION_DUR = 1800  # 30 min for speed
N_EPOCHS = int(SESSION_DUR // EPOCH_SEC)

results = []

for eta_fast, eta_slow, f_scale, label in param_sets:
    swa_fixed = []
    swa_reset = []

    for sid, df in subjects.items():
        seed_base = hash(sid) % (2**31)
        n = min(10, len(df))
        baseline = df.iloc[:n][['delta_power','theta_power','alpha_power','beta_power']].mean().to_dict()

        for cond in ['fixed_delta', 'ssa_reset']:
            ensemble = ThalamocorticalEnsemble(
                n_oscillators=64, coupling_strength=2.0, noise_sigma=0.15,
                dt=0.005, seed=seed_base + (0 if cond == 'fixed_delta' else 1),
                tau_T=10.0, alpha_TC=5.0, gamma=0.5, kappa=3.0,
                T_half=0.3, delta_lambda=1.5, beta_ext=0.05, lambda_base=-0.3,
            )
            # Set SSA params
            ensemble.eta_fast = eta_fast
            ensemble.eta_slow = eta_slow
            ensemble.f_scale = f_scale

            ensemble.initialize_from_baseline(baseline, non_responder_fraction=0.3)

            # Reset state
            ensemble.z = 0.1 * ensemble.rng.standard_normal(64) + 0.1j * ensemble.rng.standard_normal(64)
            ensemble.T = 0.0; ensemble.H = 0.0
            ensemble.A_fast = 0.0; ensemble.A_slow = 0.0
            ensemble._last_forcing_freq = -1.0
            ensemble._baseline_swa = None
            ensemble._so_buffer[:] = 0.0
            ensemble._so_buf_idx = 0
            ensemble._so_buf_filled = False
            ensemble._so_sample_counter = 0

            # Baseline
            ensemble.run_epoch(EPOCH_SEC, 1.0, 0.0)
            bp0 = ensemble.compute_band_powers()
            baseline_swa = compute_swa(bp0)
            ensemble._baseline_swa = baseline_swa

            # Run session
            if cond == 'fixed_delta':
                for _ in range(N_EPOCHS):
                    ensemble.run_epoch(EPOCH_SEC, 2.0, 0.3)
            else:
                # SSA-reset: wobble every 5 min (10 epochs)
                for ep in range(N_EPOCHS):
                    if ep > 0 and ep % 10 == 0:
                        ensemble.run_epoch(EPOCH_SEC, 1.0, 0.3)  # wobble
                    else:
                        ensemble.run_epoch(EPOCH_SEC, 2.0, 0.3)

            bp_end = ensemble.compute_band_powers()
            swa_end = compute_swa(bp_end)
            swa_enh = ((swa_end - baseline_swa) / max(baseline_swa, 1e-6)) * 100.0

            if cond == 'fixed_delta':
                swa_fixed.append(swa_enh)
            else:
                swa_reset.append(swa_enh)

    fixed_mean = np.mean(swa_fixed)
    reset_mean = np.mean(swa_reset)
    diff = np.array(swa_reset) - np.array(swa_fixed)
    d = np.mean(diff) / max(np.std(diff, ddof=1), 1e-6) if len(diff) > 1 else 0

    print(f"\n{label}:")
    print(f"  Fixed delta SWA enh: {fixed_mean:+.1f}%")
    print(f"  SSA-reset SWA enh:   {reset_mean:+.1f}%")
    print(f"  Difference:          {reset_mean - fixed_mean:+.1f}%")
    print(f"  Cohen's d:           {d:+.3f}")
    print(f"  Max total adapt:     {eta_fast + eta_slow:.0%}")

    results.append({
        'label': label, 'eta_fast': eta_fast, 'eta_slow': eta_slow,
        'f_scale': f_scale, 'fixed_mean': fixed_mean, 'reset_mean': reset_mean,
        'diff': reset_mean - fixed_mean, 'cohens_d': d,
    })

print("\n\n=== SUMMARY ===")
print(f"{'Label':40s} {'d':>8s} {'diff':>8s} {'max_adapt':>10s}")
print("-" * 70)
for r in results:
    print(f"{r['label']:40s} {r['cohens_d']:+8.3f} {r['diff']:+8.1f}% {r['eta_fast']+r['eta_slow']:8.0%}")
