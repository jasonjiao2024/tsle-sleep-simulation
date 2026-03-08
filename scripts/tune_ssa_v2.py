"""SSA parameter sweep v2: test wobble magnitude and slow recovery."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from analysis.thalamocortical_model import ThalamocorticalEnsemble, compute_swa
from analysis.protocol_comparison import EPOCH_SEC

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
subjects = {}
for csv_path in sorted(DATA_DIR.glob('*_processed.csv'))[:5]:
    sid = csv_path.stem.replace('_processed', '')
    df = pd.read_csv(csv_path)
    if all(c in df.columns for c in ['delta_power','theta_power','alpha_power','beta_power']):
        subjects[sid] = df
print(f"Loaded {len(subjects)} subjects")

SESSION_DUR = 1800
N_EPOCHS = int(SESSION_DUR // EPOCH_SEC)

# Test different approaches
tests = [
    # (wobble_freq, allow_slow_recovery, slow_recovery_fraction, label)
    (1.0, False, 0.0, "1Hz wobble, no slow recov (current)"),
    (0.5, False, 0.0, "0.5Hz wobble (bigger shift)"),
    (3.0, False, 0.0, "3Hz wobble (smaller shift, closer)"),
    (1.0, True,  0.3, "1Hz wobble + 30% slow recov"),
    (1.0, True,  0.5, "1Hz wobble + 50% slow recov"),
    (0.5, True,  0.3, "0.5Hz wobble + 30% slow recov"),
    (0.5, True,  0.5, "0.5Hz wobble + 50% slow recov"),
]

for wobble_freq, slow_recov, slow_frac, label in tests:
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
            ensemble.initialize_from_baseline(baseline, non_responder_fraction=0.3)
            ensemble.z = 0.1 * ensemble.rng.standard_normal(64) + 0.1j * ensemble.rng.standard_normal(64)
            ensemble.T = 0.0; ensemble.H = 0.0
            ensemble.A_fast = 0.0; ensemble.A_slow = 0.0
            ensemble._last_forcing_freq = -1.0
            ensemble._baseline_swa = None
            ensemble._so_buffer[:] = 0.0
            ensemble._so_buf_idx = 0
            ensemble._so_buf_filled = False
            ensemble._so_sample_counter = 0

            ensemble.run_epoch(EPOCH_SEC, 1.0, 0.0)
            bp0 = ensemble.compute_band_powers()
            baseline_swa = compute_swa(bp0)
            ensemble._baseline_swa = baseline_swa

            if cond == 'fixed_delta':
                for _ in range(N_EPOCHS):
                    ensemble.run_epoch(EPOCH_SEC, 2.0, 0.3)
            else:
                for ep in range(N_EPOCHS):
                    if ep > 0 and ep % 10 == 0:
                        # Before wobble, manually apply slow recovery if enabled
                        if slow_recov:
                            ensemble.A_slow *= (1.0 - slow_frac)
                        ensemble.run_epoch(EPOCH_SEC, wobble_freq, 0.3)
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
    print(f"  Fixed: {fixed_mean:+.1f}%  Reset: {reset_mean:+.1f}%  Diff: {reset_mean-fixed_mean:+.1f}%  d={d:+.3f}")

print("\nDone.")
