"""
Calibration sweep: quick validation of model parameters before full 8-hour run.

Tests 5 subjects × 4 conditions × parameter grid (forcing_strength × delta_lambda).
30-min sessions → ~15-20 min total runtime.

Target ranges:
  - no_stim:          -2% to +3% SWA enhancement
  - sham:              0% to 5%
  - fixed_delta:      10% to 22%
  - ssa_reset_delta:  15% to 32%

Usage:
    python scripts/run_calibration_sweep.py
    python scripts/run_calibration_sweep.py --forcing 0.01 0.02 0.03 --delta-lambda 0.15 0.25 0.35
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.thalamocortical_model import ThalamocorticalEnsemble, compute_sdr, compute_swa, compute_swa_enhancement
from analysis.protocol_comparison import EPOCH_SEC
from analysis.redesigned_protocols import define_redesigned_protocols

N_SUBJECTS = 5
SESSION_DURATION_SEC = 30 * 60  # 30 min
N_OSCILLATORS = 64
COUPLING = 2.0
NOISE = 0.15
DT = 0.005
NON_RESPONDER = 0.30

SWEEP_CONDITIONS = ['no_stim', 'sham', 'fixed_delta', 'fixed_delta_ssa_resets']

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'


def load_subjects(n: int) -> List[Tuple[str, Dict[str, float]]]:
    """Load first n subjects' baseline powers."""
    subjects = []
    for csv_path in sorted(DATA_DIR.glob('*_processed.csv')):
        subject_id = csv_path.stem.replace('_processed', '')
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in
               ['delta_power', 'theta_power', 'alpha_power', 'beta_power']):
            n_rows = min(10, len(df))
            bp = df.iloc[:n_rows][['delta_power', 'theta_power',
                                    'alpha_power', 'beta_power']].mean().to_dict()
            subjects.append((subject_id, bp))
        if len(subjects) >= n:
            break
    return subjects


def run_single(
    subject_id: str,
    baseline_powers: Dict[str, float],
    condition: str,
    forcing_strength: float,
    alpha_tc: float,
    delta_lambda: float,
) -> Dict[str, float]:
    """Run a single subject × condition and return SWA enhancement."""
    seed_base = hash(subject_id) % (2**31)
    baseline_beta = baseline_powers.get('beta_power', 0.25)

    sham_rng = np.random.default_rng(seed_base + 999)
    protocols = define_redesigned_protocols(
        session_duration_sec=SESSION_DURATION_SEC,
        rng=sham_rng,
        baseline_beta=baseline_beta,
    )

    if condition not in protocols:
        return {'swa_enhancement': float('nan')}

    proto_def = protocols[condition]
    phases = proto_def['phases']
    stim_mode = proto_def['stim_mode']

    cond_seed = seed_base * 14 + list(protocols.keys()).index(condition) * 31
    ensemble = ThalamocorticalEnsemble(
        n_oscillators=N_OSCILLATORS,
        coupling_strength=COUPLING,
        noise_sigma=NOISE,
        dt=DT,
        seed=cond_seed,
        alpha_TC=alpha_tc,
        delta_lambda=delta_lambda,
        lambda_base=-0.3,
    )
    ensemble.initialize_from_baseline(
        baseline_powers,
        non_responder_fraction=NON_RESPONDER,
        sleep_stage_fractions={'W': 0.05, 'N1': 0.10, 'N2': 0.50, 'N3': 0.25, 'REM': 0.10},
    )

    # Burn-in
    ensemble.run_epoch(60.0, 1.0, 0.0)
    ensemble._mf_buffer[:] = 0.0
    ensemble._mf_idx = 0
    ensemble._step_counter = 0

    # Baseline
    ensemble.run_epoch(EPOCH_SEC * 2, 1.0, 0.0)
    baseline_bp = ensemble.compute_band_powers()
    baseline_swa = compute_swa(baseline_bp)
    baseline_sdr = compute_sdr(baseline_bp)
    ensemble._baseline_swa = baseline_swa

    # Reset for session
    ensemble._mf_buffer[:] = 0.0
    ensemble._mf_idx = 0
    ensemble._step_counter = 0

    # Run session
    session_df = ensemble.run_progressive_session(
        baseline_powers=baseline_powers,
        protocol_phases=phases,
        forcing_strength=forcing_strength,
        epoch_sec=EPOCH_SEC,
        non_responder_fraction=NON_RESPONDER,
        baseline_sdr=baseline_sdr,
        skip_init=True,
        sleep_stage_fractions={'W': 0.05, 'N1': 0.10, 'N2': 0.50, 'N3': 0.25, 'REM': 0.10},
        stim_mode=stim_mode,
    )

    if 'swa_enhancement' in session_df.columns and len(session_df) > 0:
        swa_enh = float(session_df['swa_enhancement'].iloc[-5:].mean())
    elif 'swa' in session_df.columns and baseline_swa > 0:
        final_swa = float(session_df['swa'].iloc[-5:].mean())
        swa_enh = (final_swa - baseline_swa) / baseline_swa * 100
    else:
        swa_enh = float('nan')

    sd = float(session_df['swa_enhancement'].std()) if 'swa_enhancement' in session_df.columns else float('nan')
    skew = float(session_df['swa_enhancement'].skew()) if 'swa_enhancement' in session_df.columns else float('nan')

    return {
        'swa_enhancement': swa_enh,
        'sd': sd,
        'skewness': skew,
    }


def main():
    parser = argparse.ArgumentParser(description='Calibration sweep')
    parser.add_argument('--forcing', type=float, nargs='+',
                        default=[0.01, 0.02, 0.03, 0.05])
    parser.add_argument('--delta-lambda', type=float, nargs='+',
                        default=[0.15, 0.25, 0.35])
    parser.add_argument('--alpha-tc', type=float, default=1.5)
    parser.add_argument('--n-subjects', type=int, default=N_SUBJECTS)
    args = parser.parse_args()

    subjects = load_subjects(args.n_subjects)
    if not subjects:
        print("No subject data found!")
        return

    print(f"Calibration sweep: {len(subjects)} subjects × {len(SWEEP_CONDITIONS)} conditions")
    print(f"  forcing_strength: {args.forcing}")
    print(f"  delta_lambda:     {args.delta_lambda}")
    print(f"  alpha_TC:         {args.alpha_tc}")
    print(f"  Session: 30 min\n")

    t0 = time.time()
    results = []

    for f_str in args.forcing:
        for dl in args.delta_lambda:
            print(f"--- forcing={f_str}, delta_lambda={dl} ---")
            for condition in SWEEP_CONDITIONS:
                enhancements = []
                for subject_id, bp in subjects:
                    res = run_single(subject_id, bp, condition, f_str, args.alpha_tc, dl)
                    enhancements.append(res['swa_enhancement'])

                mean_enh = np.nanmean(enhancements)
                std_enh = np.nanstd(enhancements)
                print(f"  {condition:30s}: {mean_enh:+6.1f}% ± {std_enh:5.1f}%")

                results.append({
                    'forcing': f_str,
                    'delta_lambda': dl,
                    'condition': condition,
                    'mean_swa_enh': mean_enh,
                    'std_swa_enh': std_enh,
                })
            print()

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Print summary grid
    print("\n" + "=" * 80)
    print("SUMMARY GRID: Mean SWA Enhancement (%)")
    print("=" * 80)

    df = pd.DataFrame(results)
    for condition in SWEEP_CONDITIONS:
        print(f"\n  {condition}:")
        cond_df = df[df['condition'] == condition]
        # Pivot: rows = delta_lambda, cols = forcing
        pivot = cond_df.pivot_table(
            index='delta_lambda', columns='forcing', values='mean_swa_enh'
        )
        print(pivot.to_string(float_format=lambda x: f"{x:+.1f}%"))

    # Check targets
    print("\n" + "=" * 80)
    print("TARGET CHECK (using default forcing=0.02, delta_lambda=0.25):")
    print("=" * 80)
    for _, row in df[(df['forcing'] == 0.02) & (df['delta_lambda'] == 0.25)].iterrows():
        cond = row['condition']
        val = row['mean_swa_enh']
        targets = {
            'no_stim': (-2, 3),
            'sham': (0, 5),
            'fixed_delta': (10, 22),
            'fixed_delta_ssa_resets': (15, 32),
        }
        lo, hi = targets.get(cond, (-999, 999))
        ok = "PASS" if lo <= val <= hi else "FAIL"
        print(f"  {cond:30s}: {val:+6.1f}%  target=[{lo}, {hi}]  {ok}")


if __name__ == '__main__':
    main()
