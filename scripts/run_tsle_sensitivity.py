"""
TSLE Parameter Sensitivity Analysis.

Sweeps 5 key thalamocortical parameters to verify that the progressive > fixed-delta
finding is robust across the parameter space, not an artefact of specific tuning.

Parameters swept:
- gamma (frequency shift fraction): [0.2, 0.35, 0.5, 0.65]
- tau_T (thalamic timescale): [5.0, 10.0, 15.0, 20.0]
- alpha_TC (cortical-to-thalamic coupling): [0.5, 1.0, 1.5, 2.5]
- delta_lambda (excitability boost): [0.1, 0.3, 0.5, 0.8]
- F (forcing strength): [0.5, 1.0, 1.5, 2.0]

For each parameter combination, runs a 20-subject subset and computes
the progressive vs. fixed-delta Cohen's d.

Usage:
    python scripts/run_tsle_sensitivity.py [--workers N] [--n-subjects N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.thalamocortical_model import ThalamocorticalEnsemble, compute_sdr
from analysis.protocol_comparison import EPOCH_SEC, define_protocols

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'protocol_study'

# Parameter sweep ranges
SWEEP_PARAMS = {
    'gamma': [0.2, 0.35, 0.5, 0.65],
    'tau_T': [5.0, 10.0, 15.0, 20.0],
    'alpha_TC': [0.5, 1.0, 3.0, 5.0],
    'delta_lambda': [0.3, 0.8, 1.5, 2.5],
    'forcing': [0.15, 0.3, 0.5, 0.8],
}

# Default values (used when not sweeping a parameter)
DEFAULTS = {
    'gamma': 0.5,
    'tau_T': 10.0,
    'alpha_TC': 5.0,
    'delta_lambda': 1.5,
    'forcing': 0.3,
}


def load_subjects(n_subjects: int = 20) -> Dict[str, pd.DataFrame]:
    """Load a subset of subjects for sensitivity analysis."""
    subjects = {}
    for csv_path in sorted(DATA_DIR.glob('*_processed.csv')):
        subject_id = csv_path.stem.replace('_processed', '')
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in
               ['delta_power', 'theta_power', 'alpha_power', 'beta_power']):
            subjects[subject_id] = df
            if len(subjects) >= n_subjects:
                break
    logger.info(f"Loaded {len(subjects)} subjects for sensitivity analysis")
    return subjects


def get_baseline_powers(df: pd.DataFrame) -> Dict[str, float]:
    """Extract baseline band powers."""
    if 'sleep_stage' in df.columns:
        baseline = df[df['sleep_stage'].isin(['Wake', 'W', 'N1', '1'])]
        if len(baseline) >= 3:
            return baseline[['delta_power', 'theta_power',
                             'alpha_power', 'beta_power']].mean().to_dict()
    n = min(10, len(df))
    return df.iloc[:n][['delta_power', 'theta_power',
                        'alpha_power', 'beta_power']].mean().to_dict()


def get_sleep_stage_fractions(df: pd.DataFrame) -> Dict[str, float]:
    """Extract sleep stage fractions."""
    if 'sleep_stage' not in df.columns:
        return {'W': 0.05, 'N1': 0.10, 'N2': 0.50, 'N3': 0.25, 'REM': 0.10}
    stage_counts = df['sleep_stage'].value_counts(normalize=True)
    fractions = {}
    for stage in ['W', 'Wake', 'N1', '1', 'N2', '2', 'N3', '3', '4', 'REM', 'R']:
        if stage in stage_counts.index:
            fractions[stage] = float(stage_counts[stage])
    return fractions if fractions else {'W': 0.05, 'N1': 0.10, 'N2': 0.50, 'N3': 0.25, 'REM': 0.10}


def run_single_sweep(args: tuple) -> Dict:
    """Run progressive and fixed-delta for one subject with given parameters.

    Uses condition-specific noise (matching the main study design) so SSA
    and TC feedback produce different trajectories per condition.
    """
    (subject_id, baseline_powers, sleep_fractions,
     gamma, tau_T, alpha_TC, delta_lambda, forcing) = args

    seed_base = hash(subject_id) % (2**31)

    # Reference ensemble for shared model parameters
    ref = ThalamocorticalEnsemble(
        n_oscillators=64,
        coupling_strength=2.0,
        noise_sigma=0.15,
        dt=0.005,
        seed=seed_base,
        tau_T=tau_T,
        alpha_TC=alpha_TC,
        gamma=gamma,
        kappa=3.0,
        T_half=0.3,
        delta_lambda=delta_lambda,
        beta_ext=0.05,
        lambda_base=-0.3,
    )
    ref.initialize_from_baseline(
        baseline_powers,
        non_responder_fraction=0.30,
        sleep_stage_fractions=sleep_fractions,
    )
    shared_nf = ref.natural_freqs.copy()
    shared_l0 = ref.lambda_0.copy()
    shared_fm = ref.forcing_mask.copy()
    shared_atc = ref.alpha_TC
    shared_is = ref.I_sleep

    protocols = define_protocols(rng=np.random.default_rng(seed_base + 999))

    results = {}
    for ci, condition in enumerate(['progressive', 'fixed_delta']):
        cond_seed = seed_base * 7 + ci * 31
        ens = ThalamocorticalEnsemble(
            n_oscillators=64,
            coupling_strength=2.0,
            noise_sigma=0.15,
            dt=0.005,
            seed=cond_seed,
            tau_T=tau_T,
            alpha_TC=alpha_TC,
            gamma=gamma,
            kappa=3.0,
            T_half=0.3,
            delta_lambda=delta_lambda,
            beta_ext=0.05,
            lambda_base=-0.3,
        )
        ens.natural_freqs = shared_nf.copy()
        ens.lambda_0 = shared_l0.copy()
        ens.forcing_mask = shared_fm.copy()
        ens.alpha_TC = shared_atc
        ens.I_sleep = shared_is
        ens.z = 0.1 * ens.rng.standard_normal(64) + 0.1j * ens.rng.standard_normal(64)
        ens.T = 0.0
        ens.H = 0.0
        ens.A_hab = 0.0
        ens._last_forcing_freq = -1.0
        ens.t = 0.0
        ens._mf_buffer[:] = 0.0
        ens._mf_idx = 0
        ens._step_counter = 0

        # Baseline epoch
        ens.run_epoch(EPOCH_SEC, 1.0, 0.0)
        bl_sdr = compute_sdr(ens.compute_band_powers())

        session_df = ens.run_progressive_session(
            baseline_powers=baseline_powers,
            protocol_phases=protocols[condition],
            forcing_strength=forcing,
            epoch_sec=EPOCH_SEC,
            non_responder_fraction=0.30,
            baseline_sdr=bl_sdr,
            skip_init=True,
            sleep_stage_fractions=sleep_fractions,
        )
        n_final = min(10, len(session_df))
        results[condition] = float(session_df.tail(n_final)['sdre'].mean())

    return {
        'subject_id': subject_id,
        'progressive_sdre': results['progressive'],
        'fixed_delta_sdre': results['fixed_delta'],
    }


def paired_cohens_d(diffs: np.ndarray) -> float:
    """Compute paired Cohen's d."""
    if len(diffs) < 2 or np.std(diffs, ddof=1) < 1e-10:
        return 0.0
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def main():
    parser = argparse.ArgumentParser(description='TSLE Parameter Sensitivity Analysis')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--n-subjects', type=int, default=20)
    args = parser.parse_args()

    n_workers = args.workers or max(1, cpu_count() - 1)

    subjects = load_subjects(args.n_subjects)
    if not subjects:
        logger.error("No subject data found!")
        return

    baselines = {sid: get_baseline_powers(df) for sid, df in subjects.items()}
    sleep_fracs = {sid: get_sleep_stage_fractions(df) for sid, df in subjects.items()}
    subject_ids = list(subjects.keys())

    logger.info("=" * 70)
    logger.info("TSLE PARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    all_results = {}
    total_t0 = time.time()

    # Sweep each parameter independently (one-at-a-time)
    for param_name, param_values in SWEEP_PARAMS.items():
        logger.info(f"\nSweeping {param_name}: {param_values}")

        for val in param_values:
            # Set this parameter, use defaults for others
            params = dict(DEFAULTS)
            params[param_name] = val

            sweep_key = f"{param_name}={val}"
            logger.info(f"  Running {sweep_key}...")

            tasks = []
            for sid in subject_ids:
                tasks.append((
                    sid, baselines[sid], sleep_fracs[sid],
                    params['gamma'], params['tau_T'], params['alpha_TC'],
                    params['delta_lambda'], params['forcing'],
                ))

            t0 = time.time()
            subject_results = []
            if n_workers > 1:
                with Pool(n_workers) as pool:
                    subject_results = list(pool.map(run_single_sweep, tasks))
            else:
                for task in tasks:
                    subject_results.append(run_single_sweep(task))

            # Compute Cohen's d
            prog_sdre = np.array([r['progressive_sdre'] for r in subject_results])
            delta_sdre = np.array([r['fixed_delta_sdre'] for r in subject_results])
            diffs = prog_sdre - delta_sdre
            d = paired_cohens_d(diffs)

            elapsed = time.time() - t0
            logger.info(f"    Cohen's d (prog vs fixed-delta) = {d:.3f}  "
                        f"({elapsed:.1f}s)")

            all_results[sweep_key] = {
                'parameter': param_name,
                'value': val,
                'cohens_d': d,
                'mean_diff': float(np.mean(diffs)),
                'progressive_mean': float(np.mean(prog_sdre)),
                'fixed_delta_mean': float(np.mean(delta_sdre)),
                'n_subjects': len(subject_results),
                'prog_gt_delta_frac': float(np.mean(diffs > 0)),
            }

    total_elapsed = time.time() - total_t0

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / 'tsle_sensitivity.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TSLE SENSITIVITY ANALYSIS — SUMMARY")
    print("=" * 70)
    print(f"\nTotal runtime: {total_elapsed:.1f}s")
    print(f"Subjects per sweep: {args.n_subjects}")

    # Count how many parameter settings show progressive > fixed-delta
    n_positive = sum(1 for r in all_results.values() if r['cohens_d'] > 0.2)
    n_total = len(all_results)
    print(f"\nRobustness: {n_positive}/{n_total} settings show d > 0.2 "
          f"({100*n_positive/n_total:.0f}%)")

    print(f"\n{'Parameter':<20} {'Value':<10} {'Cohen d':<10} {'Mean diff':<12} "
          f"{'% prog>delta':<12}")
    print("-" * 64)
    for key, r in sorted(all_results.items()):
        print(f"  {r['parameter']:<18} {r['value']:<10.2f} {r['cohens_d']:<10.3f} "
              f"{r['mean_diff']:<12.4f} {r['prog_gt_delta_frac']*100:<12.0f}")

    print("=" * 70)


if __name__ == '__main__':
    main()
