"""
SO-Phase Excitability Gating — Sensitivity Sweep.

Sweeps so_modulation from 0.0 to 1.0 in steps of 0.1 for 4 key protocols:
  1. fixed_delta (continuous) — baseline continuous protocol
  2. pulsed_fixed_delta — pulsed delivery
  3. fixed_delta_ssa_resets — SSA reset protocol
  4. better_sham — phase-randomized pulsed sham

Finds the gating strength where:
  - Pulsed ≈ continuous (continuous-pulsed gap closes)
  - Better sham ≈ no_stim (sham effect diminishes)

Uses 30 subjects for speed.

Usage:
    python scripts/run_so_gating_sensitivity.py --workers 6
    python scripts/run_so_gating_sensitivity.py --workers 6 --n-subjects 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.thalamocortical_model import ThalamocorticalEnsemble, compute_sdr, compute_swa, compute_swa_enhancement
from analysis.protocol_comparison import EPOCH_SEC
from analysis.redesigned_protocols import define_redesigned_protocols

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'so_gating_study'

# Model parameters (match redesigned study)
DEFAULT_N_OSCILLATORS = 64
DEFAULT_COUPLING = 2.0
DEFAULT_NOISE = 0.15
DEFAULT_DT = 0.005
DEFAULT_FORCING = 0.3
DEFAULT_NON_RESPONDER = 0.30
DEFAULT_TAU_T = 10.0
DEFAULT_ALPHA_TC = 5.0
DEFAULT_GAMMA = 0.5
DEFAULT_KAPPA = 3.0
DEFAULT_T_HALF = 0.3
DEFAULT_DELTA_LAMBDA = 1.5
DEFAULT_BETA_EXT = 0.05
DEFAULT_LAMBDA_BASE = -0.3

SESSION_DURATION_SEC = 3600.0  # 60 min

# Key protocols to sweep
KEY_CONDITIONS = [
    'fixed_delta',
    'pulsed_fixed_delta',
    'fixed_delta_ssa_resets',
    'better_sham',
    'no_stim',
]

# SO modulation sweep values
SO_MOD_VALUES = [round(x * 0.1, 1) for x in range(11)]  # 0.0, 0.1, ..., 1.0


def load_subjects(n_subjects: int = 30) -> Dict[str, pd.DataFrame]:
    """Load a subset of subjects."""
    subjects = {}
    for csv_path in sorted(DATA_DIR.glob('*_processed.csv')):
        subject_id = csv_path.stem.replace('_processed', '')
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in
               ['delta_power', 'theta_power', 'alpha_power', 'beta_power']):
            subjects[subject_id] = df
            if len(subjects) >= n_subjects:
                break
    logger.info(f"Loaded {len(subjects)} subjects")
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
    """Worker: run key protocols for one subject at a given so_modulation."""
    subject_id, baseline_powers, sleep_fractions, so_modulation = args

    seed_base = hash(subject_id) % (2**31)
    baseline_beta = baseline_powers.get('beta_power', 0.25)

    # Define protocols (need all 14 for correct indexing, but only run KEY_CONDITIONS)
    sham_rng = np.random.default_rng(seed_base + 999)
    all_protocols = define_redesigned_protocols(
        session_duration_sec=SESSION_DURATION_SEC,
        rng=sham_rng,
        baseline_beta=baseline_beta,
    )

    # Initialize reference ensemble for shared parameters
    ref_ensemble = ThalamocorticalEnsemble(
        n_oscillators=DEFAULT_N_OSCILLATORS,
        coupling_strength=DEFAULT_COUPLING,
        noise_sigma=DEFAULT_NOISE,
        dt=DEFAULT_DT,
        seed=seed_base,
        tau_T=DEFAULT_TAU_T,
        alpha_TC=DEFAULT_ALPHA_TC,
        gamma=DEFAULT_GAMMA,
        kappa=DEFAULT_KAPPA,
        T_half=DEFAULT_T_HALF,
        delta_lambda=DEFAULT_DELTA_LAMBDA,
        beta_ext=DEFAULT_BETA_EXT,
        lambda_base=DEFAULT_LAMBDA_BASE,
        so_modulation=so_modulation,
    )
    ref_ensemble.initialize_from_baseline(
        baseline_powers,
        non_responder_fraction=DEFAULT_NON_RESPONDER,
        sleep_stage_fractions=sleep_fractions,
    )

    shared_natural_freqs = ref_ensemble.natural_freqs.copy()
    shared_lambda_0 = ref_ensemble.lambda_0.copy()
    shared_forcing_mask = ref_ensemble.forcing_mask.copy()
    shared_alpha_TC = ref_ensemble.alpha_TC
    shared_I_sleep = ref_ensemble.I_sleep

    # Get consistent condition indices for seeding
    all_cond_names = list(all_protocols.keys())

    results = {
        'subject_id': subject_id,
        'so_modulation': so_modulation,
    }

    for condition in KEY_CONDITIONS:
        if condition not in all_protocols:
            continue

        proto_def = all_protocols[condition]
        phases = proto_def['phases']
        stim_mode = proto_def['stim_mode']
        cond_idx = all_cond_names.index(condition)

        cond_seed = seed_base * 14 + cond_idx * 31
        ensemble = ThalamocorticalEnsemble(
            n_oscillators=DEFAULT_N_OSCILLATORS,
            coupling_strength=DEFAULT_COUPLING,
            noise_sigma=DEFAULT_NOISE,
            dt=DEFAULT_DT,
            seed=cond_seed,
            tau_T=DEFAULT_TAU_T,
            alpha_TC=DEFAULT_ALPHA_TC,
            gamma=DEFAULT_GAMMA,
            kappa=DEFAULT_KAPPA,
            T_half=DEFAULT_T_HALF,
            delta_lambda=DEFAULT_DELTA_LAMBDA,
            beta_ext=DEFAULT_BETA_EXT,
            lambda_base=DEFAULT_LAMBDA_BASE,
            so_modulation=so_modulation,
        )

        ensemble.natural_freqs = shared_natural_freqs.copy()
        ensemble.lambda_0 = shared_lambda_0.copy()
        ensemble.forcing_mask = shared_forcing_mask.copy()
        ensemble.alpha_TC = shared_alpha_TC
        ensemble.I_sleep = shared_I_sleep

        # For better_sham: randomize pulse phase target
        if proto_def.get('randomize_phase', False):
            ensemble.pulse_phase_target = float(
                ensemble.rng.uniform(-np.pi, np.pi)
            )

        # Fresh initial state
        ensemble.z = (
            0.1 * ensemble.rng.standard_normal(ensemble.N)
            + 0.1j * ensemble.rng.standard_normal(ensemble.N)
        )
        ensemble.T = 0.0
        ensemble.H = 0.0
        ensemble.A_fast = 0.0
        ensemble.A_slow = 0.0
        ensemble._last_forcing_freq = -1.0
        ensemble._last_pulse_time = -1.0
        ensemble._baseline_swa = None
        ensemble.so_phase = 0.0
        ensemble.t = 0.0
        ensemble._mf_buffer[:] = 0.0
        ensemble._mf_idx = 0
        ensemble._step_counter = 0
        ensemble._so_buffer[:] = 0.0
        ensemble._so_buf_idx = 0
        ensemble._so_buf_filled = False
        ensemble._so_sample_counter = 0

        # Baseline epoch
        ensemble.run_epoch(EPOCH_SEC, 1.0, 0.0)
        baseline_bp = ensemble.compute_band_powers()
        baseline_sdr = compute_sdr(baseline_bp)
        baseline_swa = compute_swa(baseline_bp)
        ensemble._baseline_swa = baseline_swa

        # Run session
        session_df = ensemble.run_progressive_session(
            baseline_powers=baseline_powers,
            protocol_phases=phases,
            forcing_strength=DEFAULT_FORCING,
            epoch_sec=EPOCH_SEC,
            non_responder_fraction=DEFAULT_NON_RESPONDER,
            baseline_sdr=baseline_sdr,
            skip_init=True,
            sleep_stage_fractions=sleep_fractions,
            stim_mode=stim_mode,
        )

        n_final = min(10, len(session_df))
        final_epochs = session_df.tail(n_final)
        session_sdre = float(final_epochs['sdre'].mean())
        results[f'{condition}_sdre'] = session_sdre

        # SWA enhancement (primary v2 metric)
        if 'swa' in session_df.columns:
            final_swa = float(final_epochs['swa'].mean())
            swa_enh = compute_swa_enhancement(final_swa, baseline_swa)
            results[f'{condition}_swa_enhancement'] = swa_enh

    return results


def main():
    parser = argparse.ArgumentParser(
        description='SO-Phase Gating Sensitivity Sweep'
    )
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--n-subjects', type=int, default=30)
    args = parser.parse_args()

    n_workers = args.workers or max(1, cpu_count() - 1)

    subjects = load_subjects(args.n_subjects)
    if not subjects:
        logger.error("No subject data found!")
        return

    baselines = {sid: get_baseline_powers(df) for sid, df in subjects.items()}
    sleep_fracs = {sid: get_sleep_stage_fractions(df) for sid, df in subjects.items()}
    subject_ids = list(subjects.keys())

    n_total_tasks = len(subject_ids) * len(SO_MOD_VALUES)

    logger.info("=" * 70)
    logger.info("SO-PHASE GATING — SENSITIVITY SWEEP")
    logger.info("=" * 70)
    logger.info(f"  Subjects: {len(subjects)}")
    logger.info(f"  Key conditions: {KEY_CONDITIONS}")
    logger.info(f"  SO modulation sweep: {SO_MOD_VALUES}")
    logger.info(f"  Total worker tasks: {n_total_tasks}")
    logger.info(f"  Workers: {n_workers}")

    total_t0 = time.time()
    all_results = []

    for so_mod in SO_MOD_VALUES:
        logger.info(f"\n--- so_modulation={so_mod} ---")
        tasks = [
            (sid, baselines[sid], sleep_fracs[sid], so_mod)
            for sid in subject_ids
        ]

        t0 = time.time()
        if n_workers > 1:
            with Pool(n_workers) as pool:
                sweep_results = list(pool.map(run_single_sweep, tasks))
        else:
            sweep_results = [run_single_sweep(t) for t in tasks]

        all_results.extend(sweep_results)
        elapsed = time.time() - t0

        # Quick summary for this so_mod
        sdre_vals = {}
        swa_enh_vals = {}
        for cond in KEY_CONDITIONS:
            key = f'{cond}_sdre'
            vals = [r[key] for r in sweep_results if key in r]
            if vals:
                sdre_vals[cond] = np.mean(vals)
            swa_key = f'{cond}_swa_enhancement'
            swa_vals = [r[swa_key] for r in sweep_results if swa_key in r]
            if swa_vals:
                swa_enh_vals[cond] = np.mean(swa_vals)

        logger.info(f"  Completed in {elapsed:.1f}s")
        for cond in KEY_CONDITIONS:
            sdre = sdre_vals.get(cond, float('nan'))
            swa_e = swa_enh_vals.get(cond, float('nan'))
            logger.info(f"    {cond:<30}: SDRE={sdre:+.4f}  SWA_enh={swa_e:+.1f}%")

    total_elapsed = time.time() - total_t0

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / 'sensitivity_sweep.csv', index=False)

    # Build aggregated summary
    agg_rows = []
    for so_mod in SO_MOD_VALUES:
        mod_data = results_df[results_df['so_modulation'] == so_mod]
        row = {'so_modulation': so_mod, 'n_subjects': len(mod_data)}
        for cond in KEY_CONDITIONS:
            key = f'{cond}_sdre'
            if key in mod_data.columns:
                row[f'{cond}_mean_sdre'] = float(mod_data[key].mean())
                row[f'{cond}_std_sdre'] = float(mod_data[key].std())
            swa_key = f'{cond}_swa_enhancement'
            if swa_key in mod_data.columns:
                row[f'{cond}_mean_swa_enh'] = float(mod_data[swa_key].mean())
                row[f'{cond}_std_swa_enh'] = float(mod_data[swa_key].std())
        # Derived metrics
        cont = row.get('fixed_delta_mean_sdre', 0)
        pulsed = row.get('pulsed_fixed_delta_mean_sdre', 0)
        row['continuous_pulsed_gap'] = cont - pulsed
        nostim = row.get('no_stim_mean_sdre', 0)
        bsham = row.get('better_sham_mean_sdre', 0)
        row['better_sham_effect'] = bsham - nostim
        # SWA-based gaps
        cont_swa = row.get('fixed_delta_mean_swa_enh', 0)
        pulsed_swa = row.get('pulsed_fixed_delta_mean_swa_enh', 0)
        row['continuous_pulsed_swa_gap'] = cont_swa - pulsed_swa
        nostim_swa = row.get('no_stim_mean_swa_enh', 0)
        bsham_swa = row.get('better_sham_mean_swa_enh', 0)
        row['better_sham_swa_effect'] = bsham_swa - nostim_swa
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(RESULTS_DIR / 'sensitivity_summary.csv', index=False)

    logger.info(f"\nResults saved to {RESULTS_DIR}")
    logger.info(f"  sensitivity_sweep.csv: {len(results_df)} rows")
    logger.info(f"  sensitivity_summary.csv: {len(agg_df)} rows")

    # Print summary
    print("\n" + "=" * 70)
    print("SO-PHASE GATING SENSITIVITY — RESULTS")
    print("=" * 70)
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Subjects: {len(subjects)}")

    print(f"\n{'so_mod':<8}", end='')
    for cond in KEY_CONDITIONS:
        label = cond[:20]
        print(f" {label:<22}", end='')
    print(f" {'Cont-Puls Gap':<14} {'BS Effect':<10}")
    print("-" * (8 + 22 * len(KEY_CONDITIONS) + 24))

    for _, row in agg_df.iterrows():
        print(f"  {row['so_modulation']:<6.1f}", end='')
        for cond in KEY_CONDITIONS:
            key = f'{cond}_mean_sdre'
            if key in row:
                print(f" {row[key]:+.4f}               ", end='')
            else:
                print(f" {'N/A':>22}", end='')
        print(f" {row['continuous_pulsed_gap']:+.4f}         {row['better_sham_effect']:+.4f}")

    # Find crossover points
    print("\nCrossover analysis:")
    print("-" * 40)
    gaps = agg_df['continuous_pulsed_gap'].values
    for i in range(len(gaps) - 1):
        if gaps[i] > 0 and gaps[i + 1] <= 0:
            crossover = SO_MOD_VALUES[i] + 0.1 * gaps[i] / (gaps[i] - gaps[i + 1])
            print(f"  Pulsed ≈ continuous at so_modulation ≈ {crossover:.2f}")
            break
    else:
        if gaps[-1] > 0:
            print(f"  Pulsed still < continuous at so_modulation=1.0 (gap={gaps[-1]:+.4f})")
        else:
            print(f"  Pulsed >= continuous already at so_modulation=0.0")

    bs_effects = agg_df['better_sham_effect'].values
    for i in range(len(bs_effects) - 1):
        if bs_effects[i] > 0.01 and bs_effects[i + 1] <= 0.01:
            crossover = SO_MOD_VALUES[i] + 0.1 * (bs_effects[i] - 0.01) / (bs_effects[i] - bs_effects[i + 1])
            print(f"  Better sham effect ≈ 0 at so_modulation ≈ {crossover:.2f}")
            break
    else:
        if bs_effects[-1] > 0.01:
            print(f"  Better sham effect persists at so_modulation=1.0 ({bs_effects[-1]:+.4f})")
        else:
            print(f"  Better sham effect already negligible at so_modulation=0.0")

    print("=" * 70)


if __name__ == '__main__':
    main()
