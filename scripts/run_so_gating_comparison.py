"""
SO-Phase Excitability Gating — Before vs After Comparison.

Runs all 14 redesigned protocols under two conditions:
  1. so_modulation=0.0 (baseline, current model — no SO gating)
  2. so_modulation=0.5 (new model — moderate SO-phase gating)

Uses 30 representative subjects for speed.

Expected effects of SO gating:
- Better sham SDRE should decrease (random-phase pulses hit down-states ~50%)
- Pulsed protocols should improve relative to continuous
- SSA-Reset should remain top (adaptation advantage is orthogonal)

Usage:
    python scripts/run_so_gating_comparison.py --workers 6
    python scripts/run_so_gating_comparison.py --workers 6 --n-subjects 10
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

from analysis.thalamocortical_model import ThalamocorticalEnsemble, compute_sdr
from analysis.protocol_comparison import EPOCH_SEC, aggregate_protocol_results
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
N_CONDITIONS = 14
SO_MODULATION_VALUES = [0.0, 0.5]


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


def run_subject_so_comparison(args: tuple) -> Tuple[str, float, Dict[str, pd.DataFrame]]:
    """Worker: run all 14 protocols for one subject at a given so_modulation."""
    subject_id, baseline_powers, sleep_stage_fractions, so_modulation = args

    seed_base = hash(subject_id) % (2**31)
    baseline_beta = baseline_powers.get('beta_power', 0.25)

    sham_rng = np.random.default_rng(seed_base + 999)
    protocols = define_redesigned_protocols(
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
        sleep_stage_fractions=sleep_stage_fractions,
    )

    shared_natural_freqs = ref_ensemble.natural_freqs.copy()
    shared_lambda_0 = ref_ensemble.lambda_0.copy()
    shared_forcing_mask = ref_ensemble.forcing_mask.copy()
    shared_alpha_TC = ref_ensemble.alpha_TC
    shared_I_sleep = ref_ensemble.I_sleep

    subject_results = {}
    for cond_idx, (condition, proto_def) in enumerate(protocols.items()):
        phases = proto_def['phases']
        stim_mode = proto_def['stim_mode']

        cond_seed = seed_base * N_CONDITIONS + cond_idx * 31
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

        # Restore shared model parameters
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
        ensemble.A_hab = 0.0
        ensemble._last_forcing_freq = -1.0
        ensemble._last_pulse_time = -1.0
        ensemble.so_phase = 0.0
        ensemble.t = 0.0
        ensemble._mf_buffer[:] = 0.0
        ensemble._mf_idx = 0
        ensemble._step_counter = 0

        # Baseline epoch (F=0)
        ensemble.run_epoch(EPOCH_SEC, 1.0, 0.0)
        baseline_bp = ensemble.compute_band_powers()
        shared_baseline_sdr = compute_sdr(baseline_bp)

        # Handle hybrid mode
        if stim_mode == 'hybrid':
            hybrid_split = proto_def.get('hybrid_split_sec', SESSION_DURATION_SEC / 2)
            first_phases = []
            second_phases = []
            cumulative = 0.0
            for p in phases:
                if cumulative < hybrid_split:
                    first_phases.append(p)
                else:
                    second_phases.append(p)
                cumulative += p['duration_sec']

            ensemble._mf_buffer[:] = 0.0
            ensemble._mf_idx = 0
            ensemble._step_counter = 0

            session_df_1 = ensemble.run_progressive_session(
                baseline_powers=baseline_powers,
                protocol_phases=first_phases,
                forcing_strength=DEFAULT_FORCING,
                epoch_sec=EPOCH_SEC,
                non_responder_fraction=DEFAULT_NON_RESPONDER,
                baseline_sdr=shared_baseline_sdr,
                skip_init=True,
                sleep_stage_fractions=sleep_stage_fractions,
                stim_mode='continuous',
            )

            session_df_2 = ensemble.run_progressive_session(
                baseline_powers=baseline_powers,
                protocol_phases=second_phases,
                forcing_strength=DEFAULT_FORCING,
                epoch_sec=EPOCH_SEC,
                non_responder_fraction=DEFAULT_NON_RESPONDER,
                baseline_sdr=shared_baseline_sdr,
                skip_init=True,
                sleep_stage_fractions=sleep_stage_fractions,
                stim_mode='pulsed',
            )

            if len(session_df_1) > 0 and len(session_df_2) > 0:
                max_idx = session_df_1['epoch_idx'].max() + 1
                max_time = session_df_1['time_sec'].max()
                session_df_2 = session_df_2.copy()
                session_df_2['epoch_idx'] += max_idx
                session_df_2['time_sec'] += max_time

            session_df = pd.concat([session_df_1, session_df_2], ignore_index=True)
        else:
            session_df = ensemble.run_progressive_session(
                baseline_powers=baseline_powers,
                protocol_phases=phases,
                forcing_strength=DEFAULT_FORCING,
                epoch_sec=EPOCH_SEC,
                non_responder_fraction=DEFAULT_NON_RESPONDER,
                baseline_sdr=shared_baseline_sdr,
                skip_init=True,
                sleep_stage_fractions=sleep_stage_fractions,
                stim_mode=stim_mode,
            )

        session_df['condition'] = condition
        session_df['subject_id'] = subject_id
        session_df['so_modulation'] = so_modulation
        subject_results[condition] = session_df

    return subject_id, so_modulation, subject_results


def main():
    parser = argparse.ArgumentParser(
        description='SO-Phase Gating Comparison Study'
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

    logger.info("=" * 70)
    logger.info("SO-PHASE EXCITABILITY GATING — COMPARISON STUDY")
    logger.info("=" * 70)
    logger.info(f"  Subjects: {len(subjects)}")
    logger.info(f"  Conditions: {N_CONDITIONS}")
    logger.info(f"  SO modulation values: {SO_MODULATION_VALUES}")
    logger.info(f"  Total sessions: {len(subjects) * N_CONDITIONS * len(SO_MODULATION_VALUES)}")
    logger.info(f"  Workers: {n_workers}")

    total_t0 = time.time()
    all_epochs_list = []
    all_results_by_mod = {}  # {so_mod: {sid: {cond: df}}}

    for so_mod in SO_MODULATION_VALUES:
        logger.info(f"\n--- Running so_modulation={so_mod} ---")
        tasks = [
            (sid, baselines[sid], sleep_fracs[sid], so_mod)
            for sid in subjects
        ]

        t0 = time.time()
        results_this_mod = {}

        if n_workers > 1:
            with Pool(n_workers) as pool:
                for i, (sid, mod_val, subject_results) in enumerate(
                    pool.imap_unordered(run_subject_so_comparison, tasks)
                ):
                    results_this_mod[sid] = subject_results
                    for cond, sdf in subject_results.items():
                        all_epochs_list.append(sdf)
                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - t0
                        rate = (i + 1) / elapsed
                        remaining = (len(tasks) - i - 1) / rate
                        logger.info(f"  Progress: {i+1}/{len(tasks)} "
                                    f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")
        else:
            for i, task in enumerate(tasks):
                sid, mod_val, subject_results = run_subject_so_comparison(task)
                results_this_mod[sid] = subject_results
                for cond, sdf in subject_results.items():
                    all_epochs_list.append(sdf)
                if (i + 1) % 5 == 0:
                    logger.info(f"  Progress: {i+1}/{len(tasks)} "
                                f"({time.time() - t0:.0f}s)")

        elapsed = time.time() - t0
        logger.info(f"  so_modulation={so_mod} complete in {elapsed:.1f}s")
        all_results_by_mod[so_mod] = results_this_mod

    total_elapsed = time.time() - total_t0

    # Combine all epochs
    all_epochs_df = pd.concat(all_epochs_list, ignore_index=True)

    # Compute per-condition summary (session SDRE = mean of last 10 epochs)
    summary_rows = []
    for so_mod, results_dict in all_results_by_mod.items():
        for sid, cond_dict in results_dict.items():
            for condition, sdf in cond_dict.items():
                n_final = min(10, len(sdf))
                session_sdre = float(sdf.tail(n_final)['sdre'].mean())
                summary_rows.append({
                    'subject_id': sid,
                    'condition': condition,
                    'so_modulation': so_mod,
                    'session_sdre': session_sdre,
                })

    summary_df = pd.DataFrame(summary_rows)

    # Pivot: per-condition mean SDRE with vs without SO gating
    comparison_rows = []
    for condition in summary_df['condition'].unique():
        for so_mod in SO_MODULATION_VALUES:
            cond_data = summary_df[
                (summary_df['condition'] == condition)
                & (summary_df['so_modulation'] == so_mod)
            ]['session_sdre']
            comparison_rows.append({
                'condition': condition,
                'so_modulation': so_mod,
                'mean_sdre': float(cond_data.mean()),
                'std_sdre': float(cond_data.std()),
                'n': len(cond_data),
            })

    comparison_df = pd.DataFrame(comparison_rows)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(RESULTS_DIR / 'comparison_summary.csv', index=False)
    all_epochs_df.to_csv(RESULTS_DIR / 'comparison_epochs.csv', index=False)
    summary_df.to_csv(RESULTS_DIR / 'comparison_session_metrics.csv', index=False)

    logger.info(f"\nResults saved to {RESULTS_DIR}")
    logger.info(f"  comparison_summary.csv: {len(comparison_df)} rows")
    logger.info(f"  comparison_epochs.csv: {len(all_epochs_df)} rows")

    # Print comparison
    print("\n" + "=" * 70)
    print("SO-PHASE GATING COMPARISON — RESULTS")
    print("=" * 70)
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Subjects: {len(subjects)}")

    print(f"\n{'Condition':<30} {'SDRE(0.0)':<12} {'SDRE(0.5)':<12} {'Delta':<10}")
    print("-" * 64)
    for condition in sorted(comparison_df['condition'].unique()):
        row_0 = comparison_df[
            (comparison_df['condition'] == condition)
            & (comparison_df['so_modulation'] == 0.0)
        ]
        row_5 = comparison_df[
            (comparison_df['condition'] == condition)
            & (comparison_df['so_modulation'] == 0.5)
        ]
        if len(row_0) > 0 and len(row_5) > 0:
            sdre_0 = float(row_0['mean_sdre'].iloc[0])
            sdre_5 = float(row_5['mean_sdre'].iloc[0])
            delta = sdre_5 - sdre_0
            print(f"  {condition:<28} {sdre_0:+.4f}      {sdre_5:+.4f}      {delta:+.4f}")

    # Key comparisons
    print("\nKey comparisons:")
    print("-" * 64)

    # Continuous vs pulsed gap
    for so_mod in SO_MODULATION_VALUES:
        cont_delta = summary_df[
            (summary_df['condition'] == 'fixed_delta')
            & (summary_df['so_modulation'] == so_mod)
        ]['session_sdre'].mean()
        pulsed_delta = summary_df[
            (summary_df['condition'] == 'pulsed_fixed_delta')
            & (summary_df['so_modulation'] == so_mod)
        ]['session_sdre'].mean()
        gap = cont_delta - pulsed_delta
        print(f"  Continuous-Pulsed gap (so_mod={so_mod}): {gap:+.4f}")

    # Better sham effect
    for so_mod in SO_MODULATION_VALUES:
        nostim = summary_df[
            (summary_df['condition'] == 'no_stim')
            & (summary_df['so_modulation'] == so_mod)
        ]['session_sdre'].mean()
        better_sham = summary_df[
            (summary_df['condition'] == 'better_sham')
            & (summary_df['so_modulation'] == so_mod)
        ]['session_sdre'].mean()
        effect = better_sham - nostim
        print(f"  Better sham effect (so_mod={so_mod}): {effect:+.4f}")

    # SSA-Reset ranking
    for so_mod in SO_MODULATION_VALUES:
        ranking = summary_df[
            summary_df['so_modulation'] == so_mod
        ].groupby('condition')['session_sdre'].mean().sort_values(ascending=False)
        top3 = list(ranking.index[:3])
        print(f"  Top 3 (so_mod={so_mod}): {', '.join(top3)}")

    print("=" * 70)


if __name__ == '__main__':
    main()
