"""
Redesigned Progressive Frequency Descent — Second-Generation Study.

Compares 14 stimulation protocols (7 original + 7 new) using a within-subject
repeated-measures design across 208 subjects from 5 sleep databases.

New conditions test:
- Pulsed (SO phase-locked) vs continuous stimulation
- Extended sessions (60/90 min) for adaptation crossover
- SSA reset mechanism (periodic frequency wobbles)
- Adaptive protocol (baseline beta-guided assignment)
- Better sham control (phase-randomized pulses)
- Hybrid protocol (continuous first half + pulsed second half)

Phases:
1. Run all protocols: N subjects x 14 conditions
2. Statistical analysis: Friedman omnibus + pairwise Wilcoxon + FDR + targeted contrasts
3. Cross-session-duration analysis (if multiple durations run)
4. Responder subgroup analysis (median split on baseline beta)

Usage:
    python scripts/run_redesigned_study.py --workers 6 --duration 60
    python scripts/run_redesigned_study.py --workers 6 --duration 60 --n-subjects 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.thalamocortical_model import ThalamocorticalEnsemble, compute_sdr, compute_swa, compute_swa_enhancement
from analysis.protocol_comparison import (
    EPOCH_SEC,
    aggregate_protocol_results,
    compute_session_metrics,
)
from analysis.redesigned_protocols import (
    define_redesigned_protocols,
    compute_extended_metrics,
)
from analysis.statistical_validation import (
    run_redesigned_validation,
)
from analysis.redesigned_figures import generate_redesigned_figures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'redesigned_study'

# Model parameters (same as original study)
DEFAULT_N_OSCILLATORS = 64
DEFAULT_COUPLING = 2.0
DEFAULT_NOISE = 0.15
DEFAULT_DT = 0.005
DEFAULT_FORCING = 0.10
DEFAULT_NON_RESPONDER = 0.30

# TSLE-specific parameters
DEFAULT_TAU_T = 10.0
DEFAULT_ALPHA_TC = 1.5
DEFAULT_GAMMA = 0.5
DEFAULT_KAPPA = 3.0
DEFAULT_T_HALF = 0.3
DEFAULT_DELTA_LAMBDA = 0.30
DEFAULT_BETA_EXT = 0.05
DEFAULT_LAMBDA_BASE = -0.3

N_CONDITIONS = 16


def load_all_subjects(max_subjects: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Load all processed subject CSVs."""
    subjects = {}
    for csv_path in sorted(DATA_DIR.glob('*_processed.csv')):
        subject_id = csv_path.stem.replace('_processed', '')
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in
               ['delta_power', 'theta_power', 'alpha_power', 'beta_power']):
            subjects[subject_id] = df
        if max_subjects and len(subjects) >= max_subjects:
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
    """Extract sleep stage fractions for I_sleep computation."""
    if 'sleep_stage' not in df.columns:
        return {'W': 0.05, 'N1': 0.10, 'N2': 0.50, 'N3': 0.25, 'REM': 0.10}

    stage_counts = df['sleep_stage'].value_counts(normalize=True)
    fractions = {}
    for stage in ['W', 'Wake', 'N1', '1', 'N2', '2', 'N3', '3', '4', 'REM', 'R']:
        if stage in stage_counts.index:
            fractions[stage] = float(stage_counts[stage])
    return fractions if fractions else {'W': 0.05, 'N1': 0.10, 'N2': 0.50, 'N3': 0.25, 'REM': 0.10}


def classify_dataset(subject_id: str) -> str:
    """Classify subject into dataset."""
    if subject_id.startswith('SC4'):
        return 'sleep_edf'
    elif subject_id.startswith('CAP'):
        return 'cap'
    elif subject_id.startswith('DREAMS'):
        return 'dreams'
    elif subject_id.startswith('HMC'):
        return 'hmc'
    elif subject_id.startswith('SLPDB'):
        return 'slpdb'
    return 'other'


def run_subject_all_conditions_redesigned(
    args: tuple,
) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """Worker: run all 14 protocols for one subject.

    Within-subject design: same model parameters (natural frequencies,
    excitability, forcing mask, TC coupling) for all conditions, but
    different noise realizations per condition.
    """
    subject_id, baseline_powers, sleep_stage_fractions, session_duration_sec = args

    seed_base = hash(subject_id) % (2**31)
    baseline_beta = baseline_powers.get('beta_power', 0.25)

    # Define redesigned protocols
    sham_rng = np.random.default_rng(seed_base + 999)
    protocols = define_redesigned_protocols(
        session_duration_sec=session_duration_sec,
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
    )
    ref_ensemble.initialize_from_baseline(
        baseline_powers,
        non_responder_fraction=DEFAULT_NON_RESPONDER,
        sleep_stage_fractions=sleep_stage_fractions,
    )

    # Save shared model parameters
    shared_natural_freqs = ref_ensemble.natural_freqs.copy()
    shared_lambda_0 = ref_ensemble.lambda_0.copy()
    shared_forcing_mask = ref_ensemble.forcing_mask.copy()
    shared_alpha_TC = ref_ensemble.alpha_TC
    shared_I_sleep = ref_ensemble.I_sleep

    subject_results = {}
    for cond_idx, (condition, proto_def) in enumerate(protocols.items()):
        phases = proto_def['phases']
        stim_mode = proto_def['stim_mode']

        # Fresh ensemble with condition-specific noise seed
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
        )

        # Restore shared model parameters
        ensemble.natural_freqs = shared_natural_freqs.copy()
        ensemble.lambda_0 = shared_lambda_0.copy()
        ensemble.forcing_mask = shared_forcing_mask.copy()
        ensemble.alpha_TC = shared_alpha_TC
        ensemble.I_sleep = shared_I_sleep

        # Apply per-condition parameter overrides
        if 'tau_slow_override' in proto_def:
            ensemble.tau_slow = proto_def['tau_slow_override']

        # Determine forcing strength for this condition
        cond_forcing = proto_def.get('forcing_override', DEFAULT_FORCING)

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

        # Burn-in: let system reach stochastic equilibrium
        ensemble.run_epoch(60.0, 1.0, 0.0)
        # Reset buffers for clean baseline measurement
        ensemble._mf_buffer[:] = 0.0
        ensemble._mf_idx = 0
        ensemble._step_counter = 0
        # Measure baseline over 2 epochs (60s)
        ensemble.run_epoch(EPOCH_SEC * 2, 1.0, 0.0)
        baseline_bp = ensemble.compute_band_powers()
        shared_baseline_sdr = compute_sdr(baseline_bp)
        baseline_swa = compute_swa(baseline_bp)
        ensemble._baseline_swa = baseline_swa

        # Handle hybrid mode: first half continuous, second half pulsed
        if stim_mode == 'hybrid':
            hybrid_split = proto_def.get('hybrid_split_sec', session_duration_sec / 2)
            # Run first half (continuous)
            first_phases = []
            second_phases = []
            cumulative = 0.0
            for p in phases:
                if cumulative < hybrid_split:
                    first_phases.append(p)
                else:
                    second_phases.append(p)
                cumulative += p['duration_sec']

            # Reset buffer for session
            ensemble._mf_buffer[:] = 0.0
            ensemble._mf_idx = 0
            ensemble._step_counter = 0
            ensemble._so_buffer[:] = 0.0
            ensemble._so_buf_idx = 0
            ensemble._so_buf_filled = False
            ensemble._so_sample_counter = 0

            # Run first half continuous
            session_df_1 = ensemble.run_progressive_session(
                baseline_powers=baseline_powers,
                protocol_phases=first_phases,
                forcing_strength=cond_forcing,
                epoch_sec=EPOCH_SEC,
                non_responder_fraction=DEFAULT_NON_RESPONDER,
                baseline_sdr=shared_baseline_sdr,
                skip_init=True,
                sleep_stage_fractions=sleep_stage_fractions,
                stim_mode='continuous',
            )

            # Run second half pulsed (state carries over)
            session_df_2 = ensemble.run_progressive_session(
                baseline_powers=baseline_powers,
                protocol_phases=second_phases,
                forcing_strength=cond_forcing,
                epoch_sec=EPOCH_SEC,
                non_responder_fraction=DEFAULT_NON_RESPONDER,
                baseline_sdr=shared_baseline_sdr,
                skip_init=True,
                sleep_stage_fractions=sleep_stage_fractions,
                stim_mode='pulsed',
            )

            # Fix epoch indices in second half
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
                forcing_strength=cond_forcing,
                epoch_sec=EPOCH_SEC,
                non_responder_fraction=DEFAULT_NON_RESPONDER,
                baseline_sdr=shared_baseline_sdr,
                skip_init=True,
                sleep_stage_fractions=sleep_stage_fractions,
                stim_mode=stim_mode,
            )

        session_df['condition'] = condition
        session_df['subject_id'] = subject_id
        session_df['baseline_beta'] = baseline_beta

        # Store adaptive choice info
        if condition == 'adaptive_protocol':
            session_df['adaptive_choice'] = proto_def.get('adaptive_choice', 'unknown')

        subject_results[condition] = session_df

    return subject_id, subject_results


# ─── Phase 1: Run All Protocols ──────────────────────────────────────

def run_phase1(
    subjects: Dict[str, pd.DataFrame],
    session_duration_sec: float,
    n_workers: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Run all 14 protocols for all subjects."""
    logger.info("=" * 70)
    logger.info(f"PHASE 1: Running All Protocols ({len(subjects)} subjects x "
                f"{N_CONDITIONS} conditions)")
    logger.info("=" * 70)

    baselines = {sid: get_baseline_powers(df) for sid, df in subjects.items()}
    sleep_fractions = {sid: get_sleep_stage_fractions(df) for sid, df in subjects.items()}
    tasks = [
        (sid, baselines[sid], sleep_fractions[sid], session_duration_sec)
        for sid in subjects
    ]

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    logger.info(f"  Subjects: {len(subjects)}")
    logger.info(f"  Conditions: {N_CONDITIONS}")
    logger.info(f"  Session duration: {session_duration_sec / 60:.0f} min")
    logger.info(f"  Total sessions: {len(subjects) * N_CONDITIONS}")
    logger.info(f"  Workers: {n_workers}")

    t0 = time.time()
    all_results = {}

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, (sid, subject_results) in enumerate(
                pool.imap_unordered(run_subject_all_conditions_redesigned, tasks)
            ):
                all_results[sid] = subject_results
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    remaining = (len(tasks) - i - 1) / rate
                    logger.info(f"  Progress: {i+1}/{len(tasks)} subjects "
                                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
    else:
        for i, task in enumerate(tasks):
            sid, subject_results = run_subject_all_conditions_redesigned(task)
            all_results[sid] = subject_results
            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                logger.info(f"  Progress: {i+1}/{len(tasks)} subjects ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"  Phase 1 complete in {elapsed:.1f}s")

    # Collect all epoch-level data
    all_epochs_list = []
    for sid, conditions in all_results.items():
        for condition, session_df in conditions.items():
            all_epochs_list.append(session_df)
    all_epochs_df = pd.concat(all_epochs_list, ignore_index=True)

    # Aggregate session-level metrics
    session_metrics_df = aggregate_protocol_results(all_results)

    # Add baseline_beta to session metrics
    for sid in all_results:
        for cond, sdf in all_results[sid].items():
            if 'baseline_beta' in sdf.columns:
                beta_val = float(sdf['baseline_beta'].iloc[0])
                mask = (session_metrics_df['subject_id'] == sid) & \
                       (session_metrics_df['condition'] == cond)
                session_metrics_df.loc[mask, 'baseline_beta'] = beta_val

    # Compute extended metrics
    extended_rows = []
    for sid, conditions in all_results.items():
        for condition, session_df in conditions.items():
            ext_metrics = compute_extended_metrics(session_df)
            ext_metrics['subject_id'] = sid
            ext_metrics['condition'] = condition
            extended_rows.append(ext_metrics)
    extended_df = pd.DataFrame(extended_rows)
    session_metrics_df = session_metrics_df.merge(
        extended_df, on=['subject_id', 'condition'], how='left',
    )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_epochs_df.to_csv(RESULTS_DIR / 'all_epochs.csv', index=False)
    session_metrics_df.to_csv(RESULTS_DIR / 'session_metrics.csv', index=False)
    logger.info(f"  Saved {len(all_epochs_df)} epoch rows")
    logger.info(f"  Saved {len(session_metrics_df)} session metric rows")

    # Summary
    logger.info("\n  Session SWA Enhancement by condition (mean +/- std):")
    if 'session_swa_enhancement' in session_metrics_df.columns:
        swa_summary = session_metrics_df.groupby('condition')['session_swa_enhancement'].agg(
            ['mean', 'std']
        ).sort_values('mean', ascending=False)
        for cond, row in swa_summary.iterrows():
            logger.info(f"    {cond:30s}: {row['mean']:+.1f}% +/- {row['std']:.1f}%")

    logger.info("\n  Session SDRE by condition (mean +/- std):")
    summary = session_metrics_df.groupby('condition')['session_sdre'].agg(
        ['mean', 'std']
    ).sort_values('mean', ascending=False)
    for cond, row in summary.iterrows():
        logger.info(f"    {cond:30s}: {row['mean']:+.4f} +/- {row['std']:.4f}")

    return all_epochs_df, session_metrics_df, all_results


# ─── Phase 2: Statistical Analysis ───────────────────────────────────

def run_phase2(session_metrics_df: pd.DataFrame) -> Dict:
    """Run statistical tests on redesigned study results."""
    logger.info("=" * 70)
    logger.info("PHASE 2: Statistical Analysis (14 conditions)")
    logger.info("=" * 70)

    stats_dir = RESULTS_DIR / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)

    report = run_redesigned_validation(
        session_metrics_df,
        output_dir=str(stats_dir),
    )

    # Print summary
    logger.info("\n  Omnibus tests:")
    for metric, result in report['omnibus_tests'].items():
        sig = ("***" if result['p_value'] < 0.001 else
               "**" if result['p_value'] < 0.01 else
               "*" if result['p_value'] < 0.05 else "ns")
        logger.info(f"    {metric}: {result['method']}, "
                    f"p={result['p_value']:.4f} {sig}")

    logger.info("\n  Key targeted contrasts (FDR-corrected):")
    for comp in report.get('targeted_contrasts', []):
        if comp.get('metric') != 'session_sdre':
            continue
        sig = "SIG" if comp.get('significant_fdr') else "ns"
        label = comp.get('contrast_label', f"{comp['target']} vs {comp['control']}")
        logger.info(f"    {label:35s}: d={comp['cohens_d']:+.3f}, "
                    f"p_fdr={comp.get('p_value_fdr', comp['p_value']):.4f} [{sig}]")

    return report


# ─── Phase 3: Cross-Duration Analysis ────────────────────────────────

def run_phase3(
    session_metrics_df: pd.DataFrame,
    duration_label: str = '60min',
) -> Dict:
    """Cross-dataset validation for redesigned study."""
    logger.info("=" * 70)
    logger.info("PHASE 3: Cross-Dataset Validation")
    logger.info("=" * 70)

    metrics_df = session_metrics_df.copy()
    metrics_df['dataset'] = metrics_df['subject_id'].apply(classify_dataset)

    discovery = metrics_df[metrics_df['dataset'] == 'sleep_edf']
    validation = metrics_df[metrics_df['dataset'] != 'sleep_edf']

    n_disc = discovery['subject_id'].nunique()
    n_val = validation['subject_id'].nunique()
    logger.info(f"  Discovery (Sleep-EDF): {n_disc} subjects")
    logger.info(f"  Validation (others): {n_val} subjects")

    result = {'duration': duration_label}
    key_conditions = ['progressive', 'fixed_delta', 'fixed_delta_ssa_resets',
                      'pulsed_progressive', 'pulsed_fixed_delta',
                      'adaptive_protocol', 'ssa_reset_fast', 'ssa_reset_slow']

    for metric in ['session_swa_enhancement', 'session_sdre']:
        metric_results = {}
        for cond in key_conditions:
            disc_data = discovery[discovery['condition'] == cond][metric]
            val_data = validation[validation['condition'] == cond][metric]
            disc_nostim = discovery[discovery['condition'] == 'no_stim'][metric]
            val_nostim = validation[validation['condition'] == 'no_stim'][metric]

            if len(disc_data) > 0 and len(disc_nostim) > 0:
                disc_effect = float(disc_data.mean() - disc_nostim.mean())
            else:
                disc_effect = float('nan')
            if len(val_data) > 0 and len(val_nostim) > 0:
                val_effect = float(val_data.mean() - val_nostim.mean())
            else:
                val_effect = float('nan')

            same_dir = (disc_effect > 0) == (val_effect > 0) if np.isfinite(disc_effect) and np.isfinite(val_effect) else False

            metric_results[cond] = {
                'discovery_effect': disc_effect,
                'validation_effect': val_effect,
                'same_direction': bool(same_dir),
            }
            logger.info(f"  {cond}: disc={disc_effect:+.4f}, val={val_effect:+.4f}, "
                        f"same_dir={same_dir}")

        result[metric] = metric_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'cross_validation.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ─── Phase 4: Responder Subgroup Analysis ─────────────────────────────

def run_phase4(session_metrics_df: pd.DataFrame) -> Dict:
    """Responder subgroup analysis: median split on baseline beta."""
    logger.info("=" * 70)
    logger.info("PHASE 4: Responder Subgroup Analysis")
    logger.info("=" * 70)

    result = {}

    if 'baseline_beta' not in session_metrics_df.columns:
        logger.warning("  No baseline_beta column — skipping subgroup analysis")
        return result

    # Get per-subject beta (from progressive condition)
    prog = session_metrics_df[session_metrics_df['condition'] == 'progressive']
    if 'baseline_beta' not in prog.columns:
        logger.warning("  baseline_beta not found in progressive data")
        return result

    median_beta = prog['baseline_beta'].median()
    logger.info(f"  Median baseline beta: {median_beta:.4f}")

    # Assign subgroups
    metrics_df = session_metrics_df.copy()
    subj_beta = prog.set_index('subject_id')['baseline_beta']
    metrics_df['beta_group'] = metrics_df['subject_id'].map(
        lambda s: 'high_beta' if subj_beta.get(s, median_beta) > median_beta else 'low_beta'
    )

    key_conditions = ['progressive', 'fixed_delta', 'fixed_delta_ssa_resets',
                      'pulsed_progressive', 'pulsed_fixed_delta',
                      'adaptive_protocol', 'ssa_reset_fast', 'ssa_reset_slow']
    key_conditions = [c for c in key_conditions
                      if c in metrics_df['condition'].unique()]

    for group in ['high_beta', 'low_beta']:
        group_data = metrics_df[metrics_df['beta_group'] == group]
        n_subj = group_data['subject_id'].nunique()
        logger.info(f"\n  {group} (n={n_subj}):")

        group_results = {}
        for cond in key_conditions:
            cond_data = group_data[group_data['condition'] == cond]
            if len(cond_data) > 0:
                swa_data = cond_data['session_swa_enhancement'] if 'session_swa_enhancement' in cond_data.columns else cond_data['session_sdre']
                mean_val = float(swa_data.mean())
                std_val = float(swa_data.std())
                group_results[cond] = {
                    'mean_swa_enhancement': mean_val,
                    'std_swa_enhancement': std_val,
                    'n': int(len(swa_data)),
                }
                logger.info(f"    {cond:30s}: SWA_enh={mean_val:+.1f}% +/- {std_val:.1f}%")

        result[group] = group_results

    # Adaptive protocol validation: check if it correctly assigns
    adaptive_data = session_metrics_df[
        session_metrics_df['condition'] == 'adaptive_protocol'
    ]
    if 'adaptive_choice' in metrics_df.columns:
        # Count from the epochs data is better
        pass

    result['median_beta_threshold'] = float(median_beta)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'responder_subgroups.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ─── Verification ─────────────────────────────────────────────────────

def run_verification(
    all_epochs_df: pd.DataFrame,
    session_metrics_df: pd.DataFrame,
) -> Dict:
    """Run verification checks on redesigned study results."""
    logger.info("=" * 70)
    logger.info("VERIFICATION CHECKS")
    logger.info("=" * 70)

    checks = {}

    # 1. All 14 conditions present
    conditions = sorted(all_epochs_df['condition'].unique())
    checks['n_conditions'] = len(conditions)
    checks['conditions'] = conditions
    logger.info(f"  Conditions found: {len(conditions)}")
    for c in conditions:
        n_epochs = len(all_epochs_df[all_epochs_df['condition'] == c])
        n_subj = all_epochs_df[all_epochs_df['condition'] == c]['subject_id'].nunique()
        logger.info(f"    {c:30s}: {n_epochs} epochs, {n_subj} subjects")

    # 2. Pulsed conditions have pulses
    for pulsed_cond in ['pulsed_progressive', 'pulsed_fixed_delta']:
        pulsed_data = all_epochs_df[all_epochs_df['condition'] == pulsed_cond]
        if 'n_pulses' in pulsed_data.columns:
            mean_pulses = float(pulsed_data['n_pulses'].mean())
            mean_duty = float(pulsed_data['pulse_duty_cycle'].mean())
            checks[f'{pulsed_cond}_mean_pulses'] = mean_pulses
            checks[f'{pulsed_cond}_mean_duty_cycle'] = mean_duty
            ok = mean_pulses > 0 and 0.01 < mean_duty < 0.50
            logger.info(f"  {pulsed_cond}: pulses={mean_pulses:.1f}, "
                        f"duty={mean_duty:.3f} {'PASS' if ok else 'WARN'}")
        else:
            logger.warning(f"  {pulsed_cond}: no pulse data found")

    # 3. SSA resets show periodic adaptation drops
    ssa_data = all_epochs_df[all_epochs_df['condition'] == 'fixed_delta_ssa_resets']
    adapt_col = 'adaptation_fast' if 'adaptation_fast' in ssa_data.columns else 'adaptation'
    if adapt_col in ssa_data.columns and len(ssa_data) > 0:
        # Check for at least one drop in adaptation
        mean_adapt_per_epoch = ssa_data.groupby('epoch_idx')[adapt_col].mean()
        adapt_diff = mean_adapt_per_epoch.diff()
        n_drops = int((adapt_diff < -0.01).sum())
        checks['ssa_resets_n_drops'] = n_drops
        logger.info(f"  SSA resets adaptation drops: {n_drops} "
                    f"{'PASS' if n_drops > 0 else 'WARN'}")

    # 4. Adaptive protocol assignment
    if 'baseline_beta' in session_metrics_df.columns:
        adapt_data = session_metrics_df[
            session_metrics_df['condition'] == 'adaptive_protocol'
        ]
        if len(adapt_data) > 0:
            n_total = len(adapt_data)
            logger.info(f"  Adaptive protocol: {n_total} subjects assigned")

    # 5. Sham hierarchy check: no_stim < sham (subthreshold) < active_sham < active
    for metric_name in ['session_swa_enhancement', 'session_sdre']:
        if metric_name not in session_metrics_df.columns:
            continue
        vals = {}
        for cond in ['no_stim', 'sham', 'active_sham', 'fixed_delta']:
            cdata = session_metrics_df[session_metrics_df['condition'] == cond]
            if len(cdata) > 0:
                vals[cond] = float(cdata[metric_name].mean())
        checks[f'sham_hierarchy_{metric_name}'] = vals
        hierarchy_ok = (vals.get('no_stim', 0) <= vals.get('sham', 0)
                        <= vals.get('active_sham', 0) <= vals.get('fixed_delta', 0))
        checks[f'sham_hierarchy_ordered_{metric_name}'] = bool(hierarchy_ok)
        logger.info(f"  Sham hierarchy ({metric_name}): "
                    + " < ".join(f"{k}={v:+.3f}" for k, v in
                                 sorted(vals.items(), key=lambda x: x[1])))
        logger.info(f"    Hierarchy correct: {hierarchy_ok}")

    # 6. SSA sensitivity check: tau_slow variants produce different magnitudes
    ssa_variants = {}
    for cond in ['fixed_delta_ssa_resets', 'ssa_reset_fast', 'ssa_reset_slow']:
        cdata = session_metrics_df[session_metrics_df['condition'] == cond]
        if len(cdata) > 0 and 'session_swa_enhancement' in cdata.columns:
            ssa_variants[cond] = float(cdata['session_swa_enhancement'].mean())
    if ssa_variants:
        checks['ssa_sensitivity_swa'] = ssa_variants
        logger.info(f"  SSA sensitivity (SWA enhancement): "
                    + ", ".join(f"{k}={v:+.1f}%" for k, v in ssa_variants.items()))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'verification.json', 'w') as f:
        json.dump(checks, f, indent=2, default=str)

    return checks


# ─── Main Pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Redesigned Progressive Frequency Descent Study'
    )
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--duration', type=int, default=60,
                        choices=[30, 60, 90],
                        help='Session duration in minutes (default: 60)')
    parser.add_argument('--n-subjects', type=int, default=None,
                        help='Number of subjects (subset for testing)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    session_duration_sec = args.duration * 60.0

    subjects = load_all_subjects(max_subjects=args.n_subjects)
    if not subjects:
        logger.error("No subject data found!")
        return

    n_workers = args.workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    logger.info(f"\nStarting Redesigned Protocol Study")
    logger.info(f"  Subjects: {len(subjects)}")
    logger.info(f"  Conditions: {N_CONDITIONS}")
    logger.info(f"  Session duration: {args.duration} min")
    logger.info(f"  Total sessions: {len(subjects) * N_CONDITIONS}")
    logger.info(f"  Workers: {n_workers}")
    total_t0 = time.time()

    # Phase 1: Run all protocols
    all_epochs_df, session_metrics_df, all_results = run_phase1(
        subjects, session_duration_sec, n_workers=n_workers,
    )

    # Phase 2: Statistical analysis
    stats_report = run_phase2(session_metrics_df)

    # Phase 3: Cross-dataset validation
    cv_result = run_phase3(
        session_metrics_df,
        duration_label=f'{args.duration}min',
    )

    # Phase 4: Responder subgroup analysis
    subgroup_result = run_phase4(session_metrics_df)

    # Verification
    verification = run_verification(all_epochs_df, session_metrics_df)

    # Generate figures
    logger.info("=" * 70)
    logger.info("Generating Figures")
    logger.info("=" * 70)
    try:
        generate_redesigned_figures(results_dir=str(RESULTS_DIR))
    except Exception as e:
        logger.error(f"Figure generation failed: {e}")

    # Summary
    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 70)
    print("REDESIGNED PROTOCOL STUDY — COMPLETE")
    print("=" * 70)
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Sessions completed: {len(subjects) * N_CONDITIONS}")
    print(f"Session duration: {args.duration} min")
    print(f"\nResults saved to: {RESULTS_DIR}")

    # Print main findings
    if 'session_swa_enhancement' in session_metrics_df.columns:
        swa_summary = session_metrics_df.groupby('condition')['session_swa_enhancement'].agg(
            ['mean', 'std']
        ).sort_values('mean', ascending=False)
        print("\nSession SWA Enhancement by condition:")
        print("-" * 60)
        for cond, row in swa_summary.iterrows():
            print(f"  {cond:30s}: {row['mean']:+.1f}% +/- {row['std']:.1f}%")

    summary = session_metrics_df.groupby('condition')['session_sdre'].agg(
        ['mean', 'std']
    ).sort_values('mean', ascending=False)
    print("\nSession SDRE by condition:")
    print("-" * 60)
    for cond, row in summary.iterrows():
        print(f"  {cond:30s}: {row['mean']:+.4f} +/- {row['std']:.4f}")

    # Key contrasts
    print("\nKey targeted contrasts:")
    print("-" * 60)
    for comp in stats_report.get('targeted_contrasts', []):
        if comp.get('metric') != 'session_sdre':
            continue
        sig = "SIG" if comp.get('significant_fdr') else "ns"
        label = comp.get('contrast_label', f"{comp['target']} vs {comp['control']}")
        print(f"  {label:35s}: d={comp['cohens_d']:+.3f}, "
              f"p_fdr={comp.get('p_value_fdr', comp['p_value']):.4f} [{sig}]")

    print("=" * 70)


if __name__ == '__main__':
    main()
