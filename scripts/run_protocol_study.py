"""
Progressive Frequency Descent Entrainment — Protocol Comparison Study.

Compares 7 stimulation protocols using a within-subject repeated-measures
design across 208 subjects from 5 sleep databases.

Uses the Thalamocortical Stuart-Landau Ensemble (TSLE) model, which provides:
- Frequency-selective resonance (Stuart-Landau amplitude dynamics)
- Thalamocortical feedback loop (cortical synchrony drives thalamic T,
  which shifts cortical frequencies toward delta)
- Sleep-state dependence via I_sleep from subject's sleep stage distribution

Phases:
1. Run all protocols: 208 subjects x 7 conditions = 1,456 sessions
2. Statistical analysis: Friedman omnibus + pairwise Wilcoxon + FDR
3. Cross-dataset validation: Sleep-EDF vs. others
4. Per-phase analysis: spectral shifts within progressive protocol

Usage:
    python scripts/run_protocol_study.py [--workers N]
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.thalamocortical_model import ThalamocorticalEnsemble, compute_sdr
from analysis.protocol_comparison import (
    EPOCH_SEC,
    aggregate_protocol_results,
    compute_session_metrics,
    define_protocols,
)
from analysis.statistical_validation import (
    run_protocol_validation,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'protocol_study'

# Model parameters
DEFAULT_N_OSCILLATORS = 64
DEFAULT_COUPLING = 2.0
DEFAULT_NOISE = 0.15     # Reduced from Kuramoto's 0.3 for better forcing SNR
DEFAULT_DT = 0.005       # Halved for Stuart-Landau stability
DEFAULT_FORCING = 0.3    # Weak forcing: TC feedback mechanism dominates over brute-force
DEFAULT_NON_RESPONDER = 0.30

# TSLE-specific parameters
DEFAULT_TAU_T = 10.0          # Thalamocortical loop timescale (seconds)
DEFAULT_ALPHA_TC = 5.0        # Cortical-to-thalamic coupling (strong R-dependence)
DEFAULT_GAMMA = 0.5           # Max frequency shift fraction
DEFAULT_KAPPA = 3.0           # Sigmoid steepness
DEFAULT_T_HALF = 0.3          # Sigmoid midpoint (lower = earlier TC engagement)
DEFAULT_DELTA_LAMBDA = 1.5    # Excitability boost from thalamus
DEFAULT_BETA_EXT = 0.05       # External drive to thalamus (weak; R dominates)
DEFAULT_LAMBDA_BASE = -0.3    # Base excitability (mildly subcritical)


def load_all_subjects() -> Dict[str, pd.DataFrame]:
    """Load all processed subject CSVs."""
    subjects = {}
    for csv_path in sorted(DATA_DIR.glob('*_processed.csv')):
        subject_id = csv_path.stem.replace('_processed', '')
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in
               ['delta_power', 'theta_power', 'alpha_power', 'beta_power']):
            subjects[subject_id] = df
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
    """Extract sleep stage fractions from subject data for I_sleep computation."""
    if 'sleep_stage' not in df.columns:
        # Default: assume light sleep distribution
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


def run_subject_all_conditions(args: tuple) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """Worker: run all 7 protocols for one subject.

    Within-subject design: same model parameters (natural frequencies,
    excitability, forcing mask, TC coupling) for all conditions, but
    different noise realizations per condition (as in a real experiment
    where each condition is tested on a separate night). Subject-level
    parameters are deterministic from the subject seed.
    """
    subject_id, baseline_powers, sleep_stage_fractions = args

    seed_base = hash(subject_id) % (2**31)

    # Define protocols with deterministic sham
    sham_rng = np.random.default_rng(seed_base + 999)
    protocols = define_protocols(rng=sham_rng)

    # Initialize once to get shared model parameters
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
    # Save the subject-level model parameters
    shared_natural_freqs = ref_ensemble.natural_freqs.copy()
    shared_lambda_0 = ref_ensemble.lambda_0.copy()
    shared_forcing_mask = ref_ensemble.forcing_mask.copy()
    shared_alpha_TC = ref_ensemble.alpha_TC
    shared_I_sleep = ref_ensemble.I_sleep

    subject_results = {}
    for cond_idx, (condition, phases) in enumerate(protocols.items()):
        # Fresh ensemble with condition-specific noise seed
        cond_seed = seed_base * 7 + cond_idx * 31
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

        # Restore shared model parameters (within-subject control)
        ensemble.natural_freqs = shared_natural_freqs.copy()
        ensemble.lambda_0 = shared_lambda_0.copy()
        ensemble.forcing_mask = shared_forcing_mask.copy()
        ensemble.alpha_TC = shared_alpha_TC
        ensemble.I_sleep = shared_I_sleep

        # Fresh initial state with condition-specific noise
        ensemble.z = (
            0.1 * ensemble.rng.standard_normal(ensemble.N)
            + 0.1j * ensemble.rng.standard_normal(ensemble.N)
        )
        ensemble.T = 0.0
        ensemble.H = 0.0
        ensemble.A_hab = 0.0
        ensemble._last_forcing_freq = -1.0
        ensemble.t = 0.0
        ensemble._mf_buffer[:] = 0.0
        ensemble._mf_idx = 0
        ensemble._step_counter = 0

        # Baseline epoch (F=0)
        ensemble.run_epoch(EPOCH_SEC, 1.0, 0.0)
        baseline_bp = ensemble.compute_band_powers()
        shared_baseline_sdr = compute_sdr(baseline_bp)

        session_df = ensemble.run_progressive_session(
            baseline_powers=baseline_powers,
            protocol_phases=phases,
            forcing_strength=DEFAULT_FORCING,
            epoch_sec=EPOCH_SEC,
            non_responder_fraction=DEFAULT_NON_RESPONDER,
            baseline_sdr=shared_baseline_sdr,
            skip_init=True,
            sleep_stage_fractions=sleep_stage_fractions,
        )
        session_df['condition'] = condition
        session_df['subject_id'] = subject_id
        subject_results[condition] = session_df

    return subject_id, subject_results


# ─── Phase 1: Run All Protocols ──────────────────────────────────────

def run_phase1(
    subjects: Dict[str, pd.DataFrame],
    n_workers: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Run all 7 protocols for all subjects."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Running All Protocols (208 subjects x 7 conditions)")
    logger.info("=" * 70)

    baselines = {sid: get_baseline_powers(df) for sid, df in subjects.items()}
    sleep_fractions = {sid: get_sleep_stage_fractions(df) for sid, df in subjects.items()}
    tasks = [(sid, baselines[sid], sleep_fractions[sid]) for sid in subjects]

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    logger.info(f"  Subjects: {len(subjects)}")
    logger.info(f"  Conditions: 7")
    logger.info(f"  Total sessions: {len(subjects) * 7}")
    logger.info(f"  Workers: {n_workers}")

    t0 = time.time()
    all_results = {}  # {subject_id: {condition: session_df}}

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, (sid, subject_results) in enumerate(
                pool.imap_unordered(run_subject_all_conditions, tasks)
            ):
                all_results[sid] = subject_results
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    logger.info(f"  Progress: {i+1}/{len(tasks)} subjects "
                                f"({elapsed:.0f}s)")
    else:
        for i, task in enumerate(tasks):
            sid, subject_results = run_subject_all_conditions(task)
            all_results[sid] = subject_results
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                logger.info(f"  Progress: {i+1}/{len(tasks)} subjects "
                            f"({elapsed:.0f}s)")

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

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_epochs_df.to_csv(RESULTS_DIR / 'all_epochs.csv', index=False)
    session_metrics_df.to_csv(RESULTS_DIR / 'session_metrics.csv', index=False)
    logger.info(f"  Saved {len(all_epochs_df)} epoch rows")
    logger.info(f"  Saved {len(session_metrics_df)} session metric rows")

    # Summary
    logger.info("\n  Session SDRE by condition (mean +/- std):")
    summary = session_metrics_df.groupby('condition')['session_sdre'].agg(
        ['mean', 'std']
    ).sort_values('mean', ascending=False)
    for cond, row in summary.iterrows():
        logger.info(f"    {cond:20s}: {row['mean']:+.4f} +/- {row['std']:.4f}")

    return all_epochs_df, session_metrics_df, all_results


# ─── Phase 2: Statistical Analysis ───────────────────────────────────

def run_phase2(session_metrics_df: pd.DataFrame) -> Dict:
    """Run statistical tests on protocol comparison results."""
    logger.info("=" * 70)
    logger.info("PHASE 2: Statistical Analysis")
    logger.info("=" * 70)

    stats_dir = RESULTS_DIR / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)

    report = run_protocol_validation(
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

    logger.info("\n  Progressive vs. each condition (FDR-corrected):")
    for comp in report['pairwise_comparisons']:
        sig = "SIG" if comp.get('significant_fdr') else "ns"
        logger.info(f"    vs {comp['control']:15s}: "
                     f"d={comp['cohens_d']:.3f}, "
                     f"p_fdr={comp.get('p_value_fdr', comp['p_value']):.4f} "
                     f"[{sig}]")

    return report


# ─── Phase 3: Cross-Dataset Validation ───────────────────────────────

def run_phase3(session_metrics_df: pd.DataFrame) -> Dict:
    """Cross-validate: does progressive advantage replicate across datasets?"""
    logger.info("=" * 70)
    logger.info("PHASE 3: Cross-Dataset Validation")
    logger.info("=" * 70)

    metrics_df = session_metrics_df.copy()
    metrics_df['dataset'] = metrics_df['subject_id'].apply(classify_dataset)

    # Discovery: Sleep-EDF; Validation: others
    discovery = metrics_df[metrics_df['dataset'] == 'sleep_edf']
    validation = metrics_df[metrics_df['dataset'] != 'sleep_edf']

    n_disc = discovery['subject_id'].nunique()
    n_val = validation['subject_id'].nunique()
    logger.info(f"  Discovery (Sleep-EDF): {n_disc} subjects")
    logger.info(f"  Validation (others): {n_val} subjects")

    result = {}
    for metric in ['session_sdre', 'cumulative_sleep_depth', 'final_delta_power']:
        disc_prog = discovery[discovery['condition'] == 'progressive'][metric]
        disc_nostim = discovery[discovery['condition'] == 'no_stim'][metric]
        val_prog = validation[validation['condition'] == 'progressive'][metric]
        val_nostim = validation[validation['condition'] == 'no_stim'][metric]

        disc_effect = float(disc_prog.mean() - disc_nostim.mean())
        val_effect = float(val_prog.mean() - val_nostim.mean())

        # Cohen's d for each set
        disc_diff = disc_prog.values - disc_nostim.values[:len(disc_prog)]
        val_diff = val_prog.values - val_nostim.values[:len(val_prog)]

        disc_d = float(np.mean(disc_diff) / (np.std(disc_diff, ddof=1) + 1e-10))
        val_d = float(np.mean(val_diff) / (np.std(val_diff, ddof=1) + 1e-10))

        same_direction = (disc_effect > 0) == (val_effect > 0)

        result[metric] = {
            'discovery_effect': disc_effect,
            'validation_effect': val_effect,
            'discovery_cohens_d': disc_d,
            'validation_cohens_d': val_d,
            'same_direction': bool(same_direction),
            'n_discovery': int(n_disc),
            'n_validation': int(n_val),
        }

        logger.info(f"\n  {metric}:")
        logger.info(f"    Discovery: effect={disc_effect:+.4f}, d={disc_d:.3f}")
        logger.info(f"    Validation: effect={val_effect:+.4f}, d={val_d:.3f}")
        logger.info(f"    Same direction: {same_direction}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'cross_validation.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ─── Phase 4: Per-Phase Analysis ─────────────────────────────────────

def run_phase4(all_epochs_df: pd.DataFrame) -> Dict:
    """Analyze per-phase spectral shifts within progressive protocol."""
    logger.info("=" * 70)
    logger.info("PHASE 4: Per-Phase Analysis (Progressive Protocol)")
    logger.info("=" * 70)

    prog_data = all_epochs_df[all_epochs_df['condition'] == 'progressive'].copy()

    if prog_data.empty:
        logger.warning("No progressive protocol data found!")
        return {}

    result = {}
    phases = prog_data['phase_name'].unique()

    for phase_name in phases:
        phase_data = prog_data[prog_data['phase_name'] == phase_name]

        phase_stats = {
            'n_epochs': int(len(phase_data) / prog_data['subject_id'].nunique()),
            'frequency': float(phase_data['frequency'].iloc[0]),
            'mean_delta_power': float(phase_data['delta_power'].mean()),
            'mean_theta_power': float(phase_data['theta_power'].mean()),
            'mean_alpha_power': float(phase_data['alpha_power'].mean()),
            'mean_beta_power': float(phase_data['beta_power'].mean()),
            'mean_plv': float(phase_data['plv'].mean()),
            'std_plv': float(phase_data['plv'].std()),
            'mean_order_parameter': float(phase_data['order_parameter'].mean()),
            'mean_sdr': float(phase_data['sdr'].mean()),
            'mean_sdre': float(phase_data['sdre'].mean()),
        }
        result[phase_name] = phase_stats

        logger.info(f"\n  {phase_name} ({phase_stats['frequency']:.1f} Hz, "
                     f"{phase_stats['n_epochs']} epochs):")
        logger.info(f"    delta={phase_stats['mean_delta_power']:.3f}, "
                     f"theta={phase_stats['mean_theta_power']:.3f}, "
                     f"alpha={phase_stats['mean_alpha_power']:.3f}, "
                     f"beta={phase_stats['mean_beta_power']:.3f}")
        logger.info(f"    PLV={phase_stats['mean_plv']:.3f}, "
                     f"SDR={phase_stats['mean_sdr']:.3f}, "
                     f"SDRE={phase_stats['mean_sdre']:.3f}")

    # Check expected spectral shifts
    expectations = {
        'alpha_10hz': 'alpha_power should be elevated',
        'alpha_8.5hz': 'alpha->theta transition expected',
        'theta_6hz': 'theta_power should increase',
        'delta_2hz': 'delta_power should increase',
    }

    logger.info("\n  Expected spectral directions:")
    for phase_name, expectation in expectations.items():
        if phase_name in result:
            logger.info(f"    {phase_name}: {expectation}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'phase_analysis.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ─── Main Pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Progressive Frequency Descent Protocol Comparison Study'
    )
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subjects = load_all_subjects()
    if not subjects:
        logger.error("No subject data found!")
        return

    logger.info(f"\nStarting Protocol Comparison Study")
    logger.info(f"  Subjects: {len(subjects)}")
    logger.info(f"  Conditions: 7")
    logger.info(f"  Total sessions: {len(subjects) * 7}")
    total_t0 = time.time()

    # Phase 1: Run all protocols
    all_epochs_df, session_metrics_df, all_results = run_phase1(
        subjects, n_workers=args.workers
    )

    # Phase 2: Statistical analysis
    stats_report = run_phase2(session_metrics_df)

    # Phase 3: Cross-dataset validation
    cv_result = run_phase3(session_metrics_df)

    # Phase 4: Per-phase analysis
    phase_result = run_phase4(all_epochs_df)

    # Summary
    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 70)
    print("PROTOCOL COMPARISON STUDY — COMPLETE")
    print("=" * 70)
    print(f"\nTotal runtime: {total_elapsed:.1f}s")
    print(f"Sessions completed: {len(subjects) * 7}")
    print(f"\nResults saved to: {RESULTS_DIR}")

    # Print main finding
    summary = session_metrics_df.groupby('condition')['session_sdre'].agg(
        ['mean', 'std']
    ).sort_values('mean', ascending=False)
    print("\nSession SDRE by condition:")
    print("-" * 50)
    for cond, row in summary.iterrows():
        print(f"  {cond:20s}: {row['mean']:+.4f} +/- {row['std']:.4f}")

    # Progressive vs. no-stim effect
    prog_sdre = session_metrics_df[
        session_metrics_df['condition'] == 'progressive'
    ]['session_sdre']
    nostim_sdre = session_metrics_df[
        session_metrics_df['condition'] == 'no_stim'
    ]['session_sdre']
    diff = prog_sdre.values - nostim_sdre.values
    d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
    print(f"\nProgressive vs. No-stim:")
    print(f"  Paired Cohen's d = {d:.3f}")
    print(f"  Mean difference = {np.mean(diff):+.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
