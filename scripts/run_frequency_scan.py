"""
Frequency Resonance Discovery Pipeline.

Runs a systematic frequency-response mapping using the Kuramoto entrainment
model across 208 subjects from 5 sleep datasets to identify the precise
optimal entrainment frequency.

Phases:
1. Coarse scan: 2.0-14.0 Hz in 0.25 Hz steps (49 frequencies x 208 subjects)
2. Fine scan: peak +/- 1 Hz in 0.05 Hz steps (41 frequencies x 208 subjects)
3. Cross-validation: discover on Sleep-EDF, validate on CAP+DREAMS+HMC+SLPDB
4. IAF analysis: express optimal as IAF offset with CI
5. Sensitivity: test robustness across F and N parameter variations

Usage:
    python scripts/run_frequency_scan.py [--phase N] [--workers N]
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

from analysis.kuramoto_entrainment import KuramotoEnsemble, compute_sdr
from analysis.frequency_resonance import (
    bootstrap_peak_ci,
    compute_effect_sizes,
    compute_iaf_offsets,
    cross_validate_peak,
    estimate_iaf,
    estimate_population_peak,
    permutation_test_peak_specificity,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'frequency_scan'

# Default model parameters
DEFAULT_N_OSCILLATORS = 64
DEFAULT_COUPLING = 2.0
DEFAULT_NOISE = 0.3
DEFAULT_FORCING = 0.10
DEFAULT_NON_RESPONDER = 0.30
DEFAULT_WARMUP = 5.0
DEFAULT_MEASUREMENT = 30.0


def load_all_subjects() -> Dict[str, pd.DataFrame]:
    """Load all processed subject CSVs."""
    subjects = {}
    for csv_path in sorted(DATA_DIR.glob('*_processed.csv')):
        subject_id = csv_path.stem.replace('_processed', '')
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in ['delta_power', 'theta_power', 'alpha_power', 'beta_power']):
            subjects[subject_id] = df
    logger.info(f"Loaded {len(subjects)} subjects")
    return subjects


def get_baseline_powers(df: pd.DataFrame) -> Dict[str, float]:
    """Extract baseline band powers (mean of first 10 epochs or Wake/N1)."""
    if 'sleep_stage' in df.columns:
        baseline = df[df['sleep_stage'].isin(['Wake', 'W', 'N1', '1'])]
        if len(baseline) >= 3:
            return baseline[['delta_power', 'theta_power', 'alpha_power', 'beta_power']].mean().to_dict()
    # Fallback: first 10 epochs
    n = min(10, len(df))
    return df.iloc[:n][['delta_power', 'theta_power', 'alpha_power', 'beta_power']].mean().to_dict()


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
    else:
        return 'other'


def scan_single_subject(args: tuple) -> Tuple[str, pd.DataFrame]:
    """Worker function for parallel frequency scanning."""
    subject_id, baseline_powers, frequencies, params = args

    ensemble = KuramotoEnsemble(
        n_oscillators=params.get('n_oscillators', DEFAULT_N_OSCILLATORS),
        coupling_strength=params.get('coupling', DEFAULT_COUPLING),
        noise_sigma=params.get('noise', DEFAULT_NOISE),
        dt=0.01,
        seed=hash(subject_id) % (2**31),
    )

    result_df = ensemble.frequency_scan(
        baseline_powers=baseline_powers,
        test_frequencies=frequencies,
        forcing_strength=params.get('forcing', DEFAULT_FORCING),
        warmup_sec=params.get('warmup', DEFAULT_WARMUP),
        measurement_sec=params.get('measurement', DEFAULT_MEASUREMENT),
        non_responder_fraction=params.get('non_responder', DEFAULT_NON_RESPONDER),
    )

    result_df['subject_id'] = subject_id

    return subject_id, result_df


# ─── Phase 1: Coarse Scan ─────────────────────────────────────────────

def run_phase1(subjects: Dict[str, pd.DataFrame], n_workers: int = None) -> pd.DataFrame:
    """Coarse frequency scan: 2.0-14.0 Hz in 0.25 Hz steps."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Coarse Frequency Scan (2.0-14.0 Hz, 0.25 Hz steps)")
    logger.info("=" * 60)

    frequencies = list(np.arange(2.0, 14.25, 0.25))
    logger.info(f"  Frequencies: {len(frequencies)} ({frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz)")
    logger.info(f"  Subjects: {len(subjects)}")

    baselines = {sid: get_baseline_powers(df) for sid, df in subjects.items()}
    params = {}

    tasks = [
        (sid, baselines[sid], frequencies, params)
        for sid in subjects
    ]

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    logger.info(f"  Workers: {n_workers}")
    t0 = time.time()

    all_results = {}
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, (sid, result_df) in enumerate(pool.imap_unordered(scan_single_subject, tasks)):
                all_results[sid] = result_df
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    logger.info(f"  Progress: {i+1}/{len(tasks)} subjects ({elapsed:.0f}s)")
    else:
        for i, task in enumerate(tasks):
            sid, result_df = scan_single_subject(task)
            all_results[sid] = result_df
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                logger.info(f"  Progress: {i+1}/{len(tasks)} subjects ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"  Phase 1 complete in {elapsed:.1f}s")

    # Aggregate: mean SDRE per frequency across subjects
    all_dfs = pd.concat(all_results.values(), ignore_index=True)
    agg = all_dfs.groupby('frequency').agg({
        'sdre': ['mean', 'std', 'count'],
        'delta_power': 'mean',
        'theta_power': 'mean',
        'alpha_power': 'mean',
        'beta_power': 'mean',
        'plv': 'mean',
        'order_parameter': 'mean',
        'sdr': 'mean',
    }).reset_index()
    agg.columns = [
        'frequency', 'sdre_mean', 'sdre_std', 'n_subjects',
        'delta_power', 'theta_power', 'alpha_power', 'beta_power',
        'plv', 'order_parameter', 'sdr',
    ]

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(RESULTS_DIR / 'phase1_coarse_scan.csv', index=False)
    all_dfs.to_csv(RESULTS_DIR / 'phase1_coarse_scan_all_subjects.csv', index=False)

    # Find coarse peak
    peak_idx = agg['sdre_mean'].idxmax()
    peak_freq = agg.loc[peak_idx, 'frequency']
    peak_sdre = agg.loc[peak_idx, 'sdre_mean']
    logger.info(f"  Coarse peak: {peak_freq:.2f} Hz (SDRE = {peak_sdre:.4f})")

    return agg, all_results


# ─── Phase 2: Fine Scan ───────────────────────────────────────────────

def run_phase2(
    subjects: Dict[str, pd.DataFrame],
    coarse_peak: float,
    n_workers: int = None,
) -> pd.DataFrame:
    """Fine frequency scan: peak +/- 1 Hz in 0.05 Hz steps."""
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Fine Frequency Scan ({coarse_peak-1:.2f}-{coarse_peak+1:.2f} Hz, 0.05 Hz steps)")
    logger.info("=" * 60)

    frequencies = list(np.arange(coarse_peak - 1.0, coarse_peak + 1.05, 0.05))
    frequencies = [round(f, 2) for f in frequencies if f >= 0.5]
    logger.info(f"  Frequencies: {len(frequencies)} ({frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz)")

    baselines = {sid: get_baseline_powers(df) for sid, df in subjects.items()}
    params = {}

    tasks = [
        (sid, baselines[sid], frequencies, params)
        for sid in subjects
    ]

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    logger.info(f"  Workers: {n_workers}")
    t0 = time.time()

    all_results = {}
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, (sid, result_df) in enumerate(pool.imap_unordered(scan_single_subject, tasks)):
                all_results[sid] = result_df
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    logger.info(f"  Progress: {i+1}/{len(tasks)} subjects ({elapsed:.0f}s)")
    else:
        for i, task in enumerate(tasks):
            sid, result_df = scan_single_subject(task)
            all_results[sid] = result_df
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                logger.info(f"  Progress: {i+1}/{len(tasks)} subjects ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"  Phase 2 complete in {elapsed:.1f}s")

    all_dfs = pd.concat(all_results.values(), ignore_index=True)
    agg = all_dfs.groupby('frequency').agg({
        'sdre': ['mean', 'std'],
        'delta_power': 'mean',
        'theta_power': 'mean',
        'alpha_power': 'mean',
        'beta_power': 'mean',
        'plv': 'mean',
        'order_parameter': 'mean',
    }).reset_index()
    agg.columns = [
        'frequency', 'sdre_mean', 'sdre_std',
        'delta_power', 'theta_power', 'alpha_power', 'beta_power',
        'plv', 'order_parameter',
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(RESULTS_DIR / 'phase2_fine_scan.csv', index=False)
    all_dfs.to_csv(RESULTS_DIR / 'phase2_fine_scan_all_subjects.csv', index=False)

    # Gaussian fit for sub-grid precision
    peak_info = estimate_population_peak(agg, metric_col='sdre_mean')
    logger.info(f"  Fine peak (Gaussian fit): {peak_info['peak_freq']:.3f} Hz")
    logger.info(f"  R-squared: {peak_info['r_squared']:.4f}")

    # Bootstrap CI
    ci_info = bootstrap_peak_ci(all_results, metric_col='sdre')
    logger.info(f"  Bootstrap peak: {ci_info['peak_freq']:.3f} Hz "
                f"(95% CI: [{ci_info['ci_low']:.3f}, {ci_info['ci_high']:.3f}])")

    return agg, all_results, peak_info, ci_info


# ─── Phase 3: Cross-Validation ────────────────────────────────────────

def run_phase3(
    all_results: Dict[str, pd.DataFrame],
) -> Dict:
    """Cross-validate: discover on Sleep-EDF, validate on others."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Cross-Dataset Validation")
    logger.info("=" * 60)

    discovery = {}
    validation = {}
    for sid, df in all_results.items():
        dataset = classify_dataset(sid)
        if dataset == 'sleep_edf':
            discovery[sid] = df
        else:
            validation[sid] = df

    logger.info(f"  Discovery set (Sleep-EDF): {len(discovery)} subjects")
    logger.info(f"  Validation set (CAP+DREAMS+HMC+SLPDB+other): {len(validation)} subjects")

    if not validation:
        logger.warning("  No validation subjects found!")
        return {'error': 'no_validation_subjects'}

    cv_result = cross_validate_peak(discovery, validation)

    logger.info(f"  Discovery peak: {cv_result['discovery_peak']:.3f} Hz "
                f"(CI: {cv_result['discovery_ci']})")
    logger.info(f"  Validation peak: {cv_result['validation_peak']:.3f} Hz "
                f"(CI: {cv_result['validation_ci']})")
    logger.info(f"  CI overlap (replicates): {cv_result['replicates']}")
    logger.info(f"  Peak difference: {cv_result['peak_difference_hz']:.3f} Hz")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'phase3_validation.json', 'w') as f:
        json.dump(cv_result, f, indent=2, default=str)

    return cv_result


# ─── Phase 4: IAF Analysis ────────────────────────────────────────────

def run_phase4(
    all_results: Dict[str, pd.DataFrame],
    subjects: Dict[str, pd.DataFrame],
    peak_freq: float,
) -> pd.DataFrame:
    """Express optimal frequency as IAF offset."""
    logger.info("=" * 60)
    logger.info("PHASE 4: IAF-Relative Analysis")
    logger.info("=" * 60)

    baselines = {sid: get_baseline_powers(df) for sid, df in subjects.items()}

    iaf_df = compute_iaf_offsets(
        subject_scan_results=all_results,
        subject_baselines=baselines,
        peak_freq=peak_freq,
    )

    mean_offset = iaf_df['population_offset_from_iaf'].mean()
    std_offset = iaf_df['population_offset_from_iaf'].std()
    mean_iaf = iaf_df['iaf'].mean()

    logger.info(f"  Mean IAF: {mean_iaf:.2f} Hz")
    logger.info(f"  Population peak: {peak_freq:.3f} Hz")
    logger.info(f"  Mean offset from IAF: {mean_offset:.3f} +/- {std_offset:.3f} Hz")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    iaf_df.to_csv(RESULTS_DIR / 'phase4_iaf_analysis.csv', index=False)

    iaf_summary = {
        'population_peak_freq': peak_freq,
        'mean_iaf': float(mean_iaf),
        'std_iaf': float(iaf_df['iaf'].std()),
        'mean_offset_from_iaf': float(mean_offset),
        'std_offset_from_iaf': float(std_offset),
        'n_subjects': len(iaf_df),
    }
    with open(RESULTS_DIR / 'phase4_iaf_summary.json', 'w') as f:
        json.dump(iaf_summary, f, indent=2)

    return iaf_df, iaf_summary


# ─── Phase 5: Sensitivity Analysis ────────────────────────────────────

def run_phase5(
    subjects: Dict[str, pd.DataFrame],
    coarse_peak: float,
    n_workers: int = None,
) -> Dict:
    """Test sensitivity to forcing strength F and oscillator count N."""
    logger.info("=" * 60)
    logger.info("PHASE 5: Sensitivity Analysis")
    logger.info("=" * 60)

    forcing_values = [0.05, 0.10, 0.15, 0.20]
    n_oscillator_values = [32, 64, 128]

    # Use fine-scan range around coarse peak
    frequencies = list(np.arange(coarse_peak - 1.0, coarse_peak + 1.05, 0.1))
    frequencies = [round(f, 1) for f in frequencies if f >= 0.5]

    # Use a subset of subjects for sensitivity (all of them)
    baselines = {sid: get_baseline_powers(df) for sid, df in subjects.items()}

    results = {}

    for F in forcing_values:
        for N in n_oscillator_values:
            label = f"F={F:.2f}_N={N}"
            logger.info(f"  Running {label}...")

            params = {
                'n_oscillators': N,
                'forcing': F,
            }

            tasks = [
                (sid, baselines[sid], frequencies, params)
                for sid in subjects
            ]

            t0 = time.time()
            scan_results = {}

            if n_workers is None:
                n_workers_eff = max(1, cpu_count() - 1)
            else:
                n_workers_eff = n_workers

            if n_workers_eff > 1:
                with Pool(n_workers_eff) as pool:
                    for sid, result_df in pool.imap_unordered(scan_single_subject, tasks):
                        scan_results[sid] = result_df
            else:
                for task in tasks:
                    sid, result_df = scan_single_subject(task)
                    scan_results[sid] = result_df

            # Find peak
            all_dfs = pd.concat(scan_results.values(), ignore_index=True)
            agg = all_dfs.groupby('frequency')['sdre'].mean().reset_index()
            peak_idx = agg['sdre'].idxmax()
            peak_freq = float(agg.loc[peak_idx, 'frequency'])
            peak_sdre = float(agg.loc[peak_idx, 'sdre'])

            elapsed = time.time() - t0
            logger.info(f"    Peak: {peak_freq:.2f} Hz, SDRE: {peak_sdre:.4f} ({elapsed:.0f}s)")

            results[label] = {
                'forcing': F,
                'n_oscillators': N,
                'peak_freq': peak_freq,
                'peak_sdre': peak_sdre,
            }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'phase5_sensitivity.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    peak_freqs = [r['peak_freq'] for r in results.values()]
    logger.info(f"  Peak frequency range: {min(peak_freqs):.2f} - {max(peak_freqs):.2f} Hz")
    logger.info(f"  Peak frequency std: {np.std(peak_freqs):.3f} Hz")

    return results


# ─── Main Pipeline ────────────────────────────────────────────────────

def run_statistics(
    all_results_fine: Dict[str, pd.DataFrame],
    peak_freq: float,
    ci_info: Dict,
) -> Dict:
    """Run statistical tests on the results."""
    logger.info("=" * 60)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("=" * 60)

    # Effect sizes at peak
    effects = compute_effect_sizes(all_results_fine, peak_freq)
    logger.info(f"  SDRE at peak: {effects['mean_sdre']:.4f} +/- {effects['std_sdre']:.4f}")
    logger.info(f"  Cohen's d (SDRE): {effects['cohens_d_sdre']:.3f}")
    logger.info(f"  95% CI: [{effects['sdre_ci_low']:.4f}, {effects['sdre_ci_high']:.4f}]")

    # Permutation test
    perm = permutation_test_peak_specificity(all_results_fine, peak_freq)
    logger.info(f"  Permutation test p-value: {perm['p_value']:.6f}")

    stats = {
        'peak_freq': peak_freq,
        'bootstrap_ci': [ci_info['ci_low'], ci_info['ci_high']],
        'effect_sizes': effects,
        'permutation_test': perm,
    }

    (RESULTS_DIR / 'statistics').mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'statistics' / 'statistical_report.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Frequency Resonance Discovery Pipeline')
    parser.add_argument('--phase', type=int, default=0,
                        help='Run specific phase (1-5), 0=all')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subjects = load_all_subjects()
    if not subjects:
        logger.error("No subject data found!")
        return

    # Track variables across phases
    coarse_peak = None
    fine_peak = None
    fine_results = None
    ci_info = None

    # Phase 1
    if args.phase == 0 or args.phase == 1:
        coarse_agg, coarse_results = run_phase1(subjects, n_workers=args.workers)
        coarse_peak = float(coarse_agg.loc[coarse_agg['sdre_mean'].idxmax(), 'frequency'])

    if coarse_peak is None and (RESULTS_DIR / 'phase1_coarse_scan.csv').exists():
        coarse_agg = pd.read_csv(RESULTS_DIR / 'phase1_coarse_scan.csv')
        coarse_peak = float(coarse_agg.loc[coarse_agg['sdre_mean'].idxmax(), 'frequency'])

    if args.phase == 1:
        return

    # Phase 2
    if args.phase == 0 or args.phase == 2:
        if coarse_peak is None:
            logger.error("Phase 2 requires Phase 1 results. Run Phase 1 first.")
            return
        fine_agg, fine_results, peak_info, ci_info = run_phase2(
            subjects, coarse_peak, n_workers=args.workers
        )
        fine_peak = peak_info['peak_freq']

    if fine_peak is None and (RESULTS_DIR / 'phase2_fine_scan.csv').exists():
        fine_agg = pd.read_csv(RESULTS_DIR / 'phase2_fine_scan.csv')
        fine_peak = float(fine_agg.loc[fine_agg['sdre_mean'].idxmax(), 'frequency'])

    if args.phase == 2:
        # Run statistics for phase 2
        if fine_results and ci_info:
            run_statistics(fine_results, fine_peak, ci_info)
        return

    # Phase 3
    if args.phase == 0 or args.phase == 3:
        if fine_results is None:
            logger.warning("Phase 3 requires Phase 2 results in memory. Run phases 0 or 2+3.")
        else:
            cv_result = run_phase3(fine_results)

    if args.phase == 3:
        return

    # Phase 4
    if args.phase == 0 or args.phase == 4:
        if fine_results is None:
            logger.warning("Phase 4 requires Phase 2 results in memory.")
        else:
            iaf_df, iaf_summary = run_phase4(fine_results, subjects, fine_peak)

    if args.phase == 4:
        return

    # Phase 5
    if args.phase == 0 or args.phase == 5:
        if coarse_peak is None:
            logger.error("Phase 5 requires Phase 1 results.")
            return
        sensitivity = run_phase5(subjects, coarse_peak, n_workers=args.workers)

    if args.phase == 5:
        return

    # Run statistics if we have fine results (phase 0 = all)
    if fine_results and ci_info:
        stats = run_statistics(fine_results, fine_peak, ci_info)

        print("\n" + "=" * 70)
        print("MAIN FINDING")
        print("=" * 70)
        print(f"The Kuramoto model predicts optimal entrainment at "
              f"{fine_peak:.2f} Hz")
        print(f"  95% CI: [{ci_info['ci_low']:.2f}, {ci_info['ci_high']:.2f}]")
        print(f"  SDRE = {stats['effect_sizes']['mean_sdre']:.4f}")
        print(f"  Cohen's d = {stats['effect_sizes']['cohens_d_sdre']:.3f}")
        print(f"  Permutation p = {stats['permutation_test']['p_value']:.6f}")
        print("=" * 70)


if __name__ == '__main__':
    main()
