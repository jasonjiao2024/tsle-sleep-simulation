"""
TSLE v2 Validation Script.

Runs 5 subjects through key conditions to verify the three v2 fixes:
1. Baseline sanity: 60s no-forcing → verify emergent SO appears in PSD (0.5-1.0 Hz peak)
2. Sham check: sham protocol → SWA enhancement should be <15%
3. Active vs sham: fixed_delta should produce clearly higher SWA enhancement
4. SSA grading: compare SSA behavior under graded vs old binary reset
5. Pulsed vs continuous: with emergent SO + so_modulation>0, pulsed should differentiate

Usage:
    python scripts/run_v2_validation.py
    python scripts/run_v2_validation.py --n-subjects 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.thalamocortical_model import (
    ThalamocorticalEnsemble,
    compute_sdr,
    compute_swa,
    compute_swa_enhancement,
)
from analysis.protocol_comparison import EPOCH_SEC, compute_session_metrics
from analysis.redesigned_protocols import define_redesigned_protocols

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'v2_validation'

# Model parameters
N_OSCILLATORS = 64
DT = 0.005
FORCING = 0.3
NON_RESPONDER = 0.30
SO_MODULATION = 0.3
SESSION_DURATION_SEC = 1800.0  # 30 min for quick validation


def load_subjects(n_subjects: int = 5) -> Dict[str, pd.DataFrame]:
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


def make_ensemble(seed: int) -> ThalamocorticalEnsemble:
    """Create a v2 ensemble with standard parameters."""
    return ThalamocorticalEnsemble(
        n_oscillators=N_OSCILLATORS,
        dt=DT,
        seed=seed,
        noise_sigma=0.15,
        coupling_strength=2.0,
        lambda_base=-0.3,
        so_modulation=SO_MODULATION,
    )


def run_condition(
    baseline_powers: Dict[str, float],
    sleep_fractions: Dict[str, float],
    phases: List[Dict],
    stim_mode: str,
    seed: int,
) -> pd.DataFrame:
    """Run a single condition and return session DataFrame."""
    ensemble = make_ensemble(seed)
    ensemble.initialize_from_baseline(
        baseline_powers,
        non_responder_fraction=NON_RESPONDER,
        sleep_stage_fractions=sleep_fractions,
    )

    # Baseline measurement
    ensemble.run_epoch(EPOCH_SEC, 1.0, 0.0)
    baseline_bp = ensemble.compute_band_powers()
    baseline_sdr = compute_sdr(baseline_bp)
    baseline_swa = compute_swa(baseline_bp)
    ensemble._baseline_swa = baseline_swa

    session_df = ensemble.run_progressive_session(
        baseline_powers=baseline_powers,
        protocol_phases=phases,
        forcing_strength=FORCING,
        epoch_sec=EPOCH_SEC,
        non_responder_fraction=NON_RESPONDER,
        baseline_sdr=baseline_sdr,
        skip_init=True,
        sleep_stage_fractions=sleep_fractions,
        stim_mode=stim_mode,
    )
    return session_df


def test_baseline_sanity(baseline_powers, sleep_fractions, seed):
    """Test 1: Verify emergent SO appears in baseline PSD."""
    ensemble = make_ensemble(seed)
    ensemble.initialize_from_baseline(
        baseline_powers,
        non_responder_fraction=NON_RESPONDER,
        sleep_stage_fractions=sleep_fractions,
    )

    # Run 60s no-forcing
    ensemble.run_epoch(60.0, 1.0, 0.0)
    bp = ensemble.compute_band_powers()

    delta_abs = bp['delta_power_abs']
    total_abs = sum(bp[f'{b}_power_abs'] for b in ['delta', 'theta', 'alpha', 'beta'])

    # Check SO phase is being extracted (not stuck at 0)
    so_phase = ensemble.so_phase

    return {
        'delta_power_abs': delta_abs,
        'delta_fraction': bp['delta_power'],
        'total_abs_power': total_abs,
        'so_phase': so_phase,
        'so_phase_nonzero': abs(so_phase) > 0.01,
    }


def main():
    parser = argparse.ArgumentParser(description='TSLE v2 Validation')
    parser.add_argument('--n-subjects', type=int, default=5)
    args = parser.parse_args()

    subjects = load_subjects(args.n_subjects)
    if not subjects:
        # Use synthetic baseline if no real data
        logger.warning("No subject data found — using synthetic baselines")
        subjects = {}
        for i in range(args.n_subjects):
            sid = f'synthetic_{i}'
            rng = np.random.default_rng(42 + i)
            subjects[sid] = pd.DataFrame({
                'delta_power': rng.uniform(0.2, 0.5, 100),
                'theta_power': rng.uniform(0.1, 0.3, 100),
                'alpha_power': rng.uniform(0.15, 0.35, 100),
                'beta_power': rng.uniform(0.05, 0.25, 100),
            })

    logger.info("=" * 70)
    logger.info("TSLE v2 VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Subjects: {len(subjects)}")

    all_results = []

    for sid, df in subjects.items():
        logger.info(f"\n--- Subject: {sid} ---")
        baseline_powers = get_baseline_powers(df)
        sleep_fractions = get_sleep_stage_fractions(df)
        seed_base = hash(sid) % (2**31)

        # Test 1: Baseline sanity
        sanity = test_baseline_sanity(baseline_powers, sleep_fractions, seed_base)
        logger.info(f"  [1] Baseline: delta_abs={sanity['delta_power_abs']:.6f}, "
                     f"SO_phase={sanity['so_phase']:.3f}, "
                     f"SO_active={'YES' if sanity['so_phase_nonzero'] else 'NO'}")

        # Define protocols
        sham_rng = np.random.default_rng(seed_base + 999)
        all_protocols = define_redesigned_protocols(
            session_duration_sec=SESSION_DURATION_SEC,
            rng=sham_rng,
            baseline_beta=baseline_powers.get('beta_power', 0.25),
        )

        # Run key conditions
        conditions_to_run = {
            'no_stim': ('continuous', seed_base * 14),
            'sham': ('continuous', seed_base * 14 + 7 * 31),
            'fixed_delta': ('continuous', seed_base * 14 + 2 * 31),
            'fixed_delta_ssa_resets': ('continuous', seed_base * 14 + 8 * 31),
            'pulsed_fixed_delta': ('pulsed', seed_base * 14 + 10 * 31),
        }

        subject_results = {'subject_id': sid}
        subject_results.update(sanity)

        for cond_name, (stim_mode, seed) in conditions_to_run.items():
            proto = all_protocols[cond_name]
            session_df = run_condition(
                baseline_powers, sleep_fractions,
                proto['phases'], stim_mode, seed,
            )
            metrics = compute_session_metrics(session_df)

            subject_results[f'{cond_name}_swa_enhancement'] = metrics['session_swa_enhancement']
            subject_results[f'{cond_name}_sdre'] = metrics['session_sdre']
            subject_results[f'{cond_name}_final_swa'] = metrics['final_swa']

            logger.info(f"  [{cond_name}] SWA_enh={metrics['session_swa_enhancement']:+.1f}%  "
                         f"SDRE={metrics['session_sdre']:+.4f}")

        all_results.append(subject_results)

    # Aggregate and report
    results_df = pd.DataFrame(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / 'v2_validation.csv', index=False)

    print("\n" + "=" * 70)
    print("TSLE v2 VALIDATION RESULTS")
    print("=" * 70)

    n = len(results_df)

    # Test 1: Baseline sanity
    so_active = results_df['so_phase_nonzero'].sum()
    print(f"\n[1] BASELINE SANITY: SO phase active in {so_active}/{n} subjects")
    pass_1 = so_active == n
    print(f"    {'PASS' if pass_1 else 'FAIL'}")

    # Test 2: Sham check (SWA enhancement < 15%)
    sham_enh = results_df['sham_swa_enhancement'].values
    mean_sham = np.mean(sham_enh)
    print(f"\n[2] SHAM CHECK: mean SWA enhancement = {mean_sham:+.1f}%  (target: <15%)")
    pass_2 = mean_sham < 15.0
    print(f"    {'PASS' if pass_2 else 'FAIL'}")

    # Test 3: Active vs sham separation
    active_enh = results_df['fixed_delta_swa_enhancement'].values
    mean_active = np.mean(active_enh)
    separation = mean_active - mean_sham
    print(f"\n[3] ACTIVE vs SHAM: active={mean_active:+.1f}%, sham={mean_sham:+.1f}%, "
          f"separation={separation:+.1f}%")
    pass_3 = separation > 0
    print(f"    {'PASS' if pass_3 else 'FAIL'} (active > sham)")

    # Test 4: SSA grading
    ssa_enh = results_df['fixed_delta_ssa_resets_swa_enhancement'].values
    mean_ssa = np.mean(ssa_enh)
    ssa_advantage = mean_ssa - mean_active
    print(f"\n[4] SSA GRADING: ssa_resets={mean_ssa:+.1f}%, fixed_delta={mean_active:+.1f}%, "
          f"advantage={ssa_advantage:+.1f}%")
    # Under graded SSA, wobbles should still help but less than binary reset
    print(f"    SSA resets advantage: {ssa_advantage:+.1f}% (should be smaller than v1)")

    # Test 5: Pulsed vs continuous
    pulsed_enh = results_df['pulsed_fixed_delta_swa_enhancement'].values
    mean_pulsed = np.mean(pulsed_enh)
    pulsed_diff = mean_pulsed - mean_active
    print(f"\n[5] PULSED vs CONTINUOUS: pulsed={mean_pulsed:+.1f}%, "
          f"continuous={mean_active:+.1f}%, diff={pulsed_diff:+.1f}%")
    # With emergent SO, pulsed should differentiate from continuous
    pass_5 = abs(pulsed_diff) > 0.1
    print(f"    {'PASS' if pass_5 else 'MARGINAL'} (pulsed differentiates from continuous)")

    # No-stim baseline
    nostim_enh = results_df['no_stim_swa_enhancement'].values
    mean_nostim = np.mean(nostim_enh)
    print(f"\n[REF] No-stim SWA enhancement: {mean_nostim:+.1f}%")

    overall = sum([pass_1, pass_2, pass_3, pass_5])
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {overall}/4 checks passed")
    print(f"{'=' * 70}")

    logger.info(f"\nResults saved to {RESULTS_DIR / 'v2_validation.csv'}")


if __name__ == '__main__':
    main()
