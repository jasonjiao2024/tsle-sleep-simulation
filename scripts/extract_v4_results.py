"""
Extract v4 experiment results and print all values needed for manuscript update.
Run this after the full 208-subject experiment completes.

Usage:
    python scripts/extract_v4_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'redesigned_study'


def main():
    # Load data
    metrics_path = RESULTS_DIR / 'session_metrics.csv'
    stats_path = RESULTS_DIR / 'statistics' / 'redesigned_statistical_report.json'
    cv_path = RESULTS_DIR / 'cross_validation.json'
    subgroup_path = RESULTS_DIR / 'responder_subgroups.json'
    verification_path = RESULTS_DIR / 'verification.json'

    if not metrics_path.exists():
        print("ERROR: session_metrics.csv not found. Run experiment first.")
        sys.exit(1)

    df = pd.read_csv(metrics_path)
    n_sessions = len(df)
    n_subjects = df['subject_id'].nunique()
    n_conditions = df['condition'].nunique()
    print(f"Loaded: {n_sessions} sessions, {n_subjects} subjects, {n_conditions} conditions")
    print(f"Epoch rows: {n_subjects * n_conditions * 120}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # 1. SWA Enhancement by condition (Table 2)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TABLE: SWA Enhancement by Condition")
    print("=" * 70)
    if 'session_swa_enhancement' in df.columns:
        swa = df.groupby('condition')['session_swa_enhancement'].agg(['mean', 'std'])
        swa = swa.sort_values('mean', ascending=False)
        for cond, row in swa.iterrows():
            print(f"  {cond:30s}: {row['mean']:+6.1f} ± {row['std']:5.1f}%")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # 2. Statistics report
    # ═══════════════════════════════════════════════════════════════════
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        print("=" * 70)
        print("OMNIBUS: Kendall's W values")
        print("=" * 70)
        for metric, result in stats['omnibus_tests'].items():
            w = result.get('kendalls_w', 'N/A')
            print(f"  {metric:30s}: W = {w:.4f}" if isinstance(w, float) else f"  {metric}: {w}")
        print()

        print("=" * 70)
        print("TARGETED CONTRASTS (SWA enhancement)")
        print("=" * 70)
        for comp in stats.get('targeted_contrasts', []):
            if comp.get('metric') != 'session_swa_enhancement':
                continue
            label = comp.get('contrast_label', f"{comp['target']} vs {comp['control']}")
            sig = "***" if comp.get('p_value_fdr', 1) < 0.001 else \
                  "**" if comp.get('p_value_fdr', 1) < 0.01 else \
                  "*" if comp.get('p_value_fdr', 1) < 0.05 else "n.s."
            print(f"  {label:40s}: d={comp['cohens_d']:+.3f}, "
                  f"p_fdr={comp.get('p_value_fdr', comp['p_value']):.4f} [{sig}]")
        print()

        # Multi-metric effect sizes (S6)
        print("=" * 70)
        print("SSA-RESET vs FIXED DELTA: Effect sizes across metrics (S6)")
        print("=" * 70)
        ssa_effects = stats.get('effect_sizes', {}).get('ssa_resets_vs_fixed_delta', {})
        for metric, es in ssa_effects.items():
            print(f"  {metric:30s}: d={es.get('cohens_d', 'N/A'):+.3f}, "
                  f"p_fdr={es.get('p_value_fdr', 'N/A')}")
        print()

        # Power analysis
        print("=" * 70)
        print("POWER ANALYSIS")
        print("=" * 70)
        power = stats.get('power_analysis', {})
        print(f"  N subjects: {power.get('n_subjects')}")
        print(f"  Min detectable d (80% power): {power.get('min_detectable_d_80pct', 'N/A'):.3f}")
        print(f"  Fraction powered (>=80%): {power.get('fraction_powered', 'N/A'):.1%}")
        print(f"  Powered comparisons: {power.get('n_comparisons_powered')} / {power.get('n_comparisons_total')}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # 3. Sham hierarchy
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("SHAM HIERARCHY")
    print("=" * 70)
    for cond in ['no_stim', 'sham', 'active_sham', 'fixed_delta', 'fixed_delta_ssa_resets']:
        cdata = df[df['condition'] == cond]
        if len(cdata) > 0 and 'session_swa_enhancement' in cdata.columns:
            m = cdata['session_swa_enhancement'].mean()
            s = cdata['session_swa_enhancement'].std()
            print(f"  {cond:30s}: {m:+6.1f} ± {s:5.1f}%")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # 4. SSA sensitivity
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("SSA SENSITIVITY (tau_slow variants)")
    print("=" * 70)
    for cond in ['fixed_delta', 'fixed_delta_ssa_resets', 'ssa_reset_fast', 'ssa_reset_slow']:
        cdata = df[df['condition'] == cond]
        if len(cdata) > 0 and 'session_swa_enhancement' in cdata.columns:
            m = cdata['session_swa_enhancement'].mean()
            s = cdata['session_swa_enhancement'].std()
            print(f"  {cond:30s}: {m:+6.1f} ± {s:5.1f}%")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # 5. Responder subgroups
    # ═══════════════════════════════════════════════════════════════════
    if subgroup_path.exists():
        with open(subgroup_path) as f:
            subgroups = json.load(f)
        print("=" * 70)
        print("RESPONDER SUBGROUPS (SWA enhancement)")
        print("=" * 70)
        for group in ['high_beta', 'low_beta']:
            gdata = subgroups.get(group, {})
            print(f"  {group}:")
            for cond, vals in gdata.items():
                m = vals.get('mean_swa_enhancement', 0)
                s = vals.get('std_swa_enhancement', 0)
                print(f"    {cond:28s}: {m:+6.1f} ± {s:5.1f}%")
        print(f"  Median beta threshold: {subgroups.get('median_beta_threshold', 'N/A')}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # 6. Key numbers for abstract
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("KEY NUMBERS FOR ABSTRACT")
    print("=" * 70)
    if 'session_swa_enhancement' in df.columns:
        swa = df.groupby('condition')['session_swa_enhancement']
        ssa_reset = swa.get_group('fixed_delta_ssa_resets')
        fixed_delta = swa.get_group('fixed_delta')
        no_stim = swa.get_group('no_stim')

        print(f"  SSA-reset: {ssa_reset.mean():+.1f} ± {ssa_reset.std():.1f}%")
        print(f"  Fixed delta: {fixed_delta.mean():+.1f} ± {fixed_delta.std():.1f}%")
        print(f"  No stim: {no_stim.mean():+.1f} ± {no_stim.std():.1f}%")

        # Active range
        active_conds = ['progressive', 'fixed_delta', 'fixed_delta_ssa_resets',
                        'adaptive_protocol', 'progressive_extended',
                        'pulsed_progressive', 'pulsed_fixed_delta',
                        'progressive_hybrid', 'ssa_reset_fast', 'ssa_reset_slow']
        active_means = []
        for c in active_conds:
            if c in df['condition'].unique():
                active_means.append(df[df['condition'] == c]['session_swa_enhancement'].mean())
        print(f"  Active range: {min(active_means):+.1f}% to {max(active_means):+.1f}%")

        # Low vs high beta on fixed delta
        if 'baseline_beta' in df.columns:
            fd = df[df['condition'] == 'fixed_delta'].copy()
            median_beta = fd['baseline_beta'].median()
            low = fd[fd['baseline_beta'] <= median_beta]['session_swa_enhancement']
            high = fd[fd['baseline_beta'] > median_beta]['session_swa_enhancement']
            print(f"  Fixed delta low-beta: {low.mean():+.1f}%")
            print(f"  Fixed delta high-beta: {high.mean():+.1f}%")
            if high.mean() != 0:
                print(f"  Ratio: {(low.mean() / high.mean() - 1) * 100:.0f}% higher for low-beta")

    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    if verification_path.exists():
        with open(verification_path) as f:
            checks = json.load(f)
        for key, val in checks.items():
            if isinstance(val, (dict, list)):
                print(f"  {key}: {json.dumps(val, default=str)[:100]}")
            else:
                print(f"  {key}: {val}")

    print("\nDone! Use these values to update manuscript.tex")


if __name__ == '__main__':
    main()
