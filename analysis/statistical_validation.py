"""
Statistical Validation Module for Frequency Resonance Study.

Provides statistical analysis for the frequency-scan experiment:
- Frequency-domain ANOVA (omnibus test across frequencies)
- Paired comparisons (peak vs baseline, peak vs neighbors)
- FDR correction for multiple comparisons
- Effect size computation (Cohen's d)
- Bootstrap confidence intervals

Metrics:
- SDRE (Sleep Depth Ratio Enhancement): primary outcome
- Delta/theta power changes: secondary outcomes
- PLV (Phase-Locking Value): entrainment measure
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PRIMARY_METRICS = ['sdre']
SECONDARY_METRICS = [
    'delta_power', 'theta_power', 'alpha_power', 'beta_power',
    'plv', 'order_parameter', 'sdr',
]


def paired_cohens_d(differences: np.ndarray) -> float:
    """Cohen's d for paired/repeated-measures design."""
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return 0.0
    sd = np.std(differences, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(differences) / sd)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: non-parametric effect size.

    Computes (#concordant - #discordant) / n_pairs.
    Range: [-1, 1]. Thresholds: |d| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, >= 0.474 large (Romano et al. 2006).
    """
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0
    n_pairs = len(x) * len(y)
    if n_pairs == 0:
        return 0.0
    # Vectorized comparison
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / n_pairs)


def one_sample_cohens_d(values: np.ndarray, mu0: float = 0.0) -> float:
    """Cohen's d for one-sample test against mu0."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    sd = np.std(values, ddof=1)
    if sd == 0:
        return 0.0
    return float((np.mean(values) - mu0) / sd)


def frequency_anova(
    scan_df: pd.DataFrame,
    metric: str = 'sdre',
    freq_col: str = 'frequency',
    subject_col: str = 'subject_id',
) -> Dict[str, float]:
    """
    One-way repeated-measures ANOVA across frequencies.

    Tests whether there is any significant variation in the metric
    across test frequencies (omnibus test).

    Falls back to Friedman test if parametric assumptions are violated.
    """
    clean = scan_df[[subject_col, freq_col, metric]].dropna()

    if len(clean) < 10 or clean[freq_col].nunique() < 3:
        return {'test_statistic': float('nan'), 'p_value': float('nan'),
                'method': 'insufficient_data'}

    # Pivot to wide format: subjects x frequencies
    pivot = clean.pivot_table(
        index=subject_col, columns=freq_col, values=metric
    ).dropna()

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {'test_statistic': float('nan'), 'p_value': float('nan'),
                'method': 'insufficient_data'}

    # Friedman test (non-parametric repeated measures)
    try:
        groups = [pivot[col].values for col in pivot.columns]
        stat, p_val = stats.friedmanchisquare(*groups)
        return {
            'test_statistic': float(stat),
            'p_value': float(p_val),
            'method': 'friedman',
            'n_subjects': int(pivot.shape[0]),
            'n_frequencies': int(pivot.shape[1]),
        }
    except Exception as e:
        logger.warning(f"Friedman test failed: {e}")
        return {'test_statistic': float('nan'), 'p_value': float('nan'),
                'method': 'failed'}


def peak_vs_neighbors_test(
    scan_df: pd.DataFrame,
    peak_freq: float,
    neighbor_range: float = 1.0,
    metric: str = 'sdre',
    subject_col: str = 'subject_id',
) -> Dict[str, float]:
    """
    Paired t-test: SDRE at peak vs. mean SDRE at neighboring frequencies.

    Tests whether the peak is specifically elevated relative to nearby
    frequencies, not just overall.
    """
    freq_col = 'frequency'
    freqs = scan_df[freq_col].unique()

    # Peak values
    peak_mask = np.abs(freqs - peak_freq) < 0.01
    if not peak_mask.any():
        peak_freq_actual = freqs[np.argmin(np.abs(freqs - peak_freq))]
    else:
        peak_freq_actual = peak_freq

    # Neighbor frequencies
    neighbor_freqs = freqs[
        (np.abs(freqs - peak_freq) > 0.01) &
        (np.abs(freqs - peak_freq) <= neighbor_range)
    ]

    if len(neighbor_freqs) == 0:
        return _nan_result()

    # Get per-subject peak values
    peak_data = scan_df[
        np.abs(scan_df[freq_col] - peak_freq_actual) < 0.01
    ].set_index(subject_col)[metric]

    # Get per-subject mean neighbor values
    neighbor_data = scan_df[
        scan_df[freq_col].isin(neighbor_freqs)
    ].groupby(subject_col)[metric].mean()

    # Align subjects
    common = peak_data.index.intersection(neighbor_data.index)
    if len(common) < 3:
        return _nan_result()

    diff = peak_data.loc[common] - neighbor_data.loc[common]
    diff = diff.replace([np.inf, -np.inf], np.nan).dropna()

    if len(diff) < 2:
        return _nan_result()

    mean_diff = float(diff.mean())
    se = float(stats.sem(diff))

    if se == 0:
        return {
            'n': len(diff),
            'mean_diff': mean_diff,
            'ci_low': mean_diff,
            'ci_high': mean_diff,
            'p_value': 1.0 if mean_diff == 0 else 0.0,
            'cohens_d': 0.0,
        }

    ci_low, ci_high = stats.t.interval(0.95, len(diff) - 1, loc=mean_diff, scale=se)
    t_stat, p_value = stats.ttest_1samp(diff, 0)

    return {
        'n': int(len(diff)),
        'mean_diff': mean_diff,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': float(p_value),
        'cohens_d': paired_cohens_d(diff.values),
        't_statistic': float(t_stat),
    }


def _nan_result(n: int = 0) -> Dict[str, float]:
    return {
        'n': n,
        'mean_diff': float('nan'),
        'ci_low': float('nan'),
        'ci_high': float('nan'),
        'p_value': float('nan'),
        'cohens_d': float('nan'),
    }


def apply_fdr_correction(
    p_values: List[float],
    alpha: float = 0.05,
    method: str = 'fdr_bh',
) -> Tuple[List[bool], List[float]]:
    """Apply Benjamini-Hochberg FDR correction."""
    valid = [p for p in p_values if np.isfinite(p)]
    if not valid:
        return [False] * len(p_values), p_values

    p_filled = [p if np.isfinite(p) else 1.0 for p in p_values]
    reject, corrected, _, _ = multipletests(p_filled, alpha=alpha, method=method)
    return list(reject), list(corrected)


def run_frequency_validation(
    scan_df: pd.DataFrame,
    peak_freq: float,
    output_dir: str = None,
) -> Dict:
    """
    Run complete statistical validation for frequency scan results.

    1. Omnibus ANOVA across frequencies
    2. Peak vs. neighbors tests for each metric
    3. FDR correction
    4. Effect sizes
    """
    all_metrics = PRIMARY_METRICS + [
        m for m in SECONDARY_METRICS if m in scan_df.columns
    ]

    # 1. Omnibus ANOVA
    omnibus = {}
    for metric in all_metrics:
        if metric in scan_df.columns:
            omnibus[metric] = frequency_anova(scan_df, metric=metric)

    # 2. Peak vs. neighbors
    comparisons = []
    for metric in all_metrics:
        if metric not in scan_df.columns:
            continue
        comp = peak_vs_neighbors_test(scan_df, peak_freq, metric=metric)
        comp['metric'] = metric
        comparisons.append(comp)

    # 3. FDR correction
    raw_p = [c['p_value'] for c in comparisons]
    reject, corrected_p = apply_fdr_correction(raw_p)
    for i, comp in enumerate(comparisons):
        comp['p_value_fdr'] = corrected_p[i]
        comp['significant_fdr'] = bool(reject[i])

    # 4. Effect sizes at peak (one-sample vs 0)
    effect_sizes = {}
    for metric in all_metrics:
        if metric not in scan_df.columns:
            continue
        peak_values = scan_df[
            np.abs(scan_df['frequency'] - peak_freq) < 0.01
        ][metric].values
        effect_sizes[metric] = {
            'cohens_d': one_sample_cohens_d(peak_values),
            'mean': float(np.nanmean(peak_values)),
            'std': float(np.nanstd(peak_values, ddof=1)),
            'n': len(peak_values),
        }

    report = {
        'peak_frequency': peak_freq,
        'omnibus_tests': omnibus,
        'peak_vs_neighbors': comparisons,
        'effect_sizes': effect_sizes,
        'n_comparisons': len(comparisons),
        'fdr_method': 'benjamini_hochberg',
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'statistical_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Statistical report saved to {output_path}")

    return report


# ─── Protocol Comparison Functions ────────────────────────────────────

PROTOCOL_PRIMARY_METRICS = ['session_swa_enhancement', 'session_sdre', 'cumulative_sleep_depth', 'final_delta_power']
PROTOCOL_SECONDARY_METRICS = ['mean_plv', 'mean_order_parameter', 'final_sdr', 'mean_sdre']


def protocol_friedman_test(
    metrics_df: pd.DataFrame,
    metric: str = 'session_sdre',
    condition_col: str = 'condition',
    subject_col: str = 'subject_id',
) -> Dict[str, float]:
    """
    Friedman test across protocol conditions (omnibus).

    Tests whether there is any significant variation in the metric
    across the 7 protocol conditions.
    """
    clean = metrics_df[[subject_col, condition_col, metric]].dropna()

    if len(clean) < 10 or clean[condition_col].nunique() < 3:
        return {'test_statistic': float('nan'), 'p_value': float('nan'),
                'method': 'insufficient_data'}

    pivot = clean.pivot_table(
        index=subject_col, columns=condition_col, values=metric
    ).dropna()

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {'test_statistic': float('nan'), 'p_value': float('nan'),
                'method': 'insufficient_data'}

    try:
        groups = [pivot[col].values for col in pivot.columns]
        stat, p_val = stats.friedmanchisquare(*groups)
        n_subjects = int(pivot.shape[0])
        k = int(pivot.shape[1])
        # Kendall's W: effect size for Friedman test
        # W = chi2 / (n * (k - 1)), ranges 0 (no agreement) to 1 (perfect)
        kendalls_w = float(stat / (n_subjects * (k - 1))) if (n_subjects * (k - 1)) > 0 else 0.0
        return {
            'test_statistic': float(stat),
            'p_value': float(p_val),
            'method': 'friedman',
            'n_subjects': n_subjects,
            'n_conditions': k,
            'kendalls_w': kendalls_w,
        }
    except Exception as e:
        logger.warning(f"Protocol Friedman test failed: {e}")
        return {'test_statistic': float('nan'), 'p_value': float('nan'),
                'method': 'failed'}


def pairwise_wilcoxon(
    metrics_df: pd.DataFrame,
    target: str = 'progressive',
    controls: List[str] = None,
    metric: str = 'session_sdre',
    subject_col: str = 'subject_id',
    condition_col: str = 'condition',
) -> List[Dict]:
    """
    Pairwise Wilcoxon signed-rank tests: target vs. each control condition.

    Args:
        metrics_df: Session metrics DataFrame.
        target: Target condition name (e.g., 'progressive').
        controls: Control condition names. If None, uses all non-target.
        metric: Metric to compare.
        subject_col: Column with subject IDs.
        condition_col: Column with condition names.

    Returns:
        List of comparison result dicts.
    """
    if controls is None:
        controls = [c for c in metrics_df[condition_col].unique() if c != target]

    target_data = metrics_df[metrics_df[condition_col] == target].set_index(
        subject_col
    )[metric]

    comparisons = []
    for control in controls:
        control_data = metrics_df[
            metrics_df[condition_col] == control
        ].set_index(subject_col)[metric]

        common = target_data.index.intersection(control_data.index)
        if len(common) < 3:
            comparisons.append({
                'target': target,
                'control': control,
                'metric': metric,
                'n': 0,
                'mean_diff': float('nan'),
                'p_value': float('nan'),
                'cohens_d': float('nan'),
            })
            continue

        diff = target_data.loc[common] - control_data.loc[common]
        diff = diff.replace([np.inf, -np.inf], np.nan).dropna()

        if len(diff) < 3:
            comparisons.append({
                'target': target,
                'control': control,
                'metric': metric,
                'n': len(diff),
                'mean_diff': float('nan'),
                'p_value': float('nan'),
                'cohens_d': float('nan'),
            })
            continue

        try:
            stat, p_val = stats.wilcoxon(diff, alternative='two-sided')
        except ValueError:
            stat, p_val = float('nan'), float('nan')

        d = paired_cohens_d(diff.values)

        # Rank-biserial correlation (effect size for Wilcoxon)
        diff_nz = diff[diff != 0].values
        n_nonzero = len(diff_nz)
        if n_nonzero > 0:
            n_pos = np.sum(diff_nz > 0)
            n_neg = np.sum(diff_nz < 0)
            r_rb = float((n_pos - n_neg) / n_nonzero)
        else:
            r_rb = 0.0

        # Cliff's delta (non-parametric effect size)
        target_vals = target_data.loc[common].values
        control_vals = control_data.loc[common].values
        cliff_d = cliffs_delta(target_vals, control_vals)

        se = float(stats.sem(diff))
        if se > 0:
            ci_low, ci_high = stats.t.interval(
                0.95, len(diff) - 1,
                loc=float(diff.mean()), scale=se,
            )
        else:
            ci_low = ci_high = float(diff.mean())

        comparisons.append({
            'target': target,
            'control': control,
            'metric': metric,
            'n': int(len(diff)),
            'mean_diff': float(diff.mean()),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'test_statistic': float(stat),
            'p_value': float(p_val),
            'cohens_d': d,
            'cliffs_delta': cliff_d,
            'rank_biserial_r': r_rb,
        })

    return comparisons


def protocol_effect_sizes(
    metrics_df: pd.DataFrame,
    target: str = 'progressive',
    control: str = 'no_stim',
    metric: str = 'session_sdre',
    subject_col: str = 'subject_id',
    condition_col: str = 'condition',
) -> Dict:
    """
    Paired Cohen's d with 95% CI for target vs. control.
    """
    target_data = metrics_df[metrics_df[condition_col] == target].set_index(
        subject_col
    )[metric]
    control_data = metrics_df[metrics_df[condition_col] == control].set_index(
        subject_col
    )[metric]

    common = target_data.index.intersection(control_data.index)
    if len(common) < 3:
        return {
            'cohens_d': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'n': 0,
        }

    diff = target_data.loc[common] - control_data.loc[common]
    diff = diff.replace([np.inf, -np.inf], np.nan).dropna()

    d = paired_cohens_d(diff.values)

    # Bootstrap CI on Cohen's d
    boot_ds = []
    rng = np.random.default_rng(42)
    for _ in range(2000):
        sample = rng.choice(diff.values, size=len(diff), replace=True)
        sd = np.std(sample, ddof=1)
        boot_d = np.mean(sample) / sd if sd > 0 else 0.0
        boot_ds.append(boot_d)
    boot_ds = np.array(boot_ds)
    ci_low = float(np.percentile(boot_ds, 2.5))
    ci_high = float(np.percentile(boot_ds, 97.5))

    return {
        'cohens_d': d,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'mean_diff': float(diff.mean()),
        'std_diff': float(diff.std(ddof=1)),
        'n': int(len(diff)),
        'target': target,
        'control': control,
        'metric': metric,
    }


def run_protocol_validation(
    metrics_df: pd.DataFrame,
    output_dir: str = None,
) -> Dict:
    """
    Full statistical validation pipeline for protocol comparison.

    1. Omnibus Friedman test across 7 conditions
    2. Pairwise Wilcoxon: progressive vs. each condition
    3. FDR correction
    4. Effect sizes with 95% CI
    """
    all_metrics = PROTOCOL_PRIMARY_METRICS + [
        m for m in PROTOCOL_SECONDARY_METRICS if m in metrics_df.columns
    ]

    # 1. Omnibus tests
    omnibus = {}
    for metric in all_metrics:
        if metric in metrics_df.columns:
            omnibus[metric] = protocol_friedman_test(metrics_df, metric=metric)

    # 2. Pairwise comparisons (progressive vs. each)
    controls = [c for c in metrics_df['condition'].unique()
                if c != 'progressive']

    all_comparisons = []
    for metric in all_metrics:
        if metric not in metrics_df.columns:
            continue
        comps = pairwise_wilcoxon(
            metrics_df, target='progressive',
            controls=controls, metric=metric,
        )
        all_comparisons.extend(comps)

    # 3. FDR correction across all pairwise comparisons
    raw_p = [c['p_value'] for c in all_comparisons]
    reject, corrected_p = apply_fdr_correction(raw_p)
    for i, comp in enumerate(all_comparisons):
        comp['p_value_fdr'] = corrected_p[i]
        comp['significant_fdr'] = bool(reject[i])

    # 4. Effect sizes (progressive vs. each control on primary metric)
    effect_sizes = {}
    for control in controls:
        effect_sizes[f'progressive_vs_{control}'] = protocol_effect_sizes(
            metrics_df, target='progressive', control=control,
        )

    report = {
        'omnibus_tests': omnibus,
        'pairwise_comparisons': all_comparisons,
        'effect_sizes': effect_sizes,
        'n_comparisons': len(all_comparisons),
        'fdr_method': 'benjamini_hochberg',
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'protocol_statistical_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Protocol statistical report saved to {output_path}")

    return report


def compute_power_analysis(
    n_subjects: int,
    comparisons: list,
    alpha: float = 0.05,
) -> Dict:
    """
    Post-hoc power analysis for paired Wilcoxon signed-rank tests.

    Uses the normal approximation: power = Phi(|d|*sqrt(n) - z_{alpha/2})
    where d is the observed paired Cohen's d.

    Also computes the minimum detectable effect size at 80% power.
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)

    # Minimum detectable d at 80% power
    z_beta = norm.ppf(0.80)
    min_detectable_d = (z_alpha + z_beta) / np.sqrt(n_subjects)

    # Power for each observed effect size
    powered_comparisons = []
    for comp in comparisons:
        d = abs(comp.get('cohens_d', 0))
        ncp = d * np.sqrt(n_subjects)
        power = float(norm.cdf(ncp - z_alpha))
        powered_comparisons.append({
            'target': comp.get('target', ''),
            'control': comp.get('control', ''),
            'metric': comp.get('metric', ''),
            'cohens_d': comp.get('cohens_d', 0),
            'power': power,
            'contrast_label': comp.get('contrast_label', ''),
        })

    # Summary: fraction of tests with power >= 0.80
    n_powered = sum(1 for c in powered_comparisons if c['power'] >= 0.80)

    return {
        'n_subjects': n_subjects,
        'alpha': alpha,
        'min_detectable_d_80pct': float(min_detectable_d),
        'n_comparisons_powered': n_powered,
        'n_comparisons_total': len(powered_comparisons),
        'fraction_powered': float(n_powered / max(len(powered_comparisons), 1)),
        'comparisons': powered_comparisons,
    }


def run_redesigned_validation(
    metrics_df: pd.DataFrame,
    output_dir: str = None,
) -> Dict:
    """
    Statistical validation for the 14-condition redesigned study.

    Extends run_protocol_validation() with:
    1. Omnibus Friedman test across all 14 conditions
    2. Pairwise Wilcoxon for all conditions
    3. Targeted contrasts:
       - Pulsed vs continuous (pulsed_progressive vs progressive,
         pulsed_fixed_delta vs fixed_delta)
       - Extended vs standard (progressive_extended vs progressive)
       - Adaptive vs uniform (adaptive_protocol vs progressive, vs fixed_delta)
       - Better sham vs controls (better_sham vs no_stim, vs sham)
       - SSA resets vs fixed (fixed_delta_ssa_resets vs fixed_delta)
    4. FDR correction across all comparisons
    """
    all_metrics = PROTOCOL_PRIMARY_METRICS + [
        m for m in PROTOCOL_SECONDARY_METRICS if m in metrics_df.columns
    ]

    # 1. Omnibus tests across all 14 conditions
    omnibus = {}
    for metric in all_metrics:
        if metric in metrics_df.columns:
            omnibus[metric] = protocol_friedman_test(metrics_df, metric=metric)

    # 2. Pairwise comparisons: progressive vs each, plus targeted contrasts
    all_conditions = sorted(metrics_df['condition'].unique())
    all_comparisons = []

    for metric in all_metrics:
        if metric not in metrics_df.columns:
            continue
        # Progressive vs each other condition
        controls = [c for c in all_conditions if c != 'progressive']
        comps = pairwise_wilcoxon(
            metrics_df, target='progressive',
            controls=controls, metric=metric,
        )
        all_comparisons.extend(comps)

    # 3. Targeted contrasts
    targeted_pairs = [
        ('pulsed_progressive', 'progressive', 'pulsed_vs_continuous_prog'),
        ('pulsed_fixed_delta', 'fixed_delta', 'pulsed_vs_continuous_delta'),
        ('progressive_extended', 'progressive', 'extended_vs_standard'),
        ('adaptive_protocol', 'progressive', 'adaptive_vs_progressive'),
        ('adaptive_protocol', 'fixed_delta', 'adaptive_vs_fixed_delta'),
        ('sham', 'no_stim', 'sham_vs_no_stim'),
        ('active_sham', 'no_stim', 'active_sham_vs_no_stim'),
        ('active_sham', 'sham', 'active_sham_vs_sham'),
        ('fixed_delta', 'active_sham', 'fixed_delta_vs_active_sham'),
        ('fixed_delta_ssa_resets', 'fixed_delta', 'ssa_resets_vs_fixed_delta'),
        ('ssa_reset_fast', 'fixed_delta', 'ssa_fast_vs_fixed_delta'),
        ('ssa_reset_slow', 'fixed_delta', 'ssa_slow_vs_fixed_delta'),
        ('ssa_reset_fast', 'fixed_delta_ssa_resets', 'ssa_fast_vs_ssa_standard'),
        ('ssa_reset_slow', 'fixed_delta_ssa_resets', 'ssa_slow_vs_ssa_standard'),
        ('progressive_hybrid', 'progressive', 'hybrid_vs_progressive'),
        ('progressive_hybrid', 'pulsed_progressive', 'hybrid_vs_pulsed_prog'),
    ]

    targeted_results = []
    for target, control, label in targeted_pairs:
        if target not in all_conditions or control not in all_conditions:
            continue
        for metric in all_metrics:
            if metric not in metrics_df.columns:
                continue
            comps = pairwise_wilcoxon(
                metrics_df, target=target,
                controls=[control], metric=metric,
            )
            for comp in comps:
                comp['contrast_label'] = label
            targeted_results.extend(comps)

    # 4. FDR correction across all pairwise comparisons
    all_pairwise = all_comparisons + targeted_results
    raw_p = [c['p_value'] for c in all_pairwise]
    reject, corrected_p = apply_fdr_correction(raw_p)
    for i, comp in enumerate(all_pairwise):
        comp['p_value_fdr'] = corrected_p[i]
        comp['significant_fdr'] = bool(reject[i])

    # 5. Effect sizes for key contrasts
    effect_sizes = {}
    for target, control, label in targeted_pairs:
        if target not in all_conditions or control not in all_conditions:
            continue
        effect_sizes[label] = protocol_effect_sizes(
            metrics_df, target=target, control=control,
        )

    # 6. Descriptive statistics per condition (median, IQR, skewness)
    descriptives = {}
    for condition in all_conditions:
        cond_data = metrics_df[metrics_df['condition'] == condition]
        cond_desc = {}
        for metric in all_metrics:
            if metric not in cond_data.columns:
                continue
            values = cond_data[metric].dropna().values
            if len(values) < 2:
                continue
            cond_desc[metric] = {
                'mean': float(np.mean(values)),
                'sd': float(np.std(values, ddof=1)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'skewness': float(stats.skew(values)),
                'n': int(len(values)),
            }
        descriptives[condition] = cond_desc

    # 7. Post-hoc power analysis
    n_subjects = metrics_df['subject_id'].nunique()
    power_analysis = compute_power_analysis(n_subjects, all_pairwise)

    report = {
        'omnibus_tests': omnibus,
        'pairwise_comparisons': all_comparisons,
        'targeted_contrasts': targeted_results,
        'effect_sizes': effect_sizes,
        'descriptives': descriptives,
        'power_analysis': power_analysis,
        'n_conditions': len(all_conditions),
        'conditions': all_conditions,
        'n_comparisons': len(all_pairwise),
        'fdr_method': 'benjamini_hochberg',
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'redesigned_statistical_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Redesigned statistical report saved to {output_path}")

    return report


def main():
    """CLI entry point."""
    results_dir = Path('results/frequency_scan')

    scan_file = results_dir / 'phase2_fine_scan_all_subjects.csv'
    if not scan_file.exists():
        scan_file = results_dir / 'phase1_coarse_scan_all_subjects.csv'

    if not scan_file.exists():
        print(f"Error: No scan results found in {results_dir}")
        print("Run scripts/run_frequency_scan.py first.")
        return

    scan_df = pd.read_csv(scan_file)

    # Find peak
    agg = scan_df.groupby('frequency')['sdre'].mean()
    peak_freq = float(agg.idxmax())

    report = run_frequency_validation(
        scan_df, peak_freq,
        output_dir=str(results_dir / 'statistics'),
    )

    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION REPORT")
    print("=" * 70)
    print(f"\nPeak frequency: {peak_freq:.3f} Hz")

    print("\nOmnibus tests:")
    for metric, result in report['omnibus_tests'].items():
        sig = "***" if result['p_value'] < 0.001 else \
              "**" if result['p_value'] < 0.01 else \
              "*" if result['p_value'] < 0.05 else "ns"
        print(f"  {metric}: {result['method']}, p={result['p_value']:.4f} {sig}")

    print("\nPeak vs. neighbors (FDR-corrected):")
    for comp in report['peak_vs_neighbors']:
        sig = "SIG" if comp.get('significant_fdr') else "ns"
        print(f"  {comp['metric']}: diff={comp['mean_diff']:.4f}, "
              f"p_fdr={comp.get('p_value_fdr', comp['p_value']):.4f}, "
              f"d={comp['cohens_d']:.3f} [{sig}]")


if __name__ == '__main__':
    main()
