"""
Frequency Resonance Analysis Module.

Provides tools for identifying optimal entrainment frequencies from
Kuramoto model frequency-scan data, including:
- Individual alpha frequency (IAF) estimation
- Population resonance peak estimation via Gaussian fitting
- Bootstrap confidence intervals on peak frequency
- Permutation tests for peak specificity
- Cross-validation across datasets
- Effect size computation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import sem, t as t_dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_iaf(
    band_powers: Dict[str, float],
    alpha_range: Tuple[float, float] = (8.0, 13.0),
) -> float:
    """
    Estimate individual alpha frequency (IAF) from baseline band powers.

    Uses the centroid of the alpha band power as an approximation.
    In real EEG data, IAF would be the peak frequency in the alpha range
    of the power spectrum. Here we use the band-power-weighted estimate
    from available data.

    Args:
        band_powers: Normalized band powers from wake/N1 baseline.
        alpha_range: Alpha band frequency range (Hz).

    Returns:
        Estimated IAF in Hz.
    """
    alpha_power = band_powers.get('alpha_power', 0.0)
    theta_power = band_powers.get('theta_power', 0.0)
    beta_power = band_powers.get('beta_power', 0.0)

    # Weighted estimate using neighboring bands as context
    # Alpha centroid shifts based on relative power distribution
    alpha_center = (alpha_range[0] + alpha_range[1]) / 2.0  # 10.5 Hz

    # Adjust based on theta/beta ratio (lower alpha if more theta)
    if alpha_power > 0.01:
        theta_pull = theta_power / (alpha_power + theta_power + 1e-10)
        beta_pull = beta_power / (alpha_power + beta_power + 1e-10)
        # Shift IAF toward theta if theta is strong, toward beta if beta is strong
        iaf = alpha_center - 1.5 * theta_pull + 1.5 * beta_pull
        iaf = np.clip(iaf, alpha_range[0], alpha_range[1])
    else:
        iaf = alpha_center

    return float(iaf)


def _gaussian(x: np.ndarray, amp: float, mu: float, sigma: float, offset: float) -> np.ndarray:
    """Gaussian function with offset for curve fitting."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def estimate_population_peak(
    scan_df: pd.DataFrame,
    freq_col: str = 'frequency',
    metric_col: str = 'sdre',
) -> Dict[str, float]:
    """
    Fit a Gaussian to the SDRE(f) curve for sub-grid precision peak estimation.

    Args:
        scan_df: DataFrame with frequency and metric columns, aggregated
                 across subjects (mean per frequency).
        freq_col: Column name for frequency.
        metric_col: Column name for the metric (default SDRE).

    Returns:
        Dict with peak_freq, peak_amplitude, sigma, offset, r_squared.
    """
    freqs = scan_df[freq_col].values
    values = scan_df[metric_col].values

    # Initial guesses
    peak_idx = np.argmax(values)
    mu0 = freqs[peak_idx]
    amp0 = values[peak_idx] - np.median(values)
    sigma0 = 1.0
    offset0 = np.median(values)

    try:
        popt, pcov = curve_fit(
            _gaussian, freqs, values,
            p0=[amp0, mu0, sigma0, offset0],
            bounds=(
                [0, freqs.min(), 0.1, -np.inf],
                [np.inf, freqs.max(), 5.0, np.inf],
            ),
            maxfev=10000,
        )

        amp, mu, sigma, offset = popt

        # R-squared
        predicted = _gaussian(freqs, *popt)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'peak_freq': float(mu),
            'peak_amplitude': float(amp),
            'sigma': float(sigma),
            'offset': float(offset),
            'r_squared': float(r_squared),
        }

    except (RuntimeError, ValueError) as e:
        logger.warning(f"Gaussian fit failed: {e}. Using grid maximum.")
        return {
            'peak_freq': float(freqs[peak_idx]),
            'peak_amplitude': float(values[peak_idx]),
            'sigma': float('nan'),
            'offset': float('nan'),
            'r_squared': float('nan'),
        }


def bootstrap_peak_ci(
    subject_scan_results: Dict[str, pd.DataFrame],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    metric_col: str = 'sdre',
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap 95% CI on peak frequency by resampling subjects.

    For each bootstrap iteration:
    1. Resample subjects with replacement
    2. Compute mean SDRE curve across resampled subjects
    3. Find the peak frequency (grid maximum)

    Args:
        subject_scan_results: Dict mapping subject_id -> scan DataFrame.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence interval level.
        metric_col: Metric column to use.
        seed: Random seed.

    Returns:
        Dict with peak_freq, ci_low, ci_high, bootstrap_std, bootstrap_peaks.
    """
    rng = np.random.default_rng(seed)
    subject_ids = list(subject_scan_results.keys())
    n_subjects = len(subject_ids)

    # Get frequency grid from first subject
    first_df = subject_scan_results[subject_ids[0]]
    freq_grid = first_df['frequency'].values

    # Build matrix of SDRE values: subjects x frequencies
    sdre_matrix = np.zeros((n_subjects, len(freq_grid)))
    for i, sid in enumerate(subject_ids):
        df = subject_scan_results[sid]
        sdre_matrix[i, :] = df[metric_col].values

    bootstrap_peaks = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        indices = rng.choice(n_subjects, size=n_subjects, replace=True)
        mean_curve = sdre_matrix[indices].mean(axis=0)
        peak_idx = np.argmax(mean_curve)
        bootstrap_peaks[b] = freq_grid[peak_idx]

    alpha = 1.0 - ci_level
    ci_low = float(np.percentile(bootstrap_peaks, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_peaks, 100 * (1 - alpha / 2)))
    peak_freq = float(np.median(bootstrap_peaks))

    return {
        'peak_freq': peak_freq,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'bootstrap_std': float(np.std(bootstrap_peaks)),
        'bootstrap_peaks': bootstrap_peaks.tolist(),
    }


def permutation_test_peak_specificity(
    subject_scan_results: Dict[str, pd.DataFrame],
    peak_freq: float,
    neighbor_range: float = 1.0,
    n_permutations: int = 10000,
    metric_col: str = 'sdre',
    seed: int = 42,
) -> Dict[str, float]:
    """
    Test whether the peak SDRE differs from neighboring frequencies.

    Null hypothesis: SDRE at peak frequency = mean SDRE at neighbors.

    Uses a permutation test by shuffling frequency labels within each
    subject's scan results.

    Args:
        subject_scan_results: Dict mapping subject_id -> scan DataFrame.
        peak_freq: The identified peak frequency.
        neighbor_range: Range around peak to define neighbors (Hz).
        n_permutations: Number of permutations.
        metric_col: Metric column.
        seed: Random seed.

    Returns:
        Dict with observed_diff, p_value, n_permutations.
    """
    rng = np.random.default_rng(seed)
    subject_ids = list(subject_scan_results.keys())

    first_df = subject_scan_results[subject_ids[0]]
    freq_grid = first_df['frequency'].values

    # Identify peak and neighbor indices
    peak_mask = np.abs(freq_grid - peak_freq) < 0.01  # exact match
    neighbor_mask = (
        (np.abs(freq_grid - peak_freq) > 0.01) &
        (np.abs(freq_grid - peak_freq) <= neighbor_range)
    )

    if not peak_mask.any() or not neighbor_mask.any():
        # Use closest frequency as peak
        peak_idx = np.argmin(np.abs(freq_grid - peak_freq))
        peak_mask = np.zeros(len(freq_grid), dtype=bool)
        peak_mask[peak_idx] = True
        neighbor_mask = (
            (np.arange(len(freq_grid)) != peak_idx) &
            (np.abs(freq_grid - peak_freq) <= neighbor_range)
        )

    if not neighbor_mask.any():
        return {
            'observed_diff': float('nan'),
            'p_value': float('nan'),
            'n_permutations': 0,
        }

    # Build SDRE matrix
    n_subjects = len(subject_ids)
    sdre_matrix = np.zeros((n_subjects, len(freq_grid)))
    for i, sid in enumerate(subject_ids):
        sdre_matrix[i, :] = subject_scan_results[sid][metric_col].values

    # Observed statistic: mean SDRE at peak - mean SDRE at neighbors
    observed_peak = sdre_matrix[:, peak_mask].mean()
    observed_neighbor = sdre_matrix[:, neighbor_mask].mean()
    observed_diff = observed_peak - observed_neighbor

    # Permutation test: shuffle within combined peak+neighbor columns
    combined_mask = peak_mask | neighbor_mask
    combined_indices = np.where(combined_mask)[0]
    n_combined = len(combined_indices)
    n_peak = peak_mask.sum()

    null_diffs = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm_matrix = sdre_matrix[:, combined_indices].copy()
        # Shuffle columns for each subject independently
        for i in range(n_subjects):
            rng.shuffle(perm_matrix[i, :])
        perm_peak = perm_matrix[:, :n_peak].mean()
        perm_neighbor = perm_matrix[:, n_peak:].mean()
        null_diffs[p] = perm_peak - perm_neighbor

    p_value = float(np.mean(null_diffs >= observed_diff))

    return {
        'observed_diff': float(observed_diff),
        'p_value': p_value,
        'n_permutations': n_permutations,
    }


def cross_validate_peak(
    discovery_results: Dict[str, pd.DataFrame],
    validation_results: Dict[str, pd.DataFrame],
    metric_col: str = 'sdre',
) -> Dict[str, object]:
    """
    Cross-validate peak frequency between discovery and validation sets.

    Args:
        discovery_results: Subject scan results for discovery set.
        validation_results: Subject scan results for validation set.
        metric_col: Metric column.

    Returns:
        Dict with discovery_peak, validation_peak, ci_overlap, replicates.
    """
    # Find peak in discovery set
    disc_peak_info = _find_set_peak(discovery_results, metric_col)
    disc_ci = bootstrap_peak_ci(discovery_results, metric_col=metric_col)

    # Find peak in validation set
    val_peak_info = _find_set_peak(validation_results, metric_col)
    val_ci = bootstrap_peak_ci(validation_results, metric_col=metric_col)

    # Check CI overlap
    ci_overlap = (
        disc_ci['ci_low'] <= val_ci['ci_high'] and
        val_ci['ci_low'] <= disc_ci['ci_high']
    )

    return {
        'discovery_peak': disc_peak_info['peak_freq'],
        'discovery_peak_sdre': disc_peak_info['peak_sdre'],
        'discovery_ci': [disc_ci['ci_low'], disc_ci['ci_high']],
        'validation_peak': val_peak_info['peak_freq'],
        'validation_peak_sdre': val_peak_info['peak_sdre'],
        'validation_ci': [val_ci['ci_low'], val_ci['ci_high']],
        'ci_overlap': ci_overlap,
        'replicates': ci_overlap,
        'peak_difference_hz': abs(disc_peak_info['peak_freq'] - val_peak_info['peak_freq']),
    }


def _find_set_peak(
    subject_results: Dict[str, pd.DataFrame],
    metric_col: str = 'sdre',
) -> Dict[str, float]:
    """Find the peak frequency in a set of subject results."""
    subject_ids = list(subject_results.keys())
    first_df = subject_results[subject_ids[0]]
    freq_grid = first_df['frequency'].values

    sdre_matrix = np.zeros((len(subject_ids), len(freq_grid)))
    for i, sid in enumerate(subject_ids):
        sdre_matrix[i, :] = subject_results[sid][metric_col].values

    mean_curve = sdre_matrix.mean(axis=0)
    peak_idx = np.argmax(mean_curve)

    return {
        'peak_freq': float(freq_grid[peak_idx]),
        'peak_sdre': float(mean_curve[peak_idx]),
    }


def compute_effect_sizes(
    subject_scan_results: Dict[str, pd.DataFrame],
    peak_freq: float,
    metric_col: str = 'sdre',
) -> Dict[str, float]:
    """
    Compute Cohen's d at peak frequency vs. baseline (SDRE=0).

    Also computes effect sizes for delta and theta power changes.

    Args:
        subject_scan_results: Dict mapping subject_id -> scan DataFrame.
        peak_freq: The identified peak frequency.
        metric_col: Metric column.

    Returns:
        Dict with cohens_d, mean_sdre, std_sdre, n, and per-band effects.
    """
    subject_ids = list(subject_scan_results.keys())
    first_df = subject_scan_results[subject_ids[0]]
    freq_grid = first_df['frequency'].values

    peak_idx = np.argmin(np.abs(freq_grid - peak_freq))

    # Collect per-subject values at peak frequency
    sdre_at_peak = []
    delta_at_peak = []
    theta_at_peak = []
    for sid in subject_ids:
        df = subject_scan_results[sid]
        sdre_at_peak.append(df.iloc[peak_idx][metric_col])
        delta_at_peak.append(df.iloc[peak_idx]['delta_power'])
        theta_at_peak.append(df.iloc[peak_idx]['theta_power'])

    sdre_arr = np.array(sdre_at_peak)
    delta_arr = np.array(delta_at_peak)
    theta_arr = np.array(theta_at_peak)

    # Cohen's d vs. zero (one-sample)
    def _cohens_d_one_sample(values):
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        if std_val < 1e-10:
            return 0.0
        return float(mean_val / std_val)

    # CI on mean SDRE
    n = len(sdre_arr)
    mean_sdre = float(np.mean(sdre_arr))
    std_sdre = float(np.std(sdre_arr, ddof=1))
    se = std_sdre / np.sqrt(n) if n > 0 else 0.0
    ci_low, ci_high = (mean_sdre, mean_sdre)
    if n > 1 and se > 0:
        ci_low, ci_high = t_dist.interval(0.95, n - 1, loc=mean_sdre, scale=se)

    return {
        'cohens_d_sdre': _cohens_d_one_sample(sdre_arr),
        'mean_sdre': mean_sdre,
        'std_sdre': std_sdre,
        'sdre_ci_low': float(ci_low),
        'sdre_ci_high': float(ci_high),
        'mean_delta_power': float(np.mean(delta_arr)),
        'mean_theta_power': float(np.mean(theta_arr)),
        'n': n,
    }


def compute_iaf_offsets(
    subject_scan_results: Dict[str, pd.DataFrame],
    subject_baselines: Dict[str, Dict[str, float]],
    peak_freq: float,
    metric_col: str = 'sdre',
) -> pd.DataFrame:
    """
    Express optimal frequency as IAF offset for each subject.

    For each subject:
    1. Estimate their IAF from baseline
    2. Compute per-subject peak frequency
    3. Express as offset = peak_freq - IAF

    Args:
        subject_scan_results: Subject scan results.
        subject_baselines: Subject baseline band powers.
        peak_freq: Population peak frequency.
        metric_col: Metric column.

    Returns:
        DataFrame with subject_id, iaf, subject_peak, offset, sdre_at_peak.
    """
    rows = []
    for sid in subject_scan_results:
        df = subject_scan_results[sid]
        baseline = subject_baselines.get(sid, {})
        iaf = estimate_iaf(baseline)

        # Per-subject peak
        peak_idx = df[metric_col].idxmax()
        subject_peak = df.loc[peak_idx, 'frequency']
        sdre_val = df.loc[peak_idx, metric_col]

        rows.append({
            'subject_id': sid,
            'iaf': iaf,
            'population_peak': peak_freq,
            'subject_peak': float(subject_peak),
            'offset_from_iaf': float(subject_peak) - iaf,
            'population_offset_from_iaf': peak_freq - iaf,
            'sdre_at_peak': float(sdre_val),
        })

    return pd.DataFrame(rows)
