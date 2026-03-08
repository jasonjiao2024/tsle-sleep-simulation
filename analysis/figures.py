"""
Publication-Quality Figure Generation for Frequency Resonance Study.

Generates 8 figures:
1. Population resonance curve (main result)
2. Band power changes by frequency (4-panel)
3. PLV vs frequency
4. Fine-grained peak with Gaussian fit
5. Cross-dataset validation
6. IAF-relative analysis
7. Sensitivity analysis heatmap
8. Individual variability (spaghetti plot)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results/frequency_scan')
FIGURES_DIR = RESULTS_DIR / 'figures'

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def fig1_resonance_curve(
    coarse_df: pd.DataFrame,
    fine_df: Optional[pd.DataFrame] = None,
    ci_info: Optional[Dict] = None,
):
    """Fig 1: Population resonance curve (main result)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Coarse scan
    ax.plot(coarse_df['frequency'], coarse_df['sdre_mean'],
            'o-', color='#2196F3', markersize=4, linewidth=1.5,
            label='Coarse scan (0.25 Hz)')

    # Error band
    if 'sdre_std' in coarse_df.columns and 'n_subjects' in coarse_df.columns:
        se = coarse_df['sdre_std'] / np.sqrt(coarse_df['n_subjects'])
        ax.fill_between(
            coarse_df['frequency'],
            coarse_df['sdre_mean'] - 1.96 * se,
            coarse_df['sdre_mean'] + 1.96 * se,
            alpha=0.2, color='#2196F3',
        )

    # Fine scan overlay
    if fine_df is not None:
        ax.plot(fine_df['frequency'], fine_df['sdre_mean'],
                's-', color='#E91E63', markersize=3, linewidth=1.5,
                label='Fine scan (0.05 Hz)')
        if 'sdre_std' in fine_df.columns:
            n = fine_df.get('n_subjects', pd.Series([208]*len(fine_df)))
            if isinstance(n, (int, float)):
                n = pd.Series([n]*len(fine_df))
            se = fine_df['sdre_std'] / np.sqrt(208)
            ax.fill_between(
                fine_df['frequency'],
                fine_df['sdre_mean'] - 1.96 * se,
                fine_df['sdre_mean'] + 1.96 * se,
                alpha=0.15, color='#E91E63',
            )

    # Peak marker
    peak_idx = coarse_df['sdre_mean'].idxmax()
    peak_f = coarse_df.loc[peak_idx, 'frequency']
    peak_v = coarse_df.loc[peak_idx, 'sdre_mean']

    if fine_df is not None:
        peak_idx = fine_df['sdre_mean'].idxmax()
        peak_f = fine_df.loc[peak_idx, 'frequency']
        peak_v = fine_df.loc[peak_idx, 'sdre_mean']

    ax.axvline(peak_f, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.plot(peak_f, peak_v, '*', color='red', markersize=15, zorder=5,
            label=f'Peak: {peak_f:.2f} Hz')

    # CI
    if ci_info:
        ax.axvspan(ci_info['ci_low'], ci_info['ci_high'],
                    alpha=0.1, color='red', label='95% CI')

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Forcing Frequency (Hz)')
    ax.set_ylabel('Sleep Depth Ratio Enhancement (SDRE)')
    ax.set_title('Population Resonance Curve: Optimal Entrainment Frequency')
    ax.legend(loc='upper right')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig1_resonance_curve.pdf')
    fig.savefig(FIGURES_DIR / 'fig1_resonance_curve.png')
    plt.close(fig)
    logger.info("Fig 1 saved")


def fig2_band_powers(scan_df: pd.DataFrame):
    """Fig 2: Band power changes by frequency (4-panel)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power']
    colors = ['#1565C0', '#4CAF50', '#FF9800', '#F44336']
    titles = ['Delta (0.5-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 'Beta (13-30 Hz)']

    for ax, band, color, title in zip(axes.flat, bands, colors, titles):
        ax.plot(scan_df['frequency'], scan_df[band], 'o-',
                color=color, markersize=3, linewidth=1.5)
        ax.set_xlabel('Forcing Frequency (Hz)')
        ax.set_ylabel('Normalized Power')
        ax.set_title(title)
        ax.axhline(0.25, color='gray', linestyle='--', alpha=0.3,
                    label='Uniform baseline')

    fig.suptitle('Emergent Band Power Changes Across Forcing Frequencies', fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig2_band_powers.pdf')
    fig.savefig(FIGURES_DIR / 'fig2_band_powers.png')
    plt.close(fig)
    logger.info("Fig 2 saved")


def fig3_plv(scan_df: pd.DataFrame):
    """Fig 3: PLV vs frequency."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(scan_df['frequency'], scan_df['plv'], 'o-',
             color='#9C27B0', markersize=4, linewidth=1.5, label='PLV')
    ax1.set_xlabel('Forcing Frequency (Hz)')
    ax1.set_ylabel('Phase-Locking Value (PLV)', color='#9C27B0')
    ax1.tick_params(axis='y', labelcolor='#9C27B0')

    # Overlay SDRE
    ax2 = ax1.twinx()
    sdre_col = 'sdre_mean' if 'sdre_mean' in scan_df.columns else 'sdre'
    ax2.plot(scan_df['frequency'], scan_df[sdre_col], 's-',
             color='#2196F3', markersize=3, linewidth=1.5, alpha=0.7, label='SDRE')
    ax2.set_ylabel('SDRE', color='#2196F3')
    ax2.tick_params(axis='y', labelcolor='#2196F3')

    # Mark peaks
    plv_peak_idx = scan_df['plv'].idxmax()
    sdre_peak_idx = scan_df[sdre_col].idxmax()

    ax1.axvline(scan_df.loc[plv_peak_idx, 'frequency'], color='#9C27B0',
                linestyle='--', alpha=0.3)
    ax1.axvline(scan_df.loc[sdre_peak_idx, 'frequency'], color='#2196F3',
                linestyle='--', alpha=0.3)

    ax1.set_title('PLV Peak vs. SDRE Peak Dissociation')
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig3_plv_vs_sdre.pdf')
    fig.savefig(FIGURES_DIR / 'fig3_plv_vs_sdre.png')
    plt.close(fig)
    logger.info("Fig 3 saved")


def fig4_fine_peak_gaussian(fine_df: pd.DataFrame):
    """Fig 4: Fine-grained peak with Gaussian fit."""
    fig, ax = plt.subplots(figsize=(8, 5))

    freqs = fine_df['frequency'].values
    sdre = fine_df['sdre_mean'].values

    ax.plot(freqs, sdre, 'ko', markersize=5, label='Data')

    # Gaussian fit
    peak_idx = np.argmax(sdre)
    try:
        popt, _ = curve_fit(
            _gaussian, freqs, sdre,
            p0=[sdre[peak_idx] - np.median(sdre), freqs[peak_idx], 0.5, np.median(sdre)],
            bounds=([0, freqs.min(), 0.1, -np.inf], [np.inf, freqs.max(), 5.0, np.inf]),
            maxfev=10000,
        )

        x_fine = np.linspace(freqs.min(), freqs.max(), 200)
        y_fit = _gaussian(x_fine, *popt)
        ax.plot(x_fine, y_fit, 'r-', linewidth=2,
                label=f'Gaussian fit (peak={popt[1]:.3f} Hz)')

        # R-squared
        predicted = _gaussian(freqs, *popt)
        ss_res = np.sum((sdre - predicted) ** 2)
        ss_tot = np.sum((sdre - np.mean(sdre)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        ax.axvline(popt[1], color='red', linestyle='--', alpha=0.5)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nPeak = {popt[1]:.3f} Hz\nσ = {popt[2]:.3f} Hz',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Gaussian fit failed for fig4: {e}")

    ax.set_xlabel('Forcing Frequency (Hz)')
    ax.set_ylabel('SDRE (mean across subjects)')
    ax.set_title('Fine-Grained Peak Resolution with Gaussian Fit')
    ax.legend()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig4_fine_peak_gaussian.pdf')
    fig.savefig(FIGURES_DIR / 'fig4_fine_peak_gaussian.png')
    plt.close(fig)
    logger.info("Fig 4 saved")


def fig5_cross_validation(
    coarse_all_subjects: pd.DataFrame,
):
    """Fig 5: Cross-dataset validation (discovery vs validation)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Split by dataset
    def classify(sid):
        if isinstance(sid, str) and sid.startswith('SC4'):
            return 'Discovery (Sleep-EDF)'
        return 'Validation (CAP+DREAMS+HMC+SLPDB)'

    coarse_all_subjects = coarse_all_subjects.copy()
    coarse_all_subjects['set'] = coarse_all_subjects['subject_id'].apply(classify)

    for label, color, marker in [
        ('Discovery (Sleep-EDF)', '#2196F3', 'o'),
        ('Validation (CAP+DREAMS+HMC+SLPDB)', '#E91E63', 's'),
    ]:
        subset = coarse_all_subjects[coarse_all_subjects['set'] == label]
        if subset.empty:
            continue
        agg = subset.groupby('frequency')['sdre'].agg(['mean', 'std']).reset_index()
        ax.plot(agg['frequency'], agg['mean'], f'{marker}-',
                color=color, markersize=4, linewidth=1.5, label=label)
        se = agg['std'] / np.sqrt(subset['subject_id'].nunique())
        ax.fill_between(agg['frequency'], agg['mean'] - 1.96 * se,
                        agg['mean'] + 1.96 * se, alpha=0.15, color=color)

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Forcing Frequency (Hz)')
    ax.set_ylabel('SDRE')
    ax.set_title('Cross-Dataset Validation: Discovery vs. Validation')
    ax.legend()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig5_cross_validation.pdf')
    fig.savefig(FIGURES_DIR / 'fig5_cross_validation.png')
    plt.close(fig)
    logger.info("Fig 5 saved")


def fig6_iaf_analysis(iaf_df: pd.DataFrame):
    """Fig 6: IAF-relative analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: IAF distribution
    ax = axes[0]
    ax.hist(iaf_df['iaf'], bins=20, color='#607D8B', edgecolor='white', alpha=0.8)
    ax.axvline(iaf_df['iaf'].mean(), color='red', linestyle='--',
               label=f'Mean: {iaf_df["iaf"].mean():.1f} Hz')
    ax.set_xlabel('Individual Alpha Frequency (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('A. IAF Distribution')
    ax.legend()

    # Panel B: Offset distribution
    ax = axes[1]
    offsets = iaf_df['population_offset_from_iaf']
    ax.hist(offsets, bins=20, color='#FF5722', edgecolor='white', alpha=0.8)
    ax.axvline(offsets.mean(), color='red', linestyle='--',
               label=f'Mean: {offsets.mean():.2f} Hz')
    ax.set_xlabel('Optimal - IAF (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('B. IAF Offset Distribution')
    ax.legend()

    # Panel C: IAF vs SDRE scatter
    ax = axes[2]
    ax.scatter(iaf_df['iaf'], iaf_df['sdre_at_peak'],
               alpha=0.4, s=20, color='#009688')
    # Trend line
    z = np.polyfit(iaf_df['iaf'], iaf_df['sdre_at_peak'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(iaf_df['iaf'].min(), iaf_df['iaf'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=1.5)
    r_val = np.corrcoef(iaf_df['iaf'], iaf_df['sdre_at_peak'])[0, 1]
    ax.set_xlabel('IAF (Hz)')
    ax.set_ylabel('SDRE at Peak')
    ax.set_title(f'C. IAF vs. SDRE (r = {r_val:.3f})')

    fig.suptitle('IAF-Relative Analysis', fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig6_iaf_analysis.pdf')
    fig.savefig(FIGURES_DIR / 'fig6_iaf_analysis.png')
    plt.close(fig)
    logger.info("Fig 6 saved")


def fig7_sensitivity(sensitivity_results: Dict):
    """Fig 7: Sensitivity analysis heatmap."""
    forcing_values = sorted(set(r['forcing'] for r in sensitivity_results.values()))
    n_values = sorted(set(r['n_oscillators'] for r in sensitivity_results.values()))

    peak_matrix = np.zeros((len(forcing_values), len(n_values)))
    sdre_matrix = np.zeros((len(forcing_values), len(n_values)))

    for label, r in sensitivity_results.items():
        fi = forcing_values.index(r['forcing'])
        ni = n_values.index(r['n_oscillators'])
        peak_matrix[fi, ni] = r['peak_freq']
        sdre_matrix[fi, ni] = r['peak_sdre']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Peak frequency
    ax = axes[0]
    im = ax.imshow(peak_matrix, aspect='auto', cmap='coolwarm',
                   origin='lower')
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels(n_values)
    ax.set_yticks(range(len(forcing_values)))
    ax.set_yticklabels([f'{f:.2f}' for f in forcing_values])
    ax.set_xlabel('N oscillators')
    ax.set_ylabel('Forcing strength F')
    ax.set_title('A. Peak Frequency (Hz)')
    for i in range(len(forcing_values)):
        for j in range(len(n_values)):
            ax.text(j, i, f'{peak_matrix[i,j]:.1f}',
                    ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel B: Peak SDRE
    ax = axes[1]
    im = ax.imshow(sdre_matrix, aspect='auto', cmap='YlOrRd',
                   origin='lower')
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels(n_values)
    ax.set_yticks(range(len(forcing_values)))
    ax.set_yticklabels([f'{f:.2f}' for f in forcing_values])
    ax.set_xlabel('N oscillators')
    ax.set_ylabel('Forcing strength F')
    ax.set_title('B. Peak SDRE')
    for i in range(len(forcing_values)):
        for j in range(len(n_values)):
            ax.text(j, i, f'{sdre_matrix[i,j]:.3f}',
                    ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Parameter Sensitivity Analysis', fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig7_sensitivity.pdf')
    fig.savefig(FIGURES_DIR / 'fig7_sensitivity.png')
    plt.close(fig)
    logger.info("Fig 7 saved")


def fig8_individual_variability(all_subjects_df: pd.DataFrame, n_show: int = 30):
    """Fig 8: Individual variability spaghetti plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    subjects = all_subjects_df['subject_id'].unique()
    # Show subset for readability
    show_subjects = subjects[:n_show] if len(subjects) > n_show else subjects

    for sid in show_subjects:
        sub_df = all_subjects_df[all_subjects_df['subject_id'] == sid]
        ax.plot(sub_df['frequency'], sub_df['sdre'],
                alpha=0.15, linewidth=0.5, color='#607D8B')

    # Population mean
    agg = all_subjects_df.groupby('frequency')['sdre'].mean()
    ax.plot(agg.index, agg.values, 'r-', linewidth=2.5, label='Population mean')

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Forcing Frequency (Hz)')
    ax.set_ylabel('SDRE')
    ax.set_title(f'Individual Variability in Frequency Response (n={len(subjects)})')
    ax.legend()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig8_individual_variability.pdf')
    fig.savefig(FIGURES_DIR / 'fig8_individual_variability.png')
    plt.close(fig)
    logger.info("Fig 8 saved")


# ─── Protocol Study Figures ───────────────────────────────────────────

PROTOCOL_RESULTS_DIR = Path('results/protocol_study')
PROTOCOL_FIGURES_DIR = PROTOCOL_RESULTS_DIR / 'figures'

CONDITION_COLORS = {
    'progressive': '#E91E63',
    'reverse': '#9C27B0',
    'fixed_delta': '#1565C0',
    'fixed_theta': '#4CAF50',
    'fixed_alpha': '#FF9800',
    'no_stim': '#607D8B',
    'sham': '#795548',
}

CONDITION_LABELS = {
    'progressive': 'Progressive (10→2 Hz)',
    'reverse': 'Reverse (2→10 Hz)',
    'fixed_delta': 'Fixed Delta (2 Hz)',
    'fixed_theta': 'Fixed Theta (6 Hz)',
    'fixed_alpha': 'Fixed Alpha (8.5 Hz)',
    'no_stim': 'No Stimulation',
    'sham': 'Sham (Random)',
}


def pfig1_sdr_time_course(all_epochs_df: pd.DataFrame):
    """Protocol Fig 1: SDR time course — 7 protocol curves over 30 min."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in CONDITION_COLORS:
        cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
        if cond_data.empty:
            continue
        agg = cond_data.groupby('time_sec')['sdr'].agg(['mean', 'std']).reset_index()
        n_subj = cond_data['subject_id'].nunique()
        se = agg['std'] / np.sqrt(n_subj)

        ax.plot(agg['time_sec'] / 60, agg['mean'],
                color=CONDITION_COLORS[condition], linewidth=2,
                label=CONDITION_LABELS[condition])
        ax.fill_between(agg['time_sec'] / 60,
                         agg['mean'] - 1.96 * se,
                         agg['mean'] + 1.96 * se,
                         color=CONDITION_COLORS[condition], alpha=0.1)

    # Phase boundaries for progressive
    phase_boundaries = [5, 13, 23]  # minutes
    phase_labels = ['10 Hz', '8.5 Hz', '6 Hz', '2 Hz']
    for t in phase_boundaries:
        ax.axvline(t, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Sleep Depth Ratio (SDR)')
    ax.set_title('SDR Time Course by Protocol Condition')
    ax.legend(loc='upper left', fontsize=8)

    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig1_sdr_time_course.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig1_sdr_time_course.png')
    plt.close(fig)
    logger.info("Protocol Fig 1 saved")


def pfig2_session_sdre_bars(session_metrics_df: pd.DataFrame):
    """Protocol Fig 2: Session SDRE bar chart with 95% CI."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Order by mean SDRE
    order = session_metrics_df.groupby('condition')['session_sdre'].mean().sort_values(ascending=False).index.tolist()

    means = []
    cis = []
    colors = []
    labels = []

    for cond in order:
        data = session_metrics_df[session_metrics_df['condition'] == cond]['session_sdre']
        mean = float(data.mean())
        se = float(data.std() / np.sqrt(len(data)))
        ci = 1.96 * se
        means.append(mean)
        cis.append(ci)
        colors.append(CONDITION_COLORS.get(cond, '#999999'))
        labels.append(CONDITION_LABELS.get(cond, cond))

    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=cis, capsize=4,
                  color=colors, edgecolor='white', linewidth=1.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Session SDRE')
    ax.set_title('Sleep Depth Ratio Enhancement by Protocol')

    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig2_session_sdre_bars.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig2_session_sdre_bars.png')
    plt.close(fig)
    logger.info("Protocol Fig 2 saved")


def pfig3_cumulative_sleep_depth(session_metrics_df: pd.DataFrame):
    """Protocol Fig 3: Cumulative sleep depth (integrated SDR) by protocol."""
    fig, ax = plt.subplots(figsize=(10, 6))

    order = session_metrics_df.groupby('condition')['cumulative_sleep_depth'].mean().sort_values(ascending=False).index.tolist()

    means = []
    cis = []
    colors = []
    labels = []

    for cond in order:
        data = session_metrics_df[session_metrics_df['condition'] == cond]['cumulative_sleep_depth']
        mean = float(data.mean())
        se = float(data.std() / np.sqrt(len(data)))
        ci = 1.96 * se
        means.append(mean)
        cis.append(ci)
        colors.append(CONDITION_COLORS.get(cond, '#999999'))
        labels.append(CONDITION_LABELS.get(cond, cond))

    x = np.arange(len(order))
    ax.bar(x, means, yerr=cis, capsize=4,
           color=colors, edgecolor='white', linewidth=1.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Cumulative Sleep Depth (SDR * sec)')
    ax.set_title('Cumulative Sleep Depth by Protocol')

    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig3_cumulative_sleep_depth.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig3_cumulative_sleep_depth.png')
    plt.close(fig)
    logger.info("Protocol Fig 3 saved")


def pfig4_per_phase_band_powers(all_epochs_df: pd.DataFrame):
    """Protocol Fig 4: Per-phase band powers (4-panel) for progressive protocol."""
    prog_data = all_epochs_df[all_epochs_df['condition'] == 'progressive']
    if prog_data.empty:
        logger.warning("No progressive data for pfig4")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power']
    colors = ['#1565C0', '#4CAF50', '#FF9800', '#F44336']
    titles = ['Delta (0.5-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 'Beta (13-30 Hz)']

    phase_order = ['alpha_10hz', 'alpha_8.5hz', 'theta_6hz', 'delta_2hz']
    phase_labels = ['10 Hz\n(5 min)', '8.5 Hz\n(8 min)', '6 Hz\n(10 min)', '2 Hz\n(7 min)']

    for ax, band, color, title in zip(axes.flat, bands, colors, titles):
        means = []
        stds = []
        for phase_name in phase_order:
            phase_data = prog_data[prog_data['phase_name'] == phase_name]
            if phase_data.empty:
                means.append(0)
                stds.append(0)
            else:
                agg = phase_data.groupby('subject_id')[band].mean()
                means.append(float(agg.mean()))
                stds.append(float(agg.std() / np.sqrt(len(agg))))

        x = np.arange(len(phase_order))
        ax.bar(x, means, yerr=[1.96 * s for s in stds], capsize=4,
               color=color, alpha=0.7, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=9)
        ax.set_ylabel('Normalized Power')
        ax.set_title(title)
        ax.axhline(0.25, color='gray', linestyle='--', alpha=0.3)

    fig.suptitle('Band Power Changes Across Progressive Protocol Phases', fontsize=14)
    fig.tight_layout()
    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig4_per_phase_band_powers.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig4_per_phase_band_powers.png')
    plt.close(fig)
    logger.info("Protocol Fig 4 saved")


def pfig5_plv_per_phase(all_epochs_df: pd.DataFrame):
    """Protocol Fig 5: PLV per phase — entrainment tracks the drive."""
    prog_data = all_epochs_df[all_epochs_df['condition'] == 'progressive']
    if prog_data.empty:
        logger.warning("No progressive data for pfig5")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    phase_order = ['alpha_10hz', 'alpha_8.5hz', 'theta_6hz', 'delta_2hz']
    phase_labels = ['10 Hz', '8.5 Hz', '6 Hz', '2 Hz']
    phase_colors = ['#FF9800', '#FFC107', '#4CAF50', '#1565C0']

    means = []
    stds = []
    for phase_name in phase_order:
        phase_data = prog_data[prog_data['phase_name'] == phase_name]
        if phase_data.empty:
            means.append(0)
            stds.append(0)
        else:
            agg = phase_data.groupby('subject_id')['plv'].mean()
            means.append(float(agg.mean()))
            stds.append(float(agg.std() / np.sqrt(len(agg))))

    x = np.arange(len(phase_order))
    bars = ax.bar(x, means, yerr=[1.96 * s for s in stds], capsize=5,
                  color=phase_colors, edgecolor='white', linewidth=1.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=11)
    ax.set_ylabel('Phase-Locking Value (PLV)')
    ax.set_title('Entrainment Strength Across Progressive Protocol Phases')

    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig5_plv_per_phase.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig5_plv_per_phase.png')
    plt.close(fig)
    logger.info("Protocol Fig 5 saved")


def pfig6_cross_dataset_replication(session_metrics_df: pd.DataFrame):
    """Protocol Fig 6: Cross-dataset replication — discovery vs. validation SDRE."""
    fig, ax = plt.subplots(figsize=(10, 6))

    def classify(sid):
        if isinstance(sid, str) and sid.startswith('SC4'):
            return 'Discovery (Sleep-EDF)'
        return 'Validation (Others)'

    df = session_metrics_df.copy()
    df['dataset_group'] = df['subject_id'].apply(classify)

    conditions = ['progressive', 'reverse', 'fixed_delta', 'fixed_theta',
                  'fixed_alpha', 'no_stim', 'sham']
    x = np.arange(len(conditions))
    width = 0.35

    for i, (group, color) in enumerate([
        ('Discovery (Sleep-EDF)', '#2196F3'),
        ('Validation (Others)', '#E91E63'),
    ]):
        group_data = df[df['dataset_group'] == group]
        means = []
        cis = []
        for cond in conditions:
            cond_data = group_data[group_data['condition'] == cond]['session_sdre']
            means.append(float(cond_data.mean()) if len(cond_data) > 0 else 0)
            se = float(cond_data.std() / np.sqrt(len(cond_data))) if len(cond_data) > 1 else 0
            cis.append(1.96 * se)

        ax.bar(x + i * width - width / 2, means, width, yerr=cis,
               capsize=3, color=color, alpha=0.7, label=group)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions],
                       rotation=35, ha='right', fontsize=8)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Session SDRE')
    ax.set_title('Cross-Dataset Replication of Protocol Effects')
    ax.legend()

    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig6_cross_dataset_replication.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig6_cross_dataset_replication.png')
    plt.close(fig)
    logger.info("Protocol Fig 6 saved")


def pfig7_individual_variability(session_metrics_df: pd.DataFrame):
    """Protocol Fig 7: Individual variability — SDRE distribution + responder fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Distribution of progressive SDRE
    ax = axes[0]
    prog_data = session_metrics_df[
        session_metrics_df['condition'] == 'progressive'
    ]['session_sdre']

    ax.hist(prog_data, bins=30, color='#E91E63', edgecolor='white', alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(prog_data.mean(), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {prog_data.mean():.3f}')
    n_responders = int((prog_data > 0).sum())
    frac = n_responders / len(prog_data) * 100
    ax.set_xlabel('Session SDRE')
    ax.set_ylabel('Count')
    ax.set_title(f'A. Progressive SDRE Distribution\n'
                 f'({n_responders}/{len(prog_data)} responders = {frac:.0f}%)')
    ax.legend()

    # Panel B: Progressive vs No-stim paired
    ax = axes[1]
    prog = session_metrics_df[session_metrics_df['condition'] == 'progressive'].set_index('subject_id')['session_sdre']
    nostim = session_metrics_df[session_metrics_df['condition'] == 'no_stim'].set_index('subject_id')['session_sdre']
    common = prog.index.intersection(nostim.index)
    diff = prog.loc[common] - nostim.loc[common]

    ax.hist(diff, bins=30, color='#9C27B0', edgecolor='white', alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(diff.mean(), color='red', linestyle='-', linewidth=2,
               label=f'Mean diff: {diff.mean():.3f}')
    ax.set_xlabel('Progressive - No Stim (SDRE)')
    ax.set_ylabel('Count')
    ax.set_title('B. Paired Difference Distribution')
    ax.legend()

    fig.suptitle('Individual Variability in Protocol Response', fontsize=14)
    fig.tight_layout()
    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig7_individual_variability.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig7_individual_variability.png')
    plt.close(fig)
    logger.info("Protocol Fig 7 saved")


def pfig8_order_parameter_dynamics(all_epochs_df: pd.DataFrame):
    """Protocol Fig 8: Order parameter synchronization trajectory by protocol."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in CONDITION_COLORS:
        cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
        if cond_data.empty:
            continue
        agg = cond_data.groupby('time_sec')['order_parameter'].agg(
            ['mean', 'std']
        ).reset_index()
        n_subj = cond_data['subject_id'].nunique()
        se = agg['std'] / np.sqrt(n_subj)

        ax.plot(agg['time_sec'] / 60, agg['mean'],
                color=CONDITION_COLORS[condition], linewidth=2,
                label=CONDITION_LABELS[condition])
        ax.fill_between(agg['time_sec'] / 60,
                         agg['mean'] - 1.96 * se,
                         agg['mean'] + 1.96 * se,
                         color=CONDITION_COLORS[condition], alpha=0.1)

    phase_boundaries = [5, 13, 23]
    for t in phase_boundaries:
        ax.axvline(t, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Order Parameter (r)')
    ax.set_title('Synchronization Dynamics by Protocol Condition')
    ax.legend(loc='upper right', fontsize=8)

    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig8_order_parameter_dynamics.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig8_order_parameter_dynamics.png')
    plt.close(fig)
    logger.info("Protocol Fig 8 saved")


def pfig9_thalamic_trajectory(all_epochs_df: pd.DataFrame):
    """Protocol Fig 9: Thalamic variable T(t) trajectory by protocol condition.

    The mechanistic "money figure" — shows how thalamocortical feedback
    evolves differently across protocols. Progressive protocol should show
    T rising early (alpha resonance drives cortical synchrony) and staying
    elevated, while fixed-delta shows slower T buildup.
    """
    if 'thalamic_T' not in all_epochs_df.columns:
        logger.warning("No thalamic_T column found — skipping pfig9 "
                       "(requires TSLE model output)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in CONDITION_COLORS:
        cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
        if cond_data.empty:
            continue
        agg = cond_data.groupby('time_sec')['thalamic_T'].agg(
            ['mean', 'std']
        ).reset_index()
        n_subj = cond_data['subject_id'].nunique()
        se = agg['std'] / np.sqrt(n_subj)

        ax.plot(agg['time_sec'] / 60, agg['mean'],
                color=CONDITION_COLORS[condition], linewidth=2,
                label=CONDITION_LABELS[condition])
        ax.fill_between(agg['time_sec'] / 60,
                         agg['mean'] - 1.96 * se,
                         agg['mean'] + 1.96 * se,
                         color=CONDITION_COLORS[condition], alpha=0.1)

    # Phase boundaries for progressive
    phase_boundaries = [5, 13, 23]  # minutes
    phase_labels_top = ['10 Hz', '8.5 Hz', '6 Hz', '2 Hz']
    for i, t in enumerate(phase_boundaries):
        ax.axvline(t, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    # Annotate progressive phases at top
    phase_starts = [0, 5, 13, 23]
    phase_ends = [5, 13, 23, 30]
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for start, end, label in zip(phase_starts, phase_ends, phase_labels_top):
        mid = (start + end) / 2
        ax.text(mid, y_top * 0.95, label, ha='center', va='top',
                fontsize=8, color='#E91E63', alpha=0.7)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Thalamic Variable T')
    ax.set_title('Thalamocortical Feedback Trajectory by Protocol')
    ax.legend(loc='upper left', fontsize=8)

    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig9_thalamic_trajectory.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig9_thalamic_trajectory.png')
    plt.close(fig)
    logger.info("Protocol Fig 9 saved")


def pfig10_frequency_shift(all_epochs_df: pd.DataFrame):
    """Protocol Fig 10: Cortical frequency distribution shift during progressive protocol.

    Shows histogram of mean effective cortical frequencies at 4 timepoints
    in the progressive protocol, demonstrating how the thalamocortical loop
    progressively shifts oscillator frequencies toward delta.
    """
    if 'mean_omega_hz' not in all_epochs_df.columns:
        logger.warning("No mean_omega_hz column found — skipping pfig10 "
                       "(requires TSLE model output)")
        return

    prog_data = all_epochs_df[all_epochs_df['condition'] == 'progressive']
    if prog_data.empty:
        logger.warning("No progressive data for pfig10")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Sample 4 timepoints: start of each phase
    # Phase boundaries: 0-5 min (10Hz), 5-13 min (8.5Hz), 13-23 min (6Hz), 23-30 min (2Hz)
    timepoints = [
        (30.0, 'Start (t=0.5 min)\n10 Hz phase'),
        (390.0, 'Alpha-theta (t=6.5 min)\n8.5 Hz phase'),
        (780.0, 'Theta (t=13 min)\n6 Hz phase'),
        (1410.0, 'Delta (t=23.5 min)\n2 Hz phase'),
    ]
    colors = ['#FF9800', '#FFC107', '#4CAF50', '#1565C0']

    for ax, (t_sec, label), color in zip(axes.flat, timepoints, colors):
        # Get mean_omega_hz at this timepoint across subjects
        epoch_data = prog_data[prog_data['time_sec'] == t_sec]
        if epoch_data.empty:
            # Find closest timepoint
            closest_t = prog_data['time_sec'].iloc[
                (prog_data['time_sec'] - t_sec).abs().argsort().iloc[0]
            ]
            epoch_data = prog_data[prog_data['time_sec'] == closest_t]

        if epoch_data.empty:
            ax.set_visible(False)
            continue

        values = epoch_data['mean_omega_hz'].values

        ax.hist(values, bins=20, color=color, edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {np.mean(values):.1f} Hz')
        ax.set_xlabel('Mean Effective Frequency (Hz)')
        ax.set_ylabel('Count')
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle('Cortical Frequency Distribution Shift During Progressive Protocol',
                 fontsize=13)
    fig.tight_layout()
    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig10_frequency_shift.pdf')
    fig.savefig(PROTOCOL_FIGURES_DIR / 'pfig10_frequency_shift.png')
    plt.close(fig)
    logger.info("Protocol Fig 10 saved")


def generate_protocol_figures():
    """Generate all 10 protocol study figures from saved results."""
    logger.info("Generating protocol study figures...")
    PROTOCOL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs_file = PROTOCOL_RESULTS_DIR / 'all_epochs.csv'
    metrics_file = PROTOCOL_RESULTS_DIR / 'session_metrics.csv'

    if not epochs_file.exists() or not metrics_file.exists():
        logger.error(f"Protocol study results not found in {PROTOCOL_RESULTS_DIR}")
        logger.error("Run scripts/run_protocol_study.py first.")
        return

    all_epochs_df = pd.read_csv(epochs_file)
    session_metrics_df = pd.read_csv(metrics_file)

    pfig1_sdr_time_course(all_epochs_df)
    pfig2_session_sdre_bars(session_metrics_df)
    pfig3_cumulative_sleep_depth(session_metrics_df)
    pfig4_per_phase_band_powers(all_epochs_df)
    pfig5_plv_per_phase(all_epochs_df)
    pfig6_cross_dataset_replication(session_metrics_df)
    pfig7_individual_variability(session_metrics_df)
    pfig8_order_parameter_dynamics(all_epochs_df)
    pfig9_thalamic_trajectory(all_epochs_df)
    pfig10_frequency_shift(all_epochs_df)

    logger.info(f"All protocol figures saved to {PROTOCOL_FIGURES_DIR}")


def generate_all_figures():
    """Generate all 8 figures from saved results."""
    logger.info("Generating all figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    coarse_df = pd.read_csv(RESULTS_DIR / 'phase1_coarse_scan.csv')
    coarse_all = pd.read_csv(RESULTS_DIR / 'phase1_coarse_scan_all_subjects.csv')

    fine_df = None
    ci_info = None
    if (RESULTS_DIR / 'phase2_fine_scan.csv').exists():
        fine_df = pd.read_csv(RESULTS_DIR / 'phase2_fine_scan.csv')

    if (RESULTS_DIR / 'statistics' / 'statistical_report.json').exists():
        with open(RESULTS_DIR / 'statistics' / 'statistical_report.json') as f:
            stats = json.load(f)
            if 'bootstrap_ci' in stats:
                ci_info = {
                    'ci_low': stats['bootstrap_ci'][0],
                    'ci_high': stats['bootstrap_ci'][1],
                }

    # Fig 1: Resonance curve
    fig1_resonance_curve(coarse_df, fine_df, ci_info)

    # Fig 2: Band powers
    fig2_band_powers(coarse_df)

    # Fig 3: PLV
    fig3_plv(coarse_df)

    # Fig 4: Fine peak Gaussian
    if fine_df is not None:
        fig4_fine_peak_gaussian(fine_df)

    # Fig 5: Cross-validation
    fig5_cross_validation(coarse_all)

    # Fig 6: IAF analysis
    if (RESULTS_DIR / 'phase4_iaf_analysis.csv').exists():
        iaf_df = pd.read_csv(RESULTS_DIR / 'phase4_iaf_analysis.csv')
        fig6_iaf_analysis(iaf_df)

    # Fig 7: Sensitivity
    if (RESULTS_DIR / 'phase5_sensitivity.json').exists():
        with open(RESULTS_DIR / 'phase5_sensitivity.json') as f:
            sensitivity = json.load(f)
        fig7_sensitivity(sensitivity)

    # Fig 8: Individual variability
    fig8_individual_variability(coarse_all)

    logger.info(f"All figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    generate_all_figures()
    generate_protocol_figures()
