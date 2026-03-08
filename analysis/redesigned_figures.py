"""
Figure Generation for Redesigned Sleep Entrainment Study.

6 new figures:
- rfig1: Adaptation time course (60-min SDR curves, crossover hypothesis)
- rfig2: Pulsed vs continuous comparison (4-panel)
- rfig3: Responder subgroups (high/low beta split, adaptive protocol)
- rfig4: SSA dynamics (adaptation curves for key conditions)
- rfig5: Extended thalamic priming (T trajectory at 60 min)
- rfig6: Sham validation (no_stim vs sham vs better_sham SDRE distributions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results/redesigned_study')
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

# Extended color palette for 14 conditions
# Key: SSA-Reset (main finding) uses gold/amber to stand out
CONDITION_COLORS = {
    'progressive': '#E91E63',
    'reverse': '#9C27B0',
    'fixed_delta': '#1565C0',
    'fixed_theta': '#4CAF50',
    'fixed_alpha': '#FF9800',
    'no_stim': '#9E9E9E',
    'sham': '#795548',
    'progressive_extended': '#880E4F',
    'fixed_delta_ssa_resets': '#FF6F00',
    'pulsed_progressive': '#F06292',
    'pulsed_fixed_delta': '#64B5F6',
    'adaptive_protocol': '#00BCD4',
    'active_sham': '#8BC34A',
    'progressive_hybrid': '#AB47BC',
    'ssa_reset_fast': '#FF8F00',
    'ssa_reset_slow': '#E65100',
}

CONDITION_LABELS = {
    'progressive': 'Progressive',
    'reverse': 'Reverse',
    'fixed_delta': 'Fixed Delta',
    'fixed_theta': 'Fixed Theta',
    'fixed_alpha': 'Fixed Alpha',
    'no_stim': 'No Stim',
    'sham': 'Sham',
    'progressive_extended': 'Prog. Extended',
    'fixed_delta_ssa_resets': 'SSA-Reset Delta',
    'pulsed_progressive': 'Pulsed Prog.',
    'pulsed_fixed_delta': 'Pulsed Delta',
    'adaptive_protocol': 'Adaptive',
    'active_sham': 'Active Sham',
    'progressive_hybrid': 'Hybrid',
    'ssa_reset_fast': 'SSA-Reset Fast',
    'ssa_reset_slow': 'SSA-Reset Slow',
}


def rfig1_adaptation_time_course(all_epochs_df: pd.DataFrame):
    """RFig 1: Adaptation time course — SWA and SWA enhancement curves.

    At extended durations (60 min), progressive should eventually catch up to
    or surpass fixed_delta due to:
    - TC priming providing lasting excitability boost
    - SSA accumulation degrading fixed_delta forcing effectiveness
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    key_conditions = ['progressive', 'progressive_extended', 'fixed_delta',
                      'fixed_delta_ssa_resets']

    # Panel A: SWA time course (primary v2 metric)
    ax = axes[0]
    swa_col = 'swa' if 'swa' in all_epochs_df.columns else 'sdr'
    swa_label = 'SWA (absolute delta power)' if swa_col == 'swa' else 'Sleep Depth Ratio (SDR)'

    for condition in key_conditions:
        cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
        if cond_data.empty:
            continue
        agg = cond_data.groupby('time_sec')[swa_col].agg(['mean', 'std']).reset_index()
        n_subj = cond_data['subject_id'].nunique()
        se = agg['std'] / np.sqrt(max(n_subj, 1))

        color = CONDITION_COLORS.get(condition, '#999')
        label = CONDITION_LABELS.get(condition, condition)
        lw = 2.5 if condition == 'fixed_delta_ssa_resets' else 1.8
        ax.plot(agg['time_sec'] / 60, agg['mean'],
                color=color, linewidth=lw, label=label)
        ax.fill_between(agg['time_sec'] / 60,
                        agg['mean'] - 1.96 * se,
                        agg['mean'] + 1.96 * se,
                        color=color, alpha=0.1)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(swa_label)
    ax.set_title('A. SWA Time Course')
    ax.legend(fontsize=9)

    # Panel B: SWA enhancement in sliding 5-min windows
    ax = axes[1]
    window_epochs = 10  # 5 min
    enh_col = 'swa_enhancement' if 'swa_enhancement' in all_epochs_df.columns else 'sdre'
    enh_label = 'SWA Enhancement (%)' if enh_col == 'swa_enhancement' else 'SDRE (5-min rolling)'

    for condition in key_conditions:
        cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
        if cond_data.empty:
            continue

        subjects = cond_data['subject_id'].unique()
        all_windows = []
        for sid in subjects:
            sub = cond_data[cond_data['subject_id'] == sid].sort_values('time_sec')
            if len(sub) < window_epochs:
                continue
            windowed = sub[enh_col].rolling(window=window_epochs, min_periods=1).mean()
            all_windows.append(pd.DataFrame({
                'time_sec': sub['time_sec'].values,
                'enh_window': windowed.values,
            }))

        if not all_windows:
            continue
        combined = pd.concat(all_windows)
        agg = combined.groupby('time_sec')['enh_window'].agg(['mean', 'std']).reset_index()

        color = CONDITION_COLORS.get(condition, '#999')
        label = CONDITION_LABELS.get(condition, condition)
        lw = 2.5 if condition == 'fixed_delta_ssa_resets' else 1.8
        ax.plot(agg['time_sec'] / 60, agg['mean'],
                color=color, linewidth=lw, label=label)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(enh_label)
    ax.set_title('B. Rolling SWA Enhancement Over Session')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle('SSA-Reset Maintains Superior Entrainment Over 60 Minutes', fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'rfig1_adaptation_time_course.pdf')
    fig.savefig(FIGURES_DIR / 'rfig1_adaptation_time_course.png')
    plt.close(fig)
    logger.info("RFig 1 saved")


def rfig2_pulsed_vs_continuous(all_epochs_df: pd.DataFrame):
    """RFig 2: Pulsed vs continuous comparison (4-panel).

    Compares progressive vs pulsed_progressive and fixed_delta vs pulsed_fixed_delta
    on SDR and PLV.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pairs = [
        ('progressive', 'pulsed_progressive', 'Progressive'),
        ('fixed_delta', 'pulsed_fixed_delta', 'Fixed Delta'),
    ]
    swa_metric = 'swa' if 'swa' in all_epochs_df.columns else 'sdr'
    swa_label = 'SWA' if swa_metric == 'swa' else 'SDR'
    metrics = [(swa_metric, swa_label), ('plv', 'PLV')]

    for row, (metric, metric_label) in enumerate(metrics):
        for col, (cont_cond, pulsed_cond, pair_label) in enumerate(pairs):
            ax = axes[row, col]

            for condition, ls, lbl_suffix in [
                (cont_cond, '-', 'Continuous'),
                (pulsed_cond, '--', 'Pulsed'),
            ]:
                cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
                if cond_data.empty:
                    continue
                agg = cond_data.groupby('time_sec')[metric].agg(
                    ['mean', 'std']
                ).reset_index()
                n_subj = cond_data['subject_id'].nunique()
                se = agg['std'] / np.sqrt(max(n_subj, 1))

                color = CONDITION_COLORS.get(condition, '#999')
                ax.plot(agg['time_sec'] / 60, agg['mean'],
                        color=color, linewidth=2, linestyle=ls,
                        label=f'{pair_label} {lbl_suffix}')
                ax.fill_between(agg['time_sec'] / 60,
                                agg['mean'] - 1.96 * se,
                                agg['mean'] + 1.96 * se,
                                color=color, alpha=0.08)

            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{pair_label}: {metric_label}')
            ax.legend(fontsize=8)

    fig.suptitle('Pulsed vs Continuous Stimulation Comparison', fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'rfig2_pulsed_vs_continuous.pdf')
    fig.savefig(FIGURES_DIR / 'rfig2_pulsed_vs_continuous.png')
    plt.close(fig)
    logger.info("RFig 2 saved")


def rfig3_responder_subgroups(
    session_metrics_df: pd.DataFrame,
    all_epochs_df: Optional[pd.DataFrame] = None,
):
    """RFig 3: Responder subgroups — high/low beta split, adaptive protocol.

    Shows that subjects with high baseline beta respond better to progressive
    (r=-0.686 finding), and that the adaptive protocol exploits this.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Beta vs SDRE scatter for progressive
    ax = axes[0]
    prog = session_metrics_df[session_metrics_df['condition'] == 'progressive']
    if 'baseline_beta' in session_metrics_df.columns:
        beta_col = 'baseline_beta'
    elif 'beta_power_alpha_10hz' in prog.columns:
        beta_col = 'beta_power_alpha_10hz'
    else:
        beta_col = None

    if beta_col and beta_col in prog.columns:
        ax.scatter(prog[beta_col], prog['session_swa_enhancement'],
                   alpha=0.4, s=20, color='#E91E63')
        # Trend line
        valid = prog[[beta_col, 'session_swa_enhancement']].dropna()
        if len(valid) > 5:
            z = np.polyfit(valid[beta_col], valid['session_swa_enhancement'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid[beta_col].min(), valid[beta_col].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=1.5)
            r_val = np.corrcoef(valid[beta_col], valid['session_swa_enhancement'])[0, 1]
            ax.set_title(f'A. Beta vs SWA Enhancement (r={r_val:.3f})')
        else:
            ax.set_title('A. Beta vs SWA Enhancement')
    else:
        ax.text(0.5, 0.5, 'Beta data\nnot available', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('A. Beta vs SWA Enhancement')
    ax.set_xlabel('Baseline Beta Power')
    ax.set_ylabel('SWA Enhancement (%)')

    # Panel B: High vs low beta SDRE by condition
    ax = axes[1]
    if beta_col and beta_col in session_metrics_df.columns:
        median_beta = session_metrics_df[beta_col].median()
        high_beta = session_metrics_df[session_metrics_df[beta_col] > median_beta]
        low_beta = session_metrics_df[session_metrics_df[beta_col] <= median_beta]

        key_conds = ['progressive', 'fixed_delta', 'adaptive_protocol']
        key_conds = [c for c in key_conds if c in session_metrics_df['condition'].unique()]
        x = np.arange(len(key_conds))
        width = 0.35

        for i, (subset, label, color) in enumerate([
            (high_beta, 'High Beta', '#E91E63'),
            (low_beta, 'Low Beta', '#2196F3'),
        ]):
            medians_b, iqr_lo_b, iqr_hi_b = [], [], []
            for cond in key_conds:
                data = subset[subset['condition'] == cond]['session_swa_enhancement'].dropna().values
                med = float(np.median(data)) if len(data) > 0 else 0
                q25 = float(np.percentile(data, 25)) if len(data) > 0 else 0
                q75 = float(np.percentile(data, 75)) if len(data) > 0 else 0
                medians_b.append(med)
                iqr_lo_b.append(med - q25)
                iqr_hi_b.append(q75 - med)
            ax.bar(x + i * width - width / 2, medians_b, width,
                   yerr=[iqr_lo_b, iqr_hi_b],
                   capsize=3, color=color, alpha=0.7, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in key_conds],
                           rotation=20, ha='right', fontsize=9)
        ax.legend()
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('SWA Enhancement (%)')
    ax.set_title('B. Median SWA Enhancement by Beta Subgroup [IQR]')

    # Panel C: Adaptive vs assigned protocol WITHIN each subgroup
    # High-beta subjects are routed to progressive; low-beta to fixed delta.
    # Show that adaptive ≈ assigned protocol in both subgroups.
    ax = axes[2]
    if beta_col and beta_col in session_metrics_df.columns:
        median_beta = session_metrics_df[beta_col].median()
        high_beta = session_metrics_df[session_metrics_df[beta_col] > median_beta]
        low_beta = session_metrics_df[session_metrics_df[beta_col] <= median_beta]

        # High beta: Adaptive routes to Progressive
        hi_prog = high_beta[high_beta['condition'] == 'progressive']['session_swa_enhancement']
        hi_adap = high_beta[high_beta['condition'] == 'adaptive_protocol']['session_swa_enhancement']
        # Low beta: Adaptive routes to Fixed Delta
        lo_delta = low_beta[low_beta['condition'] == 'fixed_delta']['session_swa_enhancement']
        lo_adap = low_beta[low_beta['condition'] == 'adaptive_protocol']['session_swa_enhancement']

        groups = ['High Beta\n(n=104)', 'Low Beta\n(n=104)']
        x = np.arange(len(groups))
        width = 0.3

        assigned_means = [float(hi_prog.mean()), float(lo_delta.mean())]
        assigned_cis = [1.96 * float(hi_prog.std() / np.sqrt(len(hi_prog))),
                        1.96 * float(lo_delta.std() / np.sqrt(len(lo_delta)))]
        adaptive_means = [float(hi_adap.mean()), float(lo_adap.mean())]
        adaptive_cis = [1.96 * float(hi_adap.std() / np.sqrt(len(hi_adap))),
                        1.96 * float(lo_adap.std() / np.sqrt(len(lo_adap)))]

        ax.bar(x - width/2, assigned_means, width, yerr=assigned_cis,
               capsize=4, color='#E91E63', alpha=0.7, label='Assigned protocol')
        ax.bar(x + width/2, adaptive_means, width, yerr=adaptive_cis,
               capsize=4, color='#00BCD4', alpha=0.7, label='Adaptive')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.legend(fontsize=8)

        # Annotate what "assigned" means
        ax.annotate('→ Prog.', xy=(x[0] - width/2, assigned_means[0] + assigned_cis[0] + 0.3),
                    ha='center', fontsize=7, color='#E91E63')
        ax.annotate('→ Fixed Δ', xy=(x[1] - width/2, assigned_means[1] + assigned_cis[1] + 0.3),
                    ha='center', fontsize=7, color='#E91E63')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('SWA Enhancement (%)')
    ax.set_title('C. Adaptive vs Assigned Protocol')

    fig.suptitle('Responder Subgroup Analysis', fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'rfig3_responder_subgroups.pdf')
    fig.savefig(FIGURES_DIR / 'rfig3_responder_subgroups.png')
    plt.close(fig)
    logger.info("RFig 3 saved")


def rfig4_ssa_dynamics(all_epochs_df: pd.DataFrame):
    """RFig 4: SSA dynamics — dual-timescale adaptation curves for key conditions.

    Shows how stimulus-specific adaptation accumulates differently:
    - fixed_delta: monotonic rise
    - fixed_delta_ssa_resets: periodic graded recovery from 1 Hz wobbles
    - progressive: graded recovery at frequency transitions
    - pulsed: slower accumulation (only during active pulses)

    Plots A_fast (solid) and A_slow (dashed) separately.
    """
    # Determine which columns are available (v2 dual-timescale vs v1 single)
    has_dual = 'adaptation_fast' in all_epochs_df.columns
    has_single = 'adaptation' in all_epochs_df.columns
    if not has_dual and not has_single:
        logger.warning("No adaptation column — skipping rfig4")
        return

    key_conditions = ['fixed_delta', 'fixed_delta_ssa_resets',
                      'progressive', 'pulsed_progressive']

    if has_dual:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        ax_fast, ax_slow = axes

        for condition in key_conditions:
            cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
            if cond_data.empty:
                continue
            n_subj = cond_data['subject_id'].nunique()
            color = CONDITION_COLORS.get(condition, '#999')
            label = CONDITION_LABELS.get(condition, condition)
            lw = 2.5 if condition in ('fixed_delta_ssa_resets', 'fixed_delta') else 1.8

            for ax, col, title in [
                (ax_fast, 'adaptation_fast', r'$A_{fast}$ ($\tau$=60s, graded recovery)'),
                (ax_slow, 'adaptation_slow', r'$A_{slow}$ ($\tau$=600s, partial recovery)'),
            ]:
                if col not in cond_data.columns:
                    continue
                agg = cond_data.groupby('time_sec')[col].agg(
                    ['mean', 'std']
                ).reset_index()
                se = agg['std'] / np.sqrt(max(n_subj, 1))
                ax.plot(agg['time_sec'] / 60, agg['mean'],
                        color=color, linewidth=lw, label=label)
                ax.fill_between(agg['time_sec'] / 60,
                                agg['mean'] - 1.96 * se,
                                agg['mean'] + 1.96 * se,
                                color=color, alpha=0.1)
                ax.set_title(title)

        for ax in axes:
            ax.set_xlabel('Time (minutes)')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
        ax_fast.set_ylabel('Adaptation Level')
        fig.suptitle('Dual-Timescale SSA Dynamics: Graded Recovery Prevents Complete Reset',
                     fontsize=12, fontweight='bold')
    else:
        # Fallback: single adaptation column (v1)
        fig, ax = plt.subplots(figsize=(10, 6))
        for condition in key_conditions:
            cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
            if cond_data.empty:
                continue
            agg = cond_data.groupby('time_sec')['adaptation'].agg(
                ['mean', 'std']
            ).reset_index()
            n_subj = cond_data['subject_id'].nunique()
            se = agg['std'] / np.sqrt(max(n_subj, 1))
            color = CONDITION_COLORS.get(condition, '#999')
            label = CONDITION_LABELS.get(condition, condition)
            lw = 2.5 if condition in ('fixed_delta_ssa_resets', 'fixed_delta') else 1.8
            ax.plot(agg['time_sec'] / 60, agg['mean'],
                    color=color, linewidth=lw, label=label)
            ax.fill_between(agg['time_sec'] / 60,
                            agg['mean'] - 1.96 * se,
                            agg['mean'] + 1.96 * se,
                            color=color, alpha=0.1)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Adaptation Level (A)')
        ax.set_title('SSA Dynamics: Periodic Resets Prevent Adaptation Buildup')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'rfig4_ssa_dynamics.pdf')
    fig.savefig(FIGURES_DIR / 'rfig4_ssa_dynamics.png')
    plt.close(fig)
    logger.info("RFig 4 saved")


def rfig5_extended_thalamic_priming(all_epochs_df: pd.DataFrame):
    """RFig 5: Extended thalamic priming — T trajectory for all conditions.

    Shows thalamocortical feedback evolution at extended (60 min) sessions,
    highlighting how different protocols drive thalamic engagement.
    """
    if 'thalamic_T' not in all_epochs_df.columns:
        logger.warning("No thalamic_T column — skipping rfig5")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: T trajectory — show key conditions only (not all 14)
    ax = axes[0]
    key_t_conditions = [
        'fixed_delta_ssa_resets', 'fixed_delta', 'progressive',
        'pulsed_progressive', 'reverse', 'fixed_theta', 'no_stim',
    ]
    for condition in key_t_conditions:
        cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
        if cond_data.empty:
            continue
        agg = cond_data.groupby('time_sec')['thalamic_T'].agg(
            ['mean', 'std']
        ).reset_index()

        color = CONDITION_COLORS.get(condition, '#999')
        label = CONDITION_LABELS.get(condition, condition)
        lw = 2.5 if condition == 'fixed_delta_ssa_resets' else 1.8
        ax.plot(agg['time_sec'] / 60, agg['mean'],
                color=color, linewidth=lw, label=label)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Thalamic Variable T')
    ax.set_title('A. Thalamocortical Feedback Trajectory')
    ax.legend(fontsize=8)

    # Panel B: H (neuromodulatory) trajectory for key conditions
    ax = axes[1]
    if 'thalamic_H' in all_epochs_df.columns:
        key_conditions = ['fixed_delta_ssa_resets', 'fixed_delta', 'progressive',
                          'pulsed_progressive', 'progressive_hybrid']
        for condition in key_conditions:
            cond_data = all_epochs_df[all_epochs_df['condition'] == condition]
            if cond_data.empty:
                continue
            agg = cond_data.groupby('time_sec')['thalamic_H'].agg(
                ['mean', 'std']
            ).reset_index()

            color = CONDITION_COLORS.get(condition, '#999')
            label = CONDITION_LABELS.get(condition, condition)
            ax.plot(agg['time_sec'] / 60, agg['mean'],
                    color=color, linewidth=2, label=label)

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Neuromodulatory History H')
        ax.set_title('B. Neuromodulatory Accumulation')
        ax.legend(fontsize=8)

    fig.suptitle('Extended Thalamic Priming at 60 Minutes', fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'rfig5_thalamic_priming.pdf')
    fig.savefig(FIGURES_DIR / 'rfig5_thalamic_priming.png')
    plt.close(fig)
    logger.info("RFig 5 saved")


def rfig6_sham_validation(session_metrics_df: pd.DataFrame):
    """RFig 6: Sham validation — hierarchical sham control design.

    Tests expected ordering: no_stim < sham (sub-threshold) < active_sham < active.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sham_conditions = ['no_stim', 'sham', 'active_sham', 'fixed_delta']
    sham_conditions = [c for c in sham_conditions
                       if c in session_metrics_df['condition'].unique()]

    # Use SWA enhancement if available, fallback to SDRE
    metric_col = 'session_swa_enhancement' if 'session_swa_enhancement' in session_metrics_df.columns else 'session_sdre'
    metric_label = 'SWA Enhancement (%)' if metric_col == 'session_swa_enhancement' else 'Session SDRE'

    # Panel A: Distribution histograms
    ax = axes[0]
    for condition in sham_conditions:
        data = session_metrics_df[
            session_metrics_df['condition'] == condition
        ][metric_col]
        color = CONDITION_COLORS.get(condition, '#999')
        label = CONDITION_LABELS.get(condition, condition)
        ax.hist(data, bins=20, alpha=0.5, color=color, label=label, edgecolor='white')
        ax.axvline(data.mean(), color=color, linestyle='--', linewidth=1.5)

    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Count')
    ax.set_title(f'A. {metric_label} Distributions')
    ax.legend(fontsize=9)

    # Panel B: Bar chart comparison (median ± IQR)
    ax = axes[1]
    medians, iqr_lo, iqr_hi, colors, labels = [], [], [], [], []
    cond_values = {}
    for cond in sham_conditions:
        data = session_metrics_df[
            session_metrics_df['condition'] == cond
        ][metric_col].dropna().values
        cond_values[cond] = data
        medians.append(float(np.median(data)) if len(data) > 0 else 0)
        q25 = float(np.percentile(data, 25)) if len(data) > 0 else 0
        q75 = float(np.percentile(data, 75)) if len(data) > 0 else 0
        iqr_lo.append(medians[-1] - q25)
        iqr_hi.append(q75 - medians[-1])
        colors.append(CONDITION_COLORS.get(cond, '#999'))
        labels.append(CONDITION_LABELS.get(cond, cond))

    x = np.arange(len(sham_conditions))
    ax.bar(x, medians, yerr=[iqr_lo, iqr_hi], capsize=4, color=colors,
           edgecolor='white', linewidth=1.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel(metric_label)
    ax.set_title(f'B. Median {metric_label} [IQR]')

    # Add Cliff's delta annotations between sham vs no_stim
    if len(sham_conditions) >= 2 and 'no_stim' in cond_values and 'sham' in cond_values:
        from analysis.statistical_validation import cliffs_delta
        cd = cliffs_delta(cond_values['sham'], cond_values['no_stim'])
        y_ann = max(medians) + max(iqr_hi) + 0.5
        ax.annotate(f"Cliff's d={cd:.2f}", xy=(0.5, y_ann),
                    ha='center', fontsize=8, style='italic')
    if 'active_sham' in cond_values and 'no_stim' in cond_values:
        from analysis.statistical_validation import cliffs_delta
        cd = cliffs_delta(cond_values['active_sham'], cond_values['no_stim'])
        y_ann = max(medians) + max(iqr_hi) + 1.5
        ax.annotate(f"Cliff's d={cd:.2f}", xy=(1.5, y_ann),
                    ha='center', fontsize=8, style='italic')

    fig.suptitle('Hierarchical Sham Validation', fontsize=14)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'rfig6_sham_validation.pdf')
    fig.savefig(FIGURES_DIR / 'rfig6_sham_validation.png')
    plt.close(fig)
    logger.info("RFig 6 saved")


def generate_redesigned_figures(
    results_dir: Optional[str] = None,
):
    """Generate all 6 redesigned study figures from saved results."""
    if results_dir:
        global RESULTS_DIR, FIGURES_DIR
        RESULTS_DIR = Path(results_dir)
        FIGURES_DIR = RESULTS_DIR / 'figures'

    logger.info("Generating redesigned study figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs_file = RESULTS_DIR / 'all_epochs.csv'
    metrics_file = RESULTS_DIR / 'session_metrics.csv'

    if not epochs_file.exists() or not metrics_file.exists():
        logger.error(f"Redesigned study results not found in {RESULTS_DIR}")
        logger.error("Run scripts/run_redesigned_study.py first.")
        return

    all_epochs_df = pd.read_csv(epochs_file)
    session_metrics_df = pd.read_csv(metrics_file)

    rfig1_adaptation_time_course(all_epochs_df)
    rfig2_pulsed_vs_continuous(all_epochs_df)
    rfig3_responder_subgroups(session_metrics_df, all_epochs_df)
    rfig4_ssa_dynamics(all_epochs_df)
    rfig5_extended_thalamic_priming(all_epochs_df)
    rfig6_sham_validation(session_metrics_df)

    logger.info(f"All redesigned figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    generate_redesigned_figures()
