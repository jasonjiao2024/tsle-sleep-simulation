"""
CLAS Empirical Comparison: Kuramoto Model vs. Hebron et al. (2024).

Compares our Kuramoto model predictions against empirical data from
a closed-loop auditory stimulation (CLAS) nap study (Experiment 5).

Hebron et al. used alpha-phase-locked auditory stimulation during
31-min naps. Three conditions: sham, pre-peak, pre-trough.

Comparison points:
1. Phase-locking values: model PLV vs. empirical ecHT_R
2. Alpha power modulation: direction of stim vs sham
3. Sleep depth trajectory: model SDR vs empirical sleep stage
4. Spectral slope: model aperiodic proxy vs empirical exponent

Reference: Hebron et al. (2024). A closed-loop auditory stimulation
approach selectively modulates alpha oscillations and sleep onset
dynamics in humans. PLOS Biology.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.kuramoto_entrainment import KuramotoEnsemble, compute_sdr

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CLAS_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'clas_hebron' / 'HHebron-aclas_PB-e887953' / 'data'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'results' / 'clas_comparison'

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_clas_data():
    """Load empirical CLAS data from Hebron et al."""
    data = {}

    # Fig 6: sleep architecture time courses (3 conditions x 61 epochs x 11 features)
    mat6 = sio.loadmat(str(CLAS_DATA_DIR / 'fig6_data.mat'))
    d6 = mat6['fig6_data'][0, 0]
    data['sleep_timecourse'] = d6['line_plot_data']  # (3, 61, 11)
    names = d6['line_plot_names']
    data['sleep_feature_names'] = [
        str(names[i, 0].flat[0]) if names[i, 0].size > 0 else f'feat_{i}'
        for i in range(names.shape[0])
    ]
    data['sleep_per_subject'] = d6['violin_plot_data']  # (16, 3, 18)

    # Fig 1: phase-locking resultants
    mat1 = sio.loadmat(str(CLAS_DATA_DIR / 'fig1_data.mat'))
    d1 = mat1['fig1_data'][0, 0]
    data['ecHT_R'] = d1['ecHT_R']
    data['frequency_off'] = d1['frequency_off']

    # Fig 5: frequency trajectory during nap
    mat5_2 = sio.loadmat(str(CLAS_DATA_DIR / 'fig5_data_2.mat'))
    d5_2 = mat5_2['fig5_data_2'][0, 0]
    data['freq_line'] = d5_2['frequency_line']  # (16, 179, 2)
    data['freq_violins'] = d5_2['frequency_violins']  # (16, 2, 2)

    return data


def simulate_clas_protocol(n_subjects=16, seed_base=42):
    """
    Simulate a CLAS-like protocol with our Kuramoto model.

    Approximates the Hebron Experiment 5:
    - 31 min nap: 1 min baseline, 15 min stim, 15 min silence
    - 3 conditions: sham (F=0), alpha-stim (10 Hz, moderate F),
      alpha-stim-antiphase (10 Hz, disruptive F)
    - 30-sec epochs
    """
    epoch_sec = 30.0

    # Approximate CLAS conditions
    # Sham: no forcing
    sham_phases = [
        {'freq': 0.0, 'duration_sec': 930, 'name': 'sham'},  # 31 min
    ]

    # Pre-peak stim: 10 Hz forcing for 15 min, then silence
    prepeak_phases = [
        {'freq': 0.0, 'duration_sec': 60, 'name': 'baseline'},    # 1 min
        {'freq': 10.0, 'duration_sec': 900, 'name': 'stim_peak'},  # 15 min
        {'freq': 0.0, 'duration_sec': 900, 'name': 'post_stim'},   # 15 min
    ]

    # Pre-trough stim: same freq but opposite phase effect
    # Modeled as weaker/disruptive forcing
    pretrough_phases = [
        {'freq': 0.0, 'duration_sec': 60, 'name': 'baseline'},
        {'freq': 10.0, 'duration_sec': 900, 'name': 'stim_trough'},
        {'freq': 0.0, 'duration_sec': 900, 'name': 'post_stim'},
    ]

    # Typical wake/drowsy baseline powers
    baseline_powers = {
        'delta_power': 0.20, 'theta_power': 0.20,
        'alpha_power': 0.40, 'beta_power': 0.20,
    }

    conditions = {
        'sham': (sham_phases, 0.0),           # F=0
        'pre_peak': (prepeak_phases, 1.0),     # F=1.0 (constructive)
        'pre_trough': (pretrough_phases, 0.3), # F=0.3 (disruptive/weak)
    }

    all_results = {}
    for cond_name, (phases, forcing) in conditions.items():
        cond_results = []
        for subj_i in range(n_subjects):
            seed = seed_base + subj_i
            ensemble = KuramotoEnsemble(
                n_oscillators=64, coupling_strength=2.0,
                noise_sigma=0.3, dt=0.01, seed=seed,
            )
            ensemble.initialize_from_baseline(baseline_powers, non_responder_fraction=0.30)

            # Get baseline
            ensemble.run_epoch(epoch_sec, 1.0, 0.0)
            bl_bp = ensemble.compute_band_powers()
            bl_sdr = compute_sdr(bl_bp)
            state = ensemble.get_state()

            ensemble.set_state(state)
            session = ensemble.run_progressive_session(
                baseline_powers=baseline_powers,
                protocol_phases=phases,
                forcing_strength=forcing,
                epoch_sec=epoch_sec,
                baseline_sdr=bl_sdr,
                skip_init=True,
            )
            session['subject_idx'] = subj_i
            cond_results.append(session)

        all_results[cond_name] = pd.concat(cond_results, ignore_index=True)

    return all_results


def run_comparison(clas_data, model_results):
    """Compare model predictions against empirical CLAS data."""
    report = {}

    # ─── 1. Phase-Locking Values ───
    logger.info("1. Phase-Locking Value Comparison")

    # Empirical: ecHT_R from experiments (very high, 0.76-0.95 range)
    # Structure: ecHT_R -> 'exp' field -> (1,2) object array -> each element is (N, 4)
    ecHT_R = clas_data['ecHT_R']
    exp_arr = ecHT_R[()]['exp'][0, 0]  # (1, 2) object array
    exp1_R = exp_arr[0, 0]  # experiment 1: (23, 4)
    exp2_R = exp_arr[0, 1]  # experiment 2: (28, 4)
    # Average across both experiments
    emp_plv_mean = float(np.mean([np.mean(exp1_R), np.mean(exp2_R)]))
    emp_plv_std = float(np.mean([np.std(exp1_R), np.std(exp2_R)]))

    # Model: PLV during stimulation epochs
    model_stim = model_results['pre_peak']
    stim_epochs = model_stim[model_stim['phase_name'] == 'stim_peak']
    model_plv_mean = float(stim_epochs['plv'].mean())
    model_plv_std = float(stim_epochs['plv'].std())

    report['phase_locking'] = {
        'empirical_ecHT_R_mean': emp_plv_mean,
        'empirical_ecHT_R_std': emp_plv_std,
        'model_plv_mean': model_plv_mean,
        'model_plv_std': model_plv_std,
        'note': 'ecHT_R measures phase-locking to the stimulation trigger. '
                'Model PLV measures phase-locking to the external drive. '
                'Empirical values are higher because ecHT measures trigger-locked '
                'coherence (closed-loop), while model uses open-loop continuous forcing.',
    }
    logger.info(f"  Empirical ecHT_R: {emp_plv_mean:.3f} +/- {emp_plv_std:.3f}")
    logger.info(f"  Model PLV:        {model_plv_mean:.3f} +/- {model_plv_std:.3f}")

    # ─── 2. Alpha Power Modulation Direction ───
    logger.info("\n2. Alpha Power Modulation Direction")

    # Empirical: alpha abundance time course (feature 7)
    emp_tc = clas_data['sleep_timecourse']  # (3, 61, 11)
    emp_alpha_sham = emp_tc[0, :, 7]       # sham
    emp_alpha_prepeak = emp_tc[1, :, 7]    # pre-peak
    emp_alpha_pretrough = emp_tc[2, :, 7]  # pre-trough

    # Stim period: epochs 2-32 (~minutes 1-16)
    emp_stim_alpha_sham = float(np.mean(emp_alpha_sham[2:32]))
    emp_stim_alpha_peak = float(np.mean(emp_alpha_prepeak[2:32]))
    emp_stim_alpha_trough = float(np.mean(emp_alpha_pretrough[2:32]))

    # Model: alpha power during stim
    model_sham = model_results['sham']
    model_peak = model_results['pre_peak']
    model_trough = model_results['pre_trough']

    model_sham_alpha = float(model_sham.groupby('subject_idx')['alpha_power'].mean().mean())
    stim_peak_data = model_peak[model_peak['phase_name'] == 'stim_peak']
    model_peak_alpha = float(stim_peak_data.groupby('subject_idx')['alpha_power'].mean().mean())
    stim_trough_data = model_trough[model_trough['phase_name'] == 'stim_trough']
    model_trough_alpha = float(stim_trough_data.groupby('subject_idx')['alpha_power'].mean().mean())

    # Direction check
    emp_peak_vs_sham = emp_stim_alpha_peak - emp_stim_alpha_sham
    model_peak_vs_sham = model_peak_alpha - model_sham_alpha

    report['alpha_modulation'] = {
        'empirical_sham_alpha': emp_stim_alpha_sham,
        'empirical_prepeak_alpha': emp_stim_alpha_peak,
        'empirical_pretrough_alpha': emp_stim_alpha_trough,
        'model_sham_alpha': model_sham_alpha,
        'model_prepeak_alpha': model_peak_alpha,
        'model_pretrough_alpha': model_trough_alpha,
        'empirical_peak_minus_sham': float(emp_peak_vs_sham),
        'model_peak_minus_sham': float(model_peak_vs_sham),
        'direction_match': bool(np.sign(emp_peak_vs_sham) == np.sign(model_peak_vs_sham)),
    }
    logger.info(f"  Empirical alpha (sham):     {emp_stim_alpha_sham:.4f}")
    logger.info(f"  Empirical alpha (pre-peak): {emp_stim_alpha_peak:.4f}")
    logger.info(f"  Empirical peak-sham:        {emp_peak_vs_sham:+.4f}")
    logger.info(f"  Model alpha (sham):         {model_sham_alpha:.4f}")
    logger.info(f"  Model alpha (pre-peak):     {model_peak_alpha:.4f}")
    logger.info(f"  Model peak-sham:            {model_peak_vs_sham:+.4f}")
    logger.info(f"  Direction match:            {report['alpha_modulation']['direction_match']}")

    # ─── 3. Sleep Depth Trajectory ───
    logger.info("\n3. Sleep Depth Trajectory Comparison")

    # Empirical: average sleep stage (feature 1) — higher = deeper
    emp_stage_sham = emp_tc[0, :, 1]
    emp_stage_peak = emp_tc[1, :, 1]

    # Model: SDR trajectory (higher = deeper)
    model_sdr_sham = model_sham.groupby('epoch_idx')['sdr'].mean()
    model_sdr_peak = model_peak.groupby('epoch_idx')['sdr'].mean()

    # Both should show: stim condition deeper than sham
    emp_stage_diff = float(np.mean(emp_stage_peak[2:32]) - np.mean(emp_stage_sham[2:32]))
    model_sdr_diff = float(model_sdr_peak.mean() - model_sdr_sham.mean())

    report['sleep_depth'] = {
        'empirical_stage_diff_peak_minus_sham': emp_stage_diff,
        'model_sdr_diff_peak_minus_sham': model_sdr_diff,
        'both_positive': bool(emp_stage_diff > 0 and model_sdr_diff > 0),
        'note': 'Empirical uses sleep stage (0=wake, 1=N1, 2=N2, 3=N3). '
                'Model uses SDR (delta+theta)/(alpha+beta). Both are proxies for sleep depth.',
    }
    logger.info(f"  Empirical stage diff (peak-sham): {emp_stage_diff:+.4f}")
    logger.info(f"  Model SDR diff (peak-sham):       {model_sdr_diff:+.4f}")

    # ─── 4. Aperiodic Slope ───
    logger.info("\n4. Aperiodic Slope (Spectral Steepness)")

    # Empirical: exponent (feature 10) — more negative = steeper = deeper sleep
    emp_exp_sham = emp_tc[0, :, 10]
    emp_exp_peak = emp_tc[1, :, 10]

    emp_slope_sham = float(np.mean(emp_exp_sham[2:32]))
    emp_slope_peak = float(np.mean(emp_exp_peak[2:32]))

    # Model proxy: SDR encodes the same idea (more low-freq power = steeper slope)
    report['aperiodic_slope'] = {
        'empirical_exponent_sham': emp_slope_sham,
        'empirical_exponent_peak': emp_slope_peak,
        'empirical_diff': float(emp_slope_peak - emp_slope_sham),
        'note': 'Model does not directly compute aperiodic exponent. '
                'SDR serves as a proxy: higher SDR implies steeper spectral slope.',
    }
    logger.info(f"  Empirical exponent (sham):     {emp_slope_sham:.4f}")
    logger.info(f"  Empirical exponent (pre-peak): {emp_slope_peak:.4f}")

    # ─── 5. Individual Alpha Frequency ───
    logger.info("\n5. Individual Alpha Frequency")

    freq_off = clas_data['frequency_off']
    freq_exp_arr = freq_off[()]['exp'][0, 0]  # (1, 2) object array
    exp1_freqs = freq_exp_arr[0, 0].flatten()
    exp2_freqs = freq_exp_arr[0, 1].flatten()
    all_freqs = np.concatenate([exp1_freqs, exp2_freqs])

    report['alpha_frequency'] = {
        'empirical_mean_iaf': float(np.mean(all_freqs)),
        'empirical_std_iaf': float(np.std(all_freqs)),
        'empirical_range': [float(np.min(all_freqs)), float(np.max(all_freqs))],
        'model_target_freq': 10.0,
        'note': 'Empirical IAF from off-periods. Model uses 10 Hz as alpha drive.',
    }
    logger.info(f"  Empirical IAF: {np.mean(all_freqs):.2f} +/- {np.std(all_freqs):.2f} Hz")
    logger.info(f"  Range: [{np.min(all_freqs):.2f}, {np.max(all_freqs):.2f}] Hz")

    return report


def generate_comparison_figures(clas_data, model_results, report):
    """Generate comparison figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figs_dir = OUTPUT_DIR / 'figures'
    figs_dir.mkdir(exist_ok=True)

    emp_tc = clas_data['sleep_timecourse']

    # ─── Figure A: Sleep depth trajectory comparison ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Empirical
    ax = axes[0]
    emp_t = np.arange(61) * 0.5  # 30-sec epochs in minutes
    ax.plot(emp_t, emp_tc[0, :, 1], color='#607D8B', linewidth=2, label='Sham')
    ax.plot(emp_t, emp_tc[1, :, 1], color='#E91E63', linewidth=2, label='Pre-peak')
    ax.plot(emp_t, emp_tc[2, :, 1], color='#9C27B0', linewidth=2, label='Pre-trough')
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(15.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Average Sleep Stage')
    ax.set_title('A. Empirical (Hebron et al. 2024)')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 3.0)

    # Model
    ax = axes[1]
    for cond, color, label in [
        ('sham', '#607D8B', 'Sham (F=0)'),
        ('pre_peak', '#E91E63', 'Stim (F=1.0)'),
        ('pre_trough', '#9C27B0', 'Weak stim (F=0.3)'),
    ]:
        df = model_results[cond]
        agg = df.groupby('epoch_idx')['sdr'].agg(['mean', 'std']).reset_index()
        n_subj = df['subject_idx'].nunique()
        se = agg['std'] / np.sqrt(n_subj)
        t = agg['epoch_idx'] * 0.5
        ax.plot(t, agg['mean'], color=color, linewidth=2, label=label)
        ax.fill_between(t, agg['mean'] - 1.96 * se, agg['mean'] + 1.96 * se,
                         color=color, alpha=0.1)

    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(15.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Sleep Depth Ratio (SDR)')
    ax.set_title('B. Kuramoto Model Prediction')
    ax.legend(fontsize=9)

    fig.suptitle('Sleep Depth Trajectory: Empirical vs. Model', fontsize=14)
    fig.tight_layout()
    fig.savefig(figs_dir / 'clas_fig1_sleep_depth_comparison.pdf')
    fig.savefig(figs_dir / 'clas_fig1_sleep_depth_comparison.png')
    plt.close(fig)
    logger.info("CLAS Fig 1 saved")

    # ─── Figure B: Alpha abundance / power comparison ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Empirical alpha abundance
    ax = axes[0]
    ax.plot(emp_t, emp_tc[0, :, 7], color='#607D8B', linewidth=2, label='Sham')
    ax.plot(emp_t, emp_tc[1, :, 7], color='#E91E63', linewidth=2, label='Pre-peak')
    ax.plot(emp_t, emp_tc[2, :, 7], color='#9C27B0', linewidth=2, label='Pre-trough')
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(15.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Alpha Abundance')
    ax.set_title('A. Empirical Alpha Abundance')
    ax.legend(fontsize=9)

    # Model alpha power
    ax = axes[1]
    for cond, color, label in [
        ('sham', '#607D8B', 'Sham'),
        ('pre_peak', '#E91E63', 'Stim (F=1.0)'),
        ('pre_trough', '#9C27B0', 'Weak stim (F=0.3)'),
    ]:
        df = model_results[cond]
        agg = df.groupby('epoch_idx')['alpha_power'].agg(['mean', 'std']).reset_index()
        n_subj = df['subject_idx'].nunique()
        se = agg['std'] / np.sqrt(n_subj)
        t = agg['epoch_idx'] * 0.5
        ax.plot(t, agg['mean'], color=color, linewidth=2, label=label)
        ax.fill_between(t, agg['mean'] - 1.96 * se, agg['mean'] + 1.96 * se,
                         color=color, alpha=0.1)

    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(15.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Alpha Power (normalized)')
    ax.set_title('B. Model Alpha Power')
    ax.legend(fontsize=9)

    fig.suptitle('Alpha Modulation: Empirical vs. Model', fontsize=14)
    fig.tight_layout()
    fig.savefig(figs_dir / 'clas_fig2_alpha_comparison.pdf')
    fig.savefig(figs_dir / 'clas_fig2_alpha_comparison.png')
    plt.close(fig)
    logger.info("CLAS Fig 2 saved")

    # ─── Figure C: PLV / phase-locking bar comparison ───
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Empirical ecHT_R — exp1: (23 participants, 4 conditions)
    ax = axes[0]
    ecHT_R = clas_data['ecHT_R']
    exp_arr = ecHT_R[()]['exp'][0, 0]  # (1, 2) object array
    exp1_R = exp_arr[0, 0]  # (23, 4)
    # Average across conditions (axis=1) to get per-participant PLR
    per_participant_R = np.mean(exp1_R, axis=1)  # (23,)
    ax.bar(range(len(per_participant_R)), per_participant_R,
           color='#2196F3', alpha=0.7, edgecolor='white')
    ax.set_ylabel('Phase-Locking Resultant (ecHT_R)')
    ax.set_xlabel('Participant')
    ax.set_title('A. Empirical Phase-Locking (Exp 1)')
    ax.set_ylim(0, 1)
    ax.axhline(float(np.mean(per_participant_R)), color='red', linestyle='--',
               label=f'Mean: {np.mean(per_participant_R):.3f}')
    ax.legend()

    # Model PLV
    ax = axes[1]
    stim_data = model_results['pre_peak']
    stim_epochs = stim_data[stim_data['phase_name'] == 'stim_peak']
    subj_plv = stim_epochs.groupby('subject_idx')['plv'].mean()
    ax.bar(range(len(subj_plv)), subj_plv.values, color='#E91E63', alpha=0.7,
           edgecolor='white')
    ax.set_ylabel('Phase-Locking Value (PLV)')
    ax.set_xlabel('Subject')
    ax.set_title('B. Model Phase-Locking')
    ax.set_ylim(0, 1)
    ax.axhline(float(subj_plv.mean()), color='red', linestyle='--',
               label=f'Mean: {subj_plv.mean():.3f}')
    ax.legend()

    fig.suptitle('Phase-Locking: Empirical (ecHT_R) vs. Model (PLV)', fontsize=14)
    fig.tight_layout()
    fig.savefig(figs_dir / 'clas_fig3_plv_comparison.pdf')
    fig.savefig(figs_dir / 'clas_fig3_plv_comparison.png')
    plt.close(fig)
    logger.info("CLAS Fig 3 saved")

    # ─── Figure D: Spectral slope / aperiodic ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Empirical exponent
    ax = axes[0]
    ax.plot(emp_t, emp_tc[0, :, 10], color='#607D8B', linewidth=2, label='Sham')
    ax.plot(emp_t, emp_tc[1, :, 10], color='#E91E63', linewidth=2, label='Pre-peak')
    ax.plot(emp_t, emp_tc[2, :, 10], color='#9C27B0', linewidth=2, label='Pre-trough')
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(15.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Aperiodic Exponent')
    ax.set_title('A. Empirical Spectral Slope')
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # Model: SDR as proxy for spectral steepness
    ax = axes[1]
    for cond, color, label in [
        ('sham', '#607D8B', 'Sham'),
        ('pre_peak', '#E91E63', 'Stim (F=1.0)'),
        ('pre_trough', '#9C27B0', 'Weak stim (F=0.3)'),
    ]:
        df = model_results[cond]
        agg = df.groupby('epoch_idx')['sdre'].agg(['mean']).reset_index()
        t = agg['epoch_idx'] * 0.5
        ax.plot(t, agg['mean'], color=color, linewidth=2, label=label)

    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(15.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('SDRE (proxy for spectral steepness)')
    ax.set_title('B. Model SDRE Trajectory')
    ax.legend(fontsize=9)

    fig.suptitle('Spectral Slope: Empirical Exponent vs. Model SDRE', fontsize=14)
    fig.tight_layout()
    fig.savefig(figs_dir / 'clas_fig4_slope_comparison.pdf')
    fig.savefig(figs_dir / 'clas_fig4_slope_comparison.png')
    plt.close(fig)
    logger.info("CLAS Fig 4 saved")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CLAS empirical data...")
    clas_data = load_clas_data()
    logger.info("  Loaded: sleep timecourse, phase-locking, frequency data")

    logger.info("Simulating CLAS-like protocol with Kuramoto model...")
    model_results = simulate_clas_protocol(n_subjects=16, seed_base=42)
    logger.info("  Simulated 16 subjects x 3 conditions")

    logger.info("\nRunning comparison analysis...")
    report = run_comparison(clas_data, model_results)

    # Save report
    with open(OUTPUT_DIR / 'clas_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("\nGenerating comparison figures...")
    generate_comparison_figures(clas_data, model_results, report)

    # Summary
    print("\n" + "=" * 70)
    print("CLAS COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nPhase-Locking:")
    print(f"  Empirical ecHT_R: {report['phase_locking']['empirical_ecHT_R_mean']:.3f}")
    print(f"  Model PLV:        {report['phase_locking']['model_plv_mean']:.3f}")
    print(f"  (Empirical higher because closed-loop is phase-targeted)")
    print(f"\nAlpha Modulation Direction:")
    print(f"  Empirical (peak-sham): {report['alpha_modulation']['empirical_peak_minus_sham']:+.4f}")
    print(f"  Model (peak-sham):     {report['alpha_modulation']['model_peak_minus_sham']:+.4f}")
    print(f"  Direction match:       {report['alpha_modulation']['direction_match']}")
    print(f"\nSleep Depth (stim vs sham):")
    print(f"  Empirical stage diff: {report['sleep_depth']['empirical_stage_diff_peak_minus_sham']:+.4f}")
    print(f"  Model SDR diff:       {report['sleep_depth']['model_sdr_diff_peak_minus_sham']:+.4f}")
    print(f"\nIndividual Alpha Frequency:")
    print(f"  Empirical IAF: {report['alpha_frequency']['empirical_mean_iaf']:.2f} Hz")
    print("=" * 70)


if __name__ == '__main__':
    main()
