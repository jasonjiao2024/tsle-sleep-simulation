"""
Protocol Comparison Module for Progressive Frequency Descent Study.

Defines 7 stimulation protocols and comparison metrics for evaluating
progressive frequency descent entrainment against fixed-frequency and
control conditions.

7 Protocol Conditions (within-subject):
1. Progressive descent: 10 -> 8.5 -> 6 -> 2 Hz
2. Reverse ascent: 2 -> 6 -> 8.5 -> 10 Hz
3. Fixed delta: 2 Hz throughout
4. Fixed theta: 6 Hz throughout
5. Fixed alpha: 8.5 Hz throughout
6. No stimulation: F=0
7. Sham: random freq per epoch, F>0
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Total session: 60 epochs x 30s = 30 min
TOTAL_EPOCHS = 60
EPOCH_SEC = 30.0
TOTAL_SESSION_SEC = TOTAL_EPOCHS * EPOCH_SEC  # 1800s = 30 min

# Phase timing for progressive protocol (sum = 60 epochs)
PROGRESSIVE_PHASES = [
    {'freq': 10.0, 'duration_sec': 300,  'name': 'alpha_10hz'},    # 5 min, 10 epochs
    {'freq': 8.5,  'duration_sec': 480,  'name': 'alpha_8.5hz'},   # 8 min, 16 epochs
    {'freq': 6.0,  'duration_sec': 600,  'name': 'theta_6hz'},     # 10 min, 20 epochs
    {'freq': 2.0,  'duration_sec': 420,  'name': 'delta_2hz'},     # 7 min, 14 epochs
]

# Reverse protocol: same durations in reverse order
REVERSE_PHASES = [
    {'freq': 2.0,  'duration_sec': 300,  'name': 'delta_2hz'},
    {'freq': 6.0,  'duration_sec': 480,  'name': 'theta_6hz'},
    {'freq': 8.5,  'duration_sec': 600,  'name': 'alpha_8.5hz'},
    {'freq': 10.0, 'duration_sec': 420,  'name': 'alpha_10hz'},
]

# Sham frequencies: drawn from 1-15 Hz range
SHAM_FREQ_RANGE = (1.0, 15.0)


def define_protocols(rng: np.random.Generator = None) -> Dict[str, List[Dict]]:
    """
    Define all 7 protocol conditions as lists of phase dicts.

    Each phase dict has keys: 'freq', 'duration_sec', 'name'.

    Args:
        rng: Random generator for sham frequencies.

    Returns:
        Dict mapping condition name to list of phase dicts.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    protocols = {}

    # 1. Progressive descent
    protocols['progressive'] = list(PROGRESSIVE_PHASES)

    # 2. Reverse ascent
    protocols['reverse'] = list(REVERSE_PHASES)

    # 3. Fixed delta (2 Hz for entire session)
    protocols['fixed_delta'] = [
        {'freq': 2.0, 'duration_sec': TOTAL_SESSION_SEC, 'name': 'fixed_delta'},
    ]

    # 4. Fixed theta (6 Hz)
    protocols['fixed_theta'] = [
        {'freq': 6.0, 'duration_sec': TOTAL_SESSION_SEC, 'name': 'fixed_theta'},
    ]

    # 5. Fixed alpha (8.5 Hz)
    protocols['fixed_alpha'] = [
        {'freq': 8.5, 'duration_sec': TOTAL_SESSION_SEC, 'name': 'fixed_alpha'},
    ]

    # 6. No stimulation
    protocols['no_stim'] = [
        {'freq': 0.0, 'duration_sec': TOTAL_SESSION_SEC, 'name': 'no_stim'},
    ]

    # 7. Sham: random frequency per epoch
    sham_phases = []
    for i in range(TOTAL_EPOCHS):
        freq = float(rng.uniform(*SHAM_FREQ_RANGE))
        sham_phases.append({
            'freq': freq,
            'duration_sec': EPOCH_SEC,
            'name': f'sham_epoch_{i}',
        })
    protocols['sham'] = sham_phases

    return protocols


def compute_session_metrics(session_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute primary and secondary outcome metrics from a session DataFrame.

    Primary outcomes (v2 — SWA-based):
    - session_swa_enhancement: SWA enhancement at session end vs baseline
    - final_swa: mean SWA in last 5 min (10 epochs)
    - cumulative_swa: time-integral of SWA(t) over session

    Legacy primary outcomes (backward compat):
    - session_sdre: SDR at session end minus baseline SDR
    - cumulative_sleep_depth: time-integral of SDR(t) over session
    - final_delta_power: mean delta power in last 5 min (10 epochs)

    Secondary outcomes:
    - mean_plv: overall mean PLV
    - mean_order_parameter: overall mean order parameter
    - final_sdr: SDR at end of session
    - mean_sdre: mean SDRE across all epochs

    Args:
        session_df: Per-epoch DataFrame from run_progressive_session()

    Returns:
        Dict with all computed metrics.
    """
    n_final = min(10, len(session_df))  # last 5 min = 10 epochs

    # Primary outcomes (legacy SDR)
    final_epochs = session_df.tail(n_final)
    session_sdre = float(final_epochs['sdre'].mean())
    cumulative_sleep_depth = float(
        np.trapezoid(session_df['sdr'].values, session_df['time_sec'].values)
    )
    final_delta_power = float(final_epochs['delta_power'].mean())

    # Primary outcomes (v2 SWA)
    if 'swa' in session_df.columns:
        final_swa = float(final_epochs['swa'].mean())
        cumulative_swa = float(
            np.trapezoid(session_df['swa'].values, session_df['time_sec'].values)
        )
        session_swa_enhancement = float(final_epochs['swa_enhancement'].mean())
    else:
        final_swa = 0.0
        cumulative_swa = 0.0
        session_swa_enhancement = 0.0

    # Secondary outcomes
    mean_plv = float(session_df['plv'].mean())
    mean_order_parameter = float(session_df['order_parameter'].mean())
    final_sdr = float(final_epochs['sdr'].mean())
    mean_sdre = float(session_df['sdre'].mean())

    # Per-phase PLV (for progressive protocol analysis)
    phase_plv = {}
    for phase_name in session_df['phase_name'].unique():
        phase_data = session_df[session_df['phase_name'] == phase_name]
        phase_plv[f'plv_{phase_name}'] = float(phase_data['plv'].mean())

    # Per-phase band powers
    phase_bands = {}
    for phase_name in session_df['phase_name'].unique():
        phase_data = session_df[session_df['phase_name'] == phase_name]
        for band in ['delta_power', 'theta_power', 'alpha_power', 'beta_power']:
            phase_bands[f'{band}_{phase_name}'] = float(phase_data[band].mean())

    metrics = {
        # Primary (v2 SWA)
        'session_swa_enhancement': session_swa_enhancement,
        'final_swa': final_swa,
        'cumulative_swa': cumulative_swa,
        # Primary (legacy SDR)
        'session_sdre': session_sdre,
        'cumulative_sleep_depth': cumulative_sleep_depth,
        'final_delta_power': final_delta_power,
        # Secondary
        'mean_plv': mean_plv,
        'mean_order_parameter': mean_order_parameter,
        'final_sdr': final_sdr,
        'mean_sdre': mean_sdre,
        'baseline_sdr': float(session_df['baseline_sdr'].iloc[0]),
    }
    if 'baseline_swa' in session_df.columns:
        metrics['baseline_swa'] = float(session_df['baseline_swa'].iloc[0])
    metrics.update(phase_plv)
    metrics.update(phase_bands)

    return metrics


def aggregate_protocol_results(
    all_results: Dict[str, Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Aggregate per-subject, per-condition session metrics into a comparison DataFrame.

    Args:
        all_results: Nested dict: {subject_id: {condition: session_df}}.

    Returns:
        DataFrame with columns: subject_id, condition, + all metrics.
    """
    rows = []
    for subject_id, conditions in all_results.items():
        for condition, session_df in conditions.items():
            metrics = compute_session_metrics(session_df)
            metrics['subject_id'] = subject_id
            metrics['condition'] = condition
            rows.append(metrics)

    result_df = pd.DataFrame(rows)

    # Reorder columns
    id_cols = ['subject_id', 'condition']
    primary_cols = ['session_swa_enhancement', 'final_swa', 'cumulative_swa',
                    'session_sdre', 'cumulative_sleep_depth', 'final_delta_power']
    secondary_cols = ['mean_plv', 'mean_order_parameter', 'final_sdr',
                      'mean_sdre', 'baseline_sdr', 'baseline_swa']
    other_cols = [c for c in result_df.columns
                  if c not in id_cols + primary_cols + secondary_cols]
    result_df = result_df[id_cols + primary_cols + secondary_cols + other_cols]

    return result_df
