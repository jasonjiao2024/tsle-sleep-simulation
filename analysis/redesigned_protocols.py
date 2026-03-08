"""
Redesigned Protocol Definitions for Second-Generation Sleep Entrainment Study.

14 conditions (7 original + 7 new) testing:
- Pulsed vs continuous stimulation (SO phase-locked delivery)
- Extended session adaptation hypothesis (crossover at 60+ min)
- SSA reset mechanism (periodic frequency wobbles)
- Individual differences (adaptive protocol based on baseline beta)
- Better sham control (phase-randomized pulses)

Builds on the original 7-condition study findings:
- Progressive descent does NOT beat fixed-delta at default parameters
- But rich mechanistic signatures (TC priming, SSA, individual diffs) exist
- Literature gaps: no pulsed/phase-locked, no extended sessions, poor sham
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.protocol_comparison import (
    EPOCH_SEC,
    PROGRESSIVE_PHASES,
    REVERSE_PHASES,
    SHAM_FREQ_RANGE,
    define_protocols,
)
from analysis.thalamocortical_model import compute_sdr, compute_swa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scale_progressive_phases(total_sec: float) -> List[Dict]:
    """
    Proportionally scale progressive protocol phases to a new total duration.

    The original progressive protocol spans 1800s (30 min) across 4 phases.
    Extra time beyond 1800s is added entirely to the delta phase (the tail),
    testing the adaptation crossover hypothesis.

    Args:
        total_sec: Total session duration in seconds.

    Returns:
        List of phase dicts with scaled durations.
    """
    original_total = sum(p['duration_sec'] for p in PROGRESSIVE_PHASES)

    if total_sec <= original_total:
        # Scale proportionally
        ratio = total_sec / original_total
        return [
            {
                'freq': p['freq'],
                'duration_sec': max(EPOCH_SEC, round(p['duration_sec'] * ratio / EPOCH_SEC) * EPOCH_SEC),
                'name': p['name'],
            }
            for p in PROGRESSIVE_PHASES
        ]

    # Extra time goes to delta tail
    phases = [dict(p) for p in PROGRESSIVE_PHASES]
    extra = total_sec - original_total
    phases[-1]['duration_sec'] += extra
    return phases


def define_redesigned_protocols(
    session_duration_sec: float,
    rng: np.random.Generator,
    baseline_beta: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Define all 14 protocol conditions for the redesigned study.

    Returns a dict mapping condition name to a dict with:
    - 'phases': list of phase dicts
    - 'stim_mode': 'continuous' or 'pulsed'
    - 'description': human-readable description

    Args:
        session_duration_sec: Total session duration (e.g. 3600 for 60 min).
        rng: Random generator for sham/randomized protocols.
        baseline_beta: Subject's baseline beta power for adaptive protocol.
            If None, adaptive defaults to progressive.

    Returns:
        Dict of 14 condition definitions.
    """
    n_epochs = int(session_duration_sec / EPOCH_SEC)
    protocols = {}

    # ─── Original 7 conditions (scaled to session duration) ───────────

    # Scale progressive phases proportionally
    prog_phases = scale_progressive_phases(session_duration_sec)

    # 1. Progressive descent
    protocols['progressive'] = {
        'phases': prog_phases,
        'stim_mode': 'continuous',
        'description': 'Progressive descent (10->8.5->6->2 Hz)',
    }

    # 2. Reverse ascent — scale reverse phases similarly
    if session_duration_sec <= 1800:
        ratio = session_duration_sec / 1800
        rev_phases = [
            {
                'freq': p['freq'],
                'duration_sec': max(EPOCH_SEC, round(p['duration_sec'] * ratio / EPOCH_SEC) * EPOCH_SEC),
                'name': p['name'],
            }
            for p in REVERSE_PHASES
        ]
    else:
        rev_phases = [dict(p) for p in REVERSE_PHASES]
        rev_phases[0]['duration_sec'] += session_duration_sec - 1800
    protocols['reverse'] = {
        'phases': rev_phases,
        'stim_mode': 'continuous',
        'description': 'Reverse ascent (2->6->8.5->10 Hz)',
    }

    # 3. Fixed delta
    protocols['fixed_delta'] = {
        'phases': [{'freq': 2.0, 'duration_sec': session_duration_sec, 'name': 'fixed_delta'}],
        'stim_mode': 'continuous',
        'description': 'Fixed delta (2 Hz) throughout',
    }

    # 4. Fixed theta
    protocols['fixed_theta'] = {
        'phases': [{'freq': 6.0, 'duration_sec': session_duration_sec, 'name': 'fixed_theta'}],
        'stim_mode': 'continuous',
        'description': 'Fixed theta (6 Hz) throughout',
    }

    # 5. Fixed alpha
    protocols['fixed_alpha'] = {
        'phases': [{'freq': 8.5, 'duration_sec': session_duration_sec, 'name': 'fixed_alpha'}],
        'stim_mode': 'continuous',
        'description': 'Fixed alpha (8.5 Hz) throughout',
    }

    # 6. No stimulation
    protocols['no_stim'] = {
        'phases': [{'freq': 0.0, 'duration_sec': session_duration_sec, 'name': 'no_stim'}],
        'stim_mode': 'continuous',
        'description': 'No stimulation (F=0)',
    }

    # 7. Sham (sub-threshold): continuous delta at 20% amplitude
    # Preserves acoustic presence while minimizing entrainment effect.
    # Sub-threshold amplitude (F_sham = 0.02) ensures negligible forcing.
    protocols['sham'] = {
        'phases': [{'freq': 2.0, 'duration_sec': session_duration_sec, 'name': 'sham_subthreshold'}],
        'stim_mode': 'continuous',
        'description': 'Sub-threshold sham (2 Hz, F=0.02)',
        'forcing_override': 0.02,
    }

    # ─── 7 new conditions ────────────────────────────────────────────

    # 8. Progressive extended: progressive phases + extended delta tail
    protocols['progressive_extended'] = {
        'phases': scale_progressive_phases(session_duration_sec),
        'stim_mode': 'continuous',
        'description': 'Progressive with extended delta tail (extra time in delta)',
    }

    # 9. Fixed delta with SSA resets: 2 Hz with brief 1.0 Hz wobbles every 5 min
    ssa_phases = []
    wobble_interval_sec = 300.0  # every 5 min
    wobble_duration_sec = EPOCH_SEC  # one epoch of wobble
    remaining = session_duration_sec
    wobble_count = 0
    while remaining > 0:
        # Delta segment
        delta_dur = min(wobble_interval_sec - wobble_duration_sec, remaining)
        if delta_dur > 0:
            ssa_phases.append({
                'freq': 2.0,
                'duration_sec': delta_dur,
                'name': f'delta_seg_{wobble_count}',
            })
            remaining -= delta_dur
        # Wobble segment (1.0 Hz to trigger SSA reset)
        if remaining > 0:
            wob_dur = min(wobble_duration_sec, remaining)
            ssa_phases.append({
                'freq': 1.0,
                'duration_sec': wob_dur,
                'name': f'wobble_{wobble_count}',
            })
            remaining -= wob_dur
            wobble_count += 1
    protocols['fixed_delta_ssa_resets'] = {
        'phases': ssa_phases,
        'stim_mode': 'continuous',
        'description': 'Fixed delta (2 Hz) with 1 Hz wobbles every 5 min (SSA reset)',
    }

    # 10. Pulsed progressive: progressive phases, pulsed delivery
    protocols['pulsed_progressive'] = {
        'phases': scale_progressive_phases(session_duration_sec),
        'stim_mode': 'pulsed',
        'description': 'Progressive descent with SO phase-locked pulsed delivery',
    }

    # 11. Pulsed fixed delta: 2 Hz pulsed delivery
    protocols['pulsed_fixed_delta'] = {
        'phases': [{'freq': 2.0, 'duration_sec': session_duration_sec, 'name': 'pulsed_delta'}],
        'stim_mode': 'pulsed',
        'description': 'Fixed delta (2 Hz) with SO phase-locked pulsed delivery',
    }

    # 12. Adaptive protocol: selects based on baseline beta power
    # High beta -> progressive (benefits from TC priming); Low beta -> fixed delta
    beta_threshold = 0.25  # median split threshold
    if baseline_beta is not None and baseline_beta > beta_threshold:
        adaptive_phases = scale_progressive_phases(session_duration_sec)
        adaptive_desc = 'Adaptive: progressive (high beta)'
    else:
        adaptive_phases = [
            {'freq': 2.0, 'duration_sec': session_duration_sec, 'name': 'fixed_delta'}
        ]
        adaptive_desc = 'Adaptive: fixed delta (low beta)'
    protocols['adaptive_protocol'] = {
        'phases': adaptive_phases,
        'stim_mode': 'continuous',
        'description': adaptive_desc,
        'adaptive_choice': 'progressive' if (baseline_beta is not None and baseline_beta > beta_threshold) else 'fixed_delta',
    }

    # 13. Active sham: continuous random frequency per epoch (full amplitude)
    # Controls for non-specific acoustic effects at full intensity.
    # Random frequencies disrupt frequency-specific entrainment while
    # preserving continuous delivery (no pulsed vs continuous confound).
    active_sham_phases = []
    for i in range(n_epochs):
        freq = float(rng.uniform(*SHAM_FREQ_RANGE))
        active_sham_phases.append({
            'freq': freq,
            'duration_sec': EPOCH_SEC,
            'name': f'active_sham_epoch_{i}',
        })
    protocols['active_sham'] = {
        'phases': active_sham_phases,
        'stim_mode': 'continuous',
        'description': 'Active sham (random freq per epoch, full F)',
    }

    # 14. Progressive hybrid: continuous first half -> pulsed 2 Hz second half
    half_dur = session_duration_sec / 2.0
    # First half: progressive continuous (scaled to half duration)
    first_half = scale_progressive_phases(half_dur)
    # Second half: pulsed 2 Hz
    second_half = [{'freq': 2.0, 'duration_sec': half_dur, 'name': 'pulsed_delta_tail'}]
    protocols['progressive_hybrid'] = {
        'phases': first_half + second_half,
        'stim_mode': 'hybrid',  # first half continuous, second half pulsed
        'description': 'Progressive continuous first half, pulsed 2 Hz second half',
        'hybrid_split_sec': half_dur,
    }

    # 15. SSA-reset with fast adaptation (tau_slow=300s)
    ssa_fast_phases = []
    remaining = session_duration_sec
    wc = 0
    while remaining > 0:
        delta_dur = min(wobble_interval_sec - wobble_duration_sec, remaining)
        if delta_dur > 0:
            ssa_fast_phases.append({
                'freq': 2.0, 'duration_sec': delta_dur,
                'name': f'delta_seg_{wc}',
            })
            remaining -= delta_dur
        if remaining > 0:
            wob_dur = min(wobble_duration_sec, remaining)
            ssa_fast_phases.append({
                'freq': 1.0, 'duration_sec': wob_dur,
                'name': f'wobble_{wc}',
            })
            remaining -= wob_dur
            wc += 1
    protocols['ssa_reset_fast'] = {
        'phases': ssa_fast_phases,
        'stim_mode': 'continuous',
        'description': 'SSA-reset delta with fast adaptation (tau_slow=300s)',
        'tau_slow_override': 300.0,
    }

    # 16. SSA-reset with slow adaptation (tau_slow=1200s)
    ssa_slow_phases = []
    remaining = session_duration_sec
    wc = 0
    while remaining > 0:
        delta_dur = min(wobble_interval_sec - wobble_duration_sec, remaining)
        if delta_dur > 0:
            ssa_slow_phases.append({
                'freq': 2.0, 'duration_sec': delta_dur,
                'name': f'delta_seg_{wc}',
            })
            remaining -= delta_dur
        if remaining > 0:
            wob_dur = min(wobble_duration_sec, remaining)
            ssa_slow_phases.append({
                'freq': 1.0, 'duration_sec': wob_dur,
                'name': f'wobble_{wc}',
            })
            remaining -= wob_dur
            wc += 1
    protocols['ssa_reset_slow'] = {
        'phases': ssa_slow_phases,
        'stim_mode': 'continuous',
        'description': 'SSA-reset delta with slow adaptation (tau_slow=1200s)',
        'tau_slow_override': 1200.0,
    }

    return protocols


def compute_extended_metrics(session_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute extended metrics for redesigned study analysis.

    1. SWA onset latency: first epoch where SWA > 1.5x baseline SWA
    2. SDR onset latency: first epoch where SDR > 2x baseline SDR (legacy)
    3. SSA-corrected forcing integral: sum(F * (1 - eta_f*A_fast - eta_s*A_slow) * dt)
    4. Thalamic priming index: max T during alpha phase
    5. Post-transition delta boost: delta power jump at frequency transitions

    Args:
        session_df: Per-epoch DataFrame from run_progressive_session().

    Returns:
        Dict with extended metric values.
    """
    metrics = {}

    # 1. SWA onset latency: first epoch where SWA > 1.5x baseline
    if 'swa' in session_df.columns and 'baseline_swa' in session_df.columns:
        baseline_swa = float(session_df['baseline_swa'].iloc[0])
        swa_threshold = 1.5 * max(baseline_swa, 1e-10)
        swa_onset = session_df[session_df['swa'] > swa_threshold]
        if len(swa_onset) > 0:
            metrics['swa_onset_latency_sec'] = float(swa_onset.iloc[0]['time_sec'])
            metrics['swa_onset_latency_epoch'] = int(swa_onset.iloc[0]['epoch_idx'])
        else:
            metrics['swa_onset_latency_sec'] = float(session_df['time_sec'].max())
            metrics['swa_onset_latency_epoch'] = int(session_df['epoch_idx'].max())
    else:
        metrics['swa_onset_latency_sec'] = float(session_df['time_sec'].max())
        metrics['swa_onset_latency_epoch'] = int(session_df['epoch_idx'].max())

    # 2. SDR onset latency (legacy): first epoch where SDR > 2x baseline
    baseline_sdr = float(session_df['baseline_sdr'].iloc[0])
    threshold = 2.0 * max(baseline_sdr, 0.1)
    onset_epochs = session_df[session_df['sdr'] > threshold]
    if len(onset_epochs) > 0:
        metrics['onset_latency_sec'] = float(onset_epochs.iloc[0]['time_sec'])
        metrics['onset_latency_epoch'] = int(onset_epochs.iloc[0]['epoch_idx'])
    else:
        metrics['onset_latency_sec'] = float(session_df['time_sec'].max())
        metrics['onset_latency_epoch'] = int(session_df['epoch_idx'].max())

    # 3. SSA-corrected forcing integral (dual-timescale)
    if 'adaptation_fast' in session_df.columns and 'adaptation_slow' in session_df.columns:
        eta_fast = 0.4
        eta_slow = 0.3
        epoch_sec = float(session_df['time_sec'].diff().median()) if len(session_df) > 1 else 30.0
        forcing_active = (session_df['frequency'] > 0).astype(float)
        corrected_integral = float(
            (forcing_active * (1.0 - eta_fast * session_df['adaptation_fast']
                               - eta_slow * session_df['adaptation_slow']) * epoch_sec).sum()
        )
        metrics['ssa_corrected_forcing_integral'] = corrected_integral
    elif 'adaptation' in session_df.columns:
        # Backward compat: single adaptation column
        eta = 0.6
        epoch_sec = float(session_df['time_sec'].diff().median()) if len(session_df) > 1 else 30.0
        forcing_active = (session_df['frequency'] > 0).astype(float)
        corrected_integral = float(
            (forcing_active * (1.0 - eta * session_df['adaptation']) * epoch_sec).sum()
        )
        metrics['ssa_corrected_forcing_integral'] = corrected_integral
    else:
        metrics['ssa_corrected_forcing_integral'] = 0.0

    # 4. Thalamic priming index: max T during alpha phases
    if 'thalamic_T' in session_df.columns:
        alpha_phases = session_df[
            session_df['phase_name'].str.contains('alpha', case=False, na=False)
        ]
        if len(alpha_phases) > 0:
            metrics['thalamic_priming_index'] = float(alpha_phases['thalamic_T'].max())
        else:
            metrics['thalamic_priming_index'] = float(session_df['thalamic_T'].max())
    else:
        metrics['thalamic_priming_index'] = 0.0

    # 5. Post-transition delta boost: delta power change at frequency transitions
    freq_changes = session_df['frequency'].diff().abs()
    transition_epochs = session_df[freq_changes > 0.5]
    if len(transition_epochs) > 0:
        boosts = []
        for idx in transition_epochs.index:
            pos = session_df.index.get_loc(idx)
            if pos > 0 and pos < len(session_df) - 1:
                before = session_df.iloc[pos - 1]['delta_power']
                after = session_df.iloc[pos]['delta_power']
                boosts.append(after - before)
        metrics['post_transition_delta_boost'] = float(np.mean(boosts)) if boosts else 0.0
    else:
        metrics['post_transition_delta_boost'] = 0.0

    return metrics
