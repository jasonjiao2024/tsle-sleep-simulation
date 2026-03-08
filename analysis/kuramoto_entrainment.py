"""
Kuramoto Oscillator Entrainment Model for Sleep Research.

A biophysically grounded model for simulating auditory entrainment effects
on neural oscillations using a forced stochastic Kuramoto ensemble.

Key features:
- Models cortical oscillators as coupled phase-oscillator ensemble
- Derives EEG band powers emergently from mean-field PSD
- Incorporates per-subject non-responder fractions
- Quantifies entrainment via phase-locking value (PLV)
- Supports frequency scanning for resonance discovery

References:
- Breakspear et al. (2010). Generative models of cortical oscillations.
  Frontiers in Human Neuroscience.
- Childs & Strogatz (2008). Stability diagram for the forced Kuramoto
  model. Chaos.
- Acebron et al. (2005). The Kuramoto model: A simple paradigm for
  synchronization phenomena. Reviews of Modern Physics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EEG frequency band edges (Hz)
BAND_EDGES = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
}


def _dominant_frequency(band_powers: Dict[str, float]) -> float:
    """Estimate dominant frequency from normalized band powers."""
    total = 0.0
    weighted_freq = 0.0
    for band, (lo, hi) in BAND_EDGES.items():
        key = f'{band}_power' if f'{band}_power' in band_powers else band
        power = band_powers.get(key, 0.0)
        center = (lo + hi) / 2.0
        weighted_freq += power * center
        total += power
    if total <= 0:
        return 10.0
    return weighted_freq / total


def _spectral_width(band_powers: Dict[str, float]) -> float:
    """Estimate spectral width (spread) from band powers."""
    dom_freq = _dominant_frequency(band_powers)
    total = 0.0
    weighted_var = 0.0
    for band, (lo, hi) in BAND_EDGES.items():
        key = f'{band}_power' if f'{band}_power' in band_powers else band
        power = band_powers.get(key, 0.0)
        center = (lo + hi) / 2.0
        weighted_var += power * (center - dom_freq) ** 2
        total += power
    if total <= 0:
        return 3.0
    return max(np.sqrt(weighted_var / total), 0.5)


def compute_sdr(band_powers: Dict[str, float]) -> float:
    """
    Compute Sleep Depth Ratio from band powers.

    SDR = (delta + theta) / (alpha + beta)

    Higher values indicate deeper sleep-promoting spectral composition.
    """
    delta = band_powers.get('delta_power', 0.0)
    theta = band_powers.get('theta_power', 0.0)
    alpha = band_powers.get('alpha_power', 0.0)
    beta = band_powers.get('beta_power', 0.0)
    denom = alpha + beta
    if denom < 1e-10:
        return 100.0  # cap when wake powers near zero
    return (delta + theta) / denom


class KuramotoEnsemble:
    """
    Stochastic forced Kuramoto oscillator ensemble.

    Models N coupled phase oscillators with:
    - Internal coupling (K): oscillator-oscillator interaction
    - External forcing (F): binaural beat auditory drive
    - Stochastic noise (sigma): biological variability

    The forced stochastic Kuramoto equation:
        dtheta_i/dt = omega_i
                    + (K/N) * sum_j sin(theta_j - theta_i)
                    + F_i * sin(Omega_ext * t - theta_i)
                    + xi_i(t)

    where F_i = 0 for non-responder oscillators.
    """

    def __init__(
        self,
        n_oscillators: int = 64,
        coupling_strength: float = 2.0,
        noise_sigma: float = 0.3,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.N = n_oscillators
        self.K = coupling_strength
        self.sigma = noise_sigma
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # State
        self.phases: np.ndarray = self.rng.uniform(0, 2 * np.pi, self.N)
        self.natural_freqs: np.ndarray = np.zeros(self.N)
        self.forcing_mask: np.ndarray = np.ones(self.N)
        self.t: float = 0.0

        # Buffer for mean-field signal (for PSD computation)
        self._mf_buffer_size = 8192
        self._mf_buffer: np.ndarray = np.zeros(self._mf_buffer_size)
        self._mf_idx: int = 0
        self._mf_sample_interval: int = max(1, int(1.0 / (256.0 * dt)))
        self._step_counter: int = 0

    def initialize_from_baseline(
        self,
        band_powers: Dict[str, float],
        non_responder_fraction: float = 0.30,
    ) -> None:
        """
        Set natural frequencies from the subject's baseline EEG spectrum.

        Natural frequencies are drawn from a Lorentzian (Cauchy) distribution
        centered at the dominant frequency of the baseline EEG.
        """
        center_freq = _dominant_frequency(band_powers)
        spread = _spectral_width(band_powers)

        self.natural_freqs = (
            center_freq
            + spread * np.tan(np.pi * (self.rng.random(self.N) - 0.5))
        )
        self.natural_freqs = np.clip(self.natural_freqs, 0.5, 40.0)
        self.natural_freqs *= 2.0 * np.pi

        n_non = int(self.N * non_responder_fraction)
        self.forcing_mask = np.ones(self.N)
        if n_non > 0:
            idx = self.rng.choice(self.N, size=n_non, replace=False)
            self.forcing_mask[idx] = 0.0

        self.phases = self.rng.uniform(0, 2 * np.pi, self.N)
        self.t = 0.0
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0

    def get_state(self) -> Dict:
        """Save complete ensemble state for later restoration."""
        return {
            'phases': self.phases.copy(),
            'natural_freqs': self.natural_freqs.copy(),
            'forcing_mask': self.forcing_mask.copy(),
            't': self.t,
            '_mf_buffer': self._mf_buffer.copy(),
            '_mf_idx': self._mf_idx,
            '_step_counter': self._step_counter,
            '_rng_state': self.rng.bit_generator.state,
        }

    def set_state(self, state: Dict) -> None:
        """Restore ensemble state from a saved snapshot."""
        self.phases = state['phases'].copy()
        self.natural_freqs = state['natural_freqs'].copy()
        self.forcing_mask = state['forcing_mask'].copy()
        self.t = state['t']
        self._mf_buffer = state['_mf_buffer'].copy()
        self._mf_idx = state['_mf_idx']
        self._step_counter = state['_step_counter']
        self.rng.bit_generator.state = state['_rng_state']

    def _reset_for_scan(self) -> None:
        """Reset phases and buffers for a new frequency scan trial."""
        self.phases = self.rng.uniform(0, 2 * np.pi, self.N)
        self.t = 0.0
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0

    def run_epoch(
        self,
        duration_sec: float,
        external_freq_hz: float,
        forcing_strength: float,
    ) -> None:
        """
        Run the ensemble for a duration using Euler-Maruyama integration.

        Vectorized batch stepping for performance.
        """
        n_steps = int(duration_sec / self.dt)
        omega_ext = 2.0 * np.pi * external_freq_hz
        sqrt_dt = np.sqrt(self.dt)

        all_noise = self.sigma * sqrt_dt * self.rng.standard_normal((n_steps, self.N))

        for step_i in range(n_steps):
            complex_order = np.mean(np.exp(1j * self.phases))
            r = np.abs(complex_order)
            psi = np.angle(complex_order)

            coupling = self.K * r * np.sin(psi - self.phases)
            forcing = (
                forcing_strength * self.forcing_mask
                * np.sin(omega_ext * self.t - self.phases)
            )

            self.phases += (
                self.natural_freqs + coupling + forcing
            ) * self.dt + all_noise[step_i]

            self.phases %= (2.0 * np.pi)
            self.t += self.dt

            self._step_counter += 1
            if self._step_counter % self._mf_sample_interval == 0:
                mf = float(np.mean(np.cos(self.phases)))
                self._mf_buffer[self._mf_idx % self._mf_buffer_size] = mf
                self._mf_idx += 1

    def compute_order_parameter(self) -> Tuple[float, float]:
        """Compute Kuramoto order parameter r and mean phase psi."""
        complex_order = np.mean(np.exp(1j * self.phases))
        r = float(np.abs(complex_order))
        psi = float(np.angle(complex_order))
        return r, psi

    def compute_band_powers(self) -> Dict[str, float]:
        """
        Derive normalized band powers from the mean-field signal PSD.

        The mean field x(t) = (1/N) * sum cos(theta_i(t)) is a synthetic
        signal whose PSD reflects the collective oscillator dynamics.
        """
        n_valid = min(self._mf_idx, self._mf_buffer_size)
        if n_valid < 16:
            return {
                'delta_power': 0.25,
                'theta_power': 0.25,
                'alpha_power': 0.25,
                'beta_power': 0.25,
            }

        mf_signal = self._mf_buffer[:n_valid]
        fs = 1.0 / (self._mf_sample_interval * self.dt)

        nperseg = min(256, n_valid)
        freqs, psd = sp_signal.welch(
            mf_signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
        )

        band_powers = {}
        for band, (lo, hi) in BAND_EDGES.items():
            mask = (freqs >= lo) & (freqs < hi)
            band_powers[f'{band}_power'] = (
                float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
            )

        total = sum(band_powers.values())
        if total > 0:
            band_powers = {k: v / total for k, v in band_powers.items()}
        else:
            band_powers = {
                'delta_power': 0.25,
                'theta_power': 0.25,
                'alpha_power': 0.25,
                'beta_power': 0.25,
            }

        return band_powers

    def compute_plv(self, external_freq_hz: float) -> float:
        """
        Phase-locking value to external drive.

        PLV = |<exp(i * (theta_i - Omega_ext * t))>|
        Averaged over responder oscillators only.
        """
        omega_ext = 2.0 * np.pi * external_freq_hz
        responder_mask = self.forcing_mask > 0
        if not responder_mask.any():
            return 0.0

        phase_diff = self.phases[responder_mask] - omega_ext * self.t
        plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
        return plv

    def frequency_scan(
        self,
        baseline_powers: Dict[str, float],
        test_frequencies: List[float],
        forcing_strength: float = 0.10,
        warmup_sec: float = 5.0,
        measurement_sec: float = 30.0,
        non_responder_fraction: float = 0.30,
    ) -> pd.DataFrame:
        """
        Sweep test frequencies and record emergent spectral properties.

        For each test frequency:
        1. Reset phases (but keep natural frequencies and forcing mask)
        2. Run warmup period to let transients die out
        3. Reset mean-field buffer
        4. Run measurement period
        5. Record band powers, PLV, order parameter, SDR

        Args:
            baseline_powers: Subject's baseline band powers for initialization.
            test_frequencies: List of forcing frequencies to test (Hz).
            forcing_strength: External forcing amplitude F.
            warmup_sec: Warmup duration to skip transients (seconds).
            measurement_sec: Measurement window duration (seconds).
            non_responder_fraction: Fraction of non-responder oscillators.

        Returns:
            DataFrame with columns: frequency, delta_power, theta_power,
            alpha_power, beta_power, plv, order_parameter, sdr, sdre
        """
        # Initialize ensemble from baseline (sets natural freqs + mask)
        self.initialize_from_baseline(
            baseline_powers,
            non_responder_fraction=non_responder_fraction,
        )

        # Save natural frequencies and forcing mask (shared across all freqs)
        saved_natural_freqs = self.natural_freqs.copy()
        saved_forcing_mask = self.forcing_mask.copy()

        # Compute baseline SDR (no forcing)
        self._reset_for_scan()
        self.run_epoch(warmup_sec, 1.0, 0.0)
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0
        self.run_epoch(measurement_sec, 1.0, 0.0)
        baseline_bp = self.compute_band_powers()
        baseline_sdr = compute_sdr(baseline_bp)

        results = []
        for freq in test_frequencies:
            # Reset phases but keep natural frequencies and forcing mask
            self.natural_freqs = saved_natural_freqs.copy()
            self.forcing_mask = saved_forcing_mask.copy()
            self._reset_for_scan()

            # Warmup with forcing
            self.run_epoch(warmup_sec, freq, forcing_strength)

            # Reset buffer for clean measurement
            self._mf_buffer[:] = 0.0
            self._mf_idx = 0
            self._step_counter = 0

            # Measurement period
            self.run_epoch(measurement_sec, freq, forcing_strength)

            # Record results
            bp = self.compute_band_powers()
            r, _ = self.compute_order_parameter()
            plv = self.compute_plv(freq)
            sdr = compute_sdr(bp)
            sdre = sdr - baseline_sdr

            results.append({
                'frequency': freq,
                'delta_power': bp['delta_power'],
                'theta_power': bp['theta_power'],
                'alpha_power': bp['alpha_power'],
                'beta_power': bp['beta_power'],
                'plv': plv,
                'order_parameter': r,
                'sdr': sdr,
                'sdre': sdre,
                'baseline_sdr': baseline_sdr,
            })

        return pd.DataFrame(results)

    def run_progressive_session(
        self,
        baseline_powers: Dict[str, float],
        protocol_phases: List[Dict],
        forcing_strength: float = 0.10,
        epoch_sec: float = 30.0,
        non_responder_fraction: float = 0.30,
        baseline_sdr: Optional[float] = None,
        skip_init: bool = False,
    ) -> pd.DataFrame:
        """
        Simulate a multi-phase entrainment session with continuous oscillator state.

        Unlike frequency_scan(), this method does NOT reset phases between phases.
        Synchronization built at one frequency carries over when the drive shifts,
        enabling progressive entrainment protocols.

        Args:
            baseline_powers: Subject's baseline band powers for initialization.
            protocol_phases: List of phase dicts, each with keys:
                - 'freq': forcing frequency in Hz (0 = no stim)
                - 'duration_sec': total phase duration in seconds
                - 'name': phase label (e.g. 'alpha', 'theta')
            forcing_strength: External forcing amplitude F.
            epoch_sec: Epoch length for measurements (default 30s).
            non_responder_fraction: Fraction of non-responder oscillators.
            baseline_sdr: Pre-computed baseline SDR. If provided, skips
                baseline initialization and epoch (use with set_state()
                for true within-subject designs).
            skip_init: If True, skip initialize_from_baseline (caller
                must have already initialized or restored state).

        Returns:
            DataFrame with per-epoch rows: epoch_idx, time_sec, phase_name,
            frequency, delta/theta/alpha/beta_power, plv, order_parameter,
            sdr, sdre, baseline_sdr
        """
        if not skip_init:
            # Initialize ensemble from baseline
            self.initialize_from_baseline(
                baseline_powers,
                non_responder_fraction=non_responder_fraction,
            )

        if baseline_sdr is None:
            # Run baseline epoch (F=0) to get reference SDR
            self.run_epoch(epoch_sec, 1.0, 0.0)
            baseline_bp = self.compute_band_powers()
            baseline_sdr = compute_sdr(baseline_bp)

        # Reset buffer for session measurement (but keep phases/state)
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0

        results = []
        epoch_idx = 0
        cumulative_time = 0.0

        for phase in protocol_phases:
            freq = phase['freq']
            phase_duration = phase['duration_sec']
            phase_name = phase['name']
            n_epochs = max(1, int(phase_duration / epoch_sec))

            for _ in range(n_epochs):
                # Reset mean-field buffer for this epoch
                self._mf_buffer[:] = 0.0
                self._mf_idx = 0
                self._step_counter = 0

                # Run epoch (continuous state — no phase reset)
                self.run_epoch(epoch_sec, freq if freq > 0 else 1.0,
                               forcing_strength if freq > 0 else 0.0)

                # Compute metrics
                bp = self.compute_band_powers()
                r, _ = self.compute_order_parameter()
                plv = self.compute_plv(freq) if freq > 0 else 0.0
                sdr = compute_sdr(bp)
                sdre = sdr - baseline_sdr

                cumulative_time += epoch_sec

                results.append({
                    'epoch_idx': epoch_idx,
                    'time_sec': cumulative_time,
                    'phase_name': phase_name,
                    'frequency': freq,
                    'delta_power': bp['delta_power'],
                    'theta_power': bp['theta_power'],
                    'alpha_power': bp['alpha_power'],
                    'beta_power': bp['beta_power'],
                    'plv': plv,
                    'order_parameter': r,
                    'sdr': sdr,
                    'sdre': sdre,
                    'baseline_sdr': baseline_sdr,
                })
                epoch_idx += 1

        return pd.DataFrame(results)
