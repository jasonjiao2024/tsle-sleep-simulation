"""
Thalamocortical Stuart-Landau Ensemble (TSLE) for Sleep Entrainment.

A biophysically enriched model replacing the Kuramoto phase-oscillator ensemble
with Stuart-Landau oscillators coupled through a thalamocortical feedback loop.

Key advances over the Kuramoto model:
- Frequency-selective resonance: forcing near natural frequency produces stronger response
- Amplitude dynamics: |z_i| grows under forcing, decays without (no amplitude in Kuramoto)
- Sleep-state dependence via thalamic input current I_sleep
- Thalamocortical feedback loop: T (fast) drives frequency shift, H (slow) drives
  excitability boost via neuromodulatory accumulation
- Stimulus-specific adaptation (SSA): dual-timescale with graded frequency-distance
  recovery (Ulanovsky et al. 2003, 2004). Replaces binary reset with A_fast (tau=60s,
  graded recovery) and A_slow (tau=600s, no recovery).
- Emergent slow oscillation (SO) phase: extracted from low-pass filtered mean-field
  via Hilbert transform, replacing the free-running clock.
- Absolute SWA metric: unnormalized delta-band power as primary outcome measure,
  matching CLAS literature standards (Ngo et al. 2013; Besedovsky et al. 2017).

The core prediction: progressive descent (10->8.5->6->2 Hz) outperforms fixed-delta (2 Hz)
through two mechanisms:
1. Thalamocortical priming: alpha forcing engages the TC loop (high T->H),
   providing a lasting excitability boost when delta forcing begins.
2. Adaptation advantage: progressive triggers graded SSA recovery at frequency
   transitions via A_fast, maintaining higher effective forcing.

Cortical ensemble (N Stuart-Landau oscillators):
    dz_i/dt = (lambda_i(t) + i*omega_i(t))*z_i - |z_i|^2*z_i
              + (K/N)*sum(z_j - z_i) + F_i*exp(i*Omega*t) + sigma*dW_i

Thalamic slow variable:
    tau_T * dT/dt = -T + alpha_TC * A(t) + I_sleep + beta_ext * F

Thalamocortical feedback:
    omega_i(t) = omega_i,0 * (1 - gamma * sigmoid(T - T_half))
    lambda_i(t) = lambda_i,0 + delta_lambda * sigmoid(T - T_half)

References:
- Deco et al. (2017). The dynamics of resting fluctuations in the brain.
  Nature Reviews Neuroscience.
- Breakspear et al. (2010). Generative models of cortical oscillations.
  Frontiers in Human Neuroscience.
- Crunelli & Hughes (2010). The slow (<1 Hz) rhythm of non-REM sleep.
  Sleep Medicine Reviews.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EEG frequency band edges (Hz) — same as Kuramoto model
BAND_EDGES = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
}

# Sleep stage weights for I_sleep
SLEEP_STAGE_WEIGHTS = {
    'W': 0.0, 'Wake': 0.0,
    'N1': 0.2, '1': 0.2,
    'N2': 0.5, '2': 0.5,
    'N3': 0.8, '3': 0.8, '4': 0.8,
    'REM': 0.3, 'R': 0.3,
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
    Compute Sleep Depth Ratio from band powers (backward compat).

    SDR = (delta + theta) / (alpha + beta + eps)
    Higher values indicate deeper sleep-promoting spectral composition.

    The regularization constant eps prevents SDR divergence when
    strong entrainment drives alpha+beta near zero. This is necessary
    because normalized band powers sum to 1, so extreme spectral
    concentration in a single band can make the denominator vanish.
    """
    delta = band_powers.get('delta_power', 0.0)
    theta = band_powers.get('theta_power', 0.0)
    alpha = band_powers.get('alpha_power', 0.0)
    beta = band_powers.get('beta_power', 0.0)
    # Regularization: eps=0.05 caps max SDR at ~20 for normalized powers
    eps = 0.05
    return (delta + theta) / (alpha + beta + eps)


def compute_swa(band_powers: Dict[str, float]) -> float:
    """
    Compute absolute Slow-Wave Activity (SWA).

    SWA = absolute integrated power in the delta band (0.5-4 Hz).
    This is the standard CLAS outcome measure (Ngo et al. 2013, 2015;
    Besedovsky et al. 2017 Nat Commun).

    Args:
        band_powers: Dict containing 'delta_power_abs' key (absolute power).

    Returns:
        Absolute SWA value. Falls back to normalized delta_power if
        absolute power is unavailable (backward compat).
    """
    return band_powers.get('delta_power_abs', band_powers.get('delta_power', 0.0))


def compute_swa_enhancement(stim_swa: float, baseline_swa: float) -> float:
    """
    Compute SWA enhancement as percent change from baseline.

    Literature target: ~18-22% for active vs sham (Besedovsky et al. 2017).
    Sham should produce <15% above no-stimulation baseline.

    Args:
        stim_swa: SWA during stimulation condition.
        baseline_swa: SWA during baseline (no-stimulation) condition.

    Returns:
        Percent change: 100 * (stim - baseline) / baseline.
        Returns 0.0 if baseline is near-zero.
    """
    if baseline_swa < 1e-10:
        return 0.0
    return 100.0 * (stim_swa - baseline_swa) / baseline_swa


def _sigmoid(x: np.ndarray, kappa: float = 3.0) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-kappa * x))."""
    return 1.0 / (1.0 + np.exp(-kappa * np.clip(x, -20.0, 20.0)))


class ThalamocorticalEnsemble:
    """
    Thalamocortical Stuart-Landau Ensemble (TSLE).

    N complex Stuart-Landau oscillators coupled through mean-field interaction
    and driven by external periodic forcing, with a slow thalamic variable
    providing frequency-shifting feedback.

    The Stuart-Landau equation per oscillator:
        dz_i/dt = (lambda_i + i*omega_i)*z_i - |z_i|^2*z_i
                  + (K/N)*sum(z_j - z_i) + F_i*exp(i*Omega*t) + noise

    Thalamic ODE (fast, tau_T=10s):
        tau_T * dT/dt = -T + alpha_TC * overlap(omega_eff, Omega) * mean_amp + I_sleep + beta_ext * F

    Neuromodulatory history ODE (slow, tau_H=600s):
        dH/dt = (T - H) / tau_H

    Two-timescale feedback:
        omega_i(t) = omega_i,0 * (1 - gamma * sigmoid(T - T_half))    [fast freq shift]
        lambda_i(t) = lambda_i,0 + delta_lambda * sigmoid(H - T_half)  [slow excitability]

    Stimulus-specific adaptation (dual-timescale, graded recovery):
        dA_fast/dt = (1 - A_fast) / tau_fast    (graded recovery on freq change)
        dA_slow/dt = (1 - A_slow) / tau_slow    (no recovery on freq change)
        Recovery: A_fast *= exp(-|Δf| / f_scale)
        effective_F = F * (1 - eta_fast*A_fast - eta_slow*A_slow)

    Emergent SO phase:
        Extracted via Hilbert transform of low-pass filtered mean-field.
        Replaces free-running clock for biophysically grounded SO gating.

    Same interface as KuramotoEnsemble for drop-in replacement.
    """

    def __init__(
        self,
        n_oscillators: int = 64,
        coupling_strength: float = 2.0,
        noise_sigma: float = 0.3,
        dt: float = 0.005,
        seed: Optional[int] = None,
        # TSLE-specific parameters
        tau_T: float = 10.0,
        alpha_TC: float = 5.0,
        gamma: float = 0.5,
        kappa: float = 3.0,
        T_half: float = 0.3,
        delta_lambda: float = 0.30,
        beta_ext: float = 0.05,
        lambda_base: float = -0.5,
        # SO-phase excitability gating parameters
        so_freq_hz: float = 0.75,
        so_modulation: float = 0.0,
        so_phase_init: float = 0.0,
    ):
        self.N = n_oscillators
        self.K = coupling_strength
        self.sigma = noise_sigma
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # TSLE parameters
        self.tau_T = tau_T
        self._alpha_TC_base = alpha_TC  # store constructor value for rescaling
        self.alpha_TC = alpha_TC
        self.gamma = gamma
        self.kappa = kappa
        self.T_half = T_half
        self.delta_lambda = delta_lambda
        self.beta_ext = beta_ext
        self.lambda_base = lambda_base
        self.lambda_cap = 0.0  # cap at marginally critical
        self.homeo_rate = 0.001  # homeostatic decay rate for H

        # SO-phase excitability gating — emergent from Hilbert transform
        # so_freq_hz is accepted but ignored (backward compat); SO phase
        # is now extracted from low-pass filtered mean-field dynamics.
        self.so_freq_hz = so_freq_hz  # kept for backward compat, not used
        self.so_modulation = so_modulation
        self.so_phase: float = so_phase_init

        # Emergent SO phase extraction via Hilbert transform
        # Circular buffer for mean-field samples (~256 Hz, 2048 samples ≈ 8s)
        self._so_buffer_size = 2048
        self._so_buffer: np.ndarray = np.zeros(self._so_buffer_size)
        self._so_buf_idx: int = 0
        self._so_buf_filled: bool = False
        self._so_update_interval: int = 256  # update phase every ~256 samples (~1s)
        self._so_sample_counter: int = 0
        # 4th-order Butterworth low-pass at 1.5 Hz for SO extraction
        self._so_fs = 1.0 / (max(1, int(1.0 / (256.0 * dt))) * dt)
        nyq = self._so_fs / 2.0
        cutoff = min(1.5, nyq * 0.9)  # guard against aliasing
        self._so_sos = sp_signal.butter(4, cutoff, btype='low', fs=self._so_fs, output='sos')

        # Cortical state: complex amplitudes z_i
        self.z: np.ndarray = (
            0.1 * self.rng.standard_normal(self.N)
            + 0.1j * self.rng.standard_normal(self.N)
        )

        # Natural frequencies (rad/s) and base excitability
        self.natural_freqs: np.ndarray = np.zeros(self.N)  # omega_i,0 in rad/s
        self.lambda_0: np.ndarray = np.ones(self.N) * lambda_base  # base excitability

        # Forcing mask (non-responders)
        self.forcing_mask: np.ndarray = np.ones(self.N)

        # Thalamic variable (fast, tracks instantaneous resonance)
        self.T: float = 0.0

        # Neuromodulatory history variable (slow, accumulates thalamic activation)
        # Models slow-timescale state changes (adenosine, neuromodulatory tone)
        # that accumulate over minutes and provide lasting excitability boost.
        # T drives fast frequency shift; H drives slow excitability boost.
        self.H: float = 0.0
        self.tau_H: float = 600.0  # 10-minute timescale — persists across phases

        # Stimulus-specific adaptation (SSA): dual-timescale with graded
        # frequency-distance recovery (Ulanovsky et al. 2003, 2004; Nelken 2014).
        #
        # A_fast (tau=60s): fast adaptation, strong graded recovery on freq change
        #   Recovery: A_fast *= exp(-|Δf| / f_scale), f_scale=2.0 Hz
        #   1 Hz wobble → ~39% recovery; 6 Hz jump → ~95% recovery
        # A_slow (tau=600s): slow adaptation, partial recovery on freq change
        #   Recovery: A_slow *= (1 - slow_recovery_frac) on any freq change > 0.1 Hz
        #   Weaker recovery than fast channel; ensures diminishing returns
        # Effective forcing: F * (1 - eta_fast*A_fast - eta_slow*A_slow)
        #   Max reduction: 70% (eta_fast=0.4 + eta_slow=0.3)
        #
        # Parameter justification (Ulanovsky et al. 2004):
        # - Multiple timescales observed: 6.6ms, 150ms, 1.5s, 3-15s, 48s, 630s
        # - tau_fast=60s and tau_slow=600s bracket the slower timescales
        # - Frequency-distance dependence: larger Δf → stronger SSA recovery
        # - Both timescales show some recovery to novel stimuli
        # - Adaptation magnitude 11-72% → our max 70% is realistic
        self.A_fast: float = 0.0       # fast adaptation (0=fresh, 1=fully adapted)
        self.A_slow: float = 0.0       # slow adaptation (0=fresh, 1=fully adapted)
        self.tau_fast: float = 60.0    # fast timescale (1 min)
        self.tau_slow: float = 600.0   # slow timescale (10 min)
        self.eta_fast: float = 0.4     # fast adaptation weight
        self.eta_slow: float = 0.3     # slow adaptation weight
        self.f_scale: float = 2.0      # frequency-distance scale for recovery (Hz)
        self.slow_recovery_frac: float = 0.5  # fraction of A_slow recovery on any freq change
        self._last_forcing_freq: float = -1.0  # track frequency for recovery

        # Sleep input current
        self.I_sleep: float = 0.0

        # Time
        self.t: float = 0.0

        # Pulsed stimulation parameters
        self.pulse_duration_sec = 0.1           # 100ms pulse
        self.pulse_phase_target = 0.0           # up-state peak (phase 0)
        self.pulse_phase_window = np.pi / 4     # ±π/4 around target
        self._pulse_refractory_sec = 0.5        # 500ms refractory
        self._last_pulse_time = -1.0            # last pulse onset time

        # Baseline SWA for enhancement calculation
        self._baseline_swa: Optional[float] = None

        # Mean-field buffer for PSD computation (same as Kuramoto)
        self._mf_buffer_size = 8192
        self._mf_buffer: np.ndarray = np.zeros(self._mf_buffer_size)
        self._mf_idx: int = 0
        self._mf_sample_interval: int = max(1, int(1.0 / (256.0 * dt)))
        self._step_counter: int = 0

    def initialize_from_baseline(
        self,
        band_powers: Dict[str, float],
        non_responder_fraction: float = 0.30,
        sleep_stage_fractions: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Set natural frequencies and excitability from baseline EEG spectrum.

        Natural frequencies are drawn from a truncated Gaussian distribution
        centered at the dominant frequency. This produces a peaked distribution
        that supports frequency-selective resonance: forcing near the center
        frequency entrains more oscillators simultaneously, producing higher R.

        A small subpopulation (15%) is assigned delta-range frequencies to
        represent intrinsic slow-wave generators.

        Args:
            band_powers: Dict with delta_power, theta_power, alpha_power, beta_power.
            non_responder_fraction: Fraction of oscillators unaffected by external drive.
            sleep_stage_fractions: Optional dict of sleep stage fractions for I_sleep.
        """
        center_freq = _dominant_frequency(band_powers)
        spread = _spectral_width(band_powers)

        # Main population: Gaussian distribution centered at dominant freq
        # This creates a peaked distribution enabling frequency-selective resonance
        n_delta_osc = int(0.15 * self.N)  # 15% delta-range subpopulation
        n_main = self.N - n_delta_osc

        main_freqs = center_freq + spread * self.rng.standard_normal(n_main)
        main_freqs = np.clip(main_freqs, 1.0, 30.0)

        # Delta subpopulation: represents thalamocortical slow generators
        delta_freqs = 1.0 + 2.0 * self.rng.random(n_delta_osc)  # 1-3 Hz
        self.natural_freqs = np.concatenate([main_freqs, delta_freqs])
        self.rng.shuffle(self.natural_freqs)
        self.natural_freqs *= 2.0 * np.pi  # convert to rad/s

        # Base excitability: most oscillators are subcritical (lambda < 0,
        # damped without input). Only forcing near their natural frequency
        # sustains oscillation. This creates genuine frequency-selective resonance.
        # Higher alpha power makes alpha-range oscillators less subcritical.
        alpha_power = band_powers.get('alpha_power', 0.25)
        alpha_scale = alpha_power / 0.25  # relative to uniform

        self.lambda_0 = np.full(self.N, self.lambda_base)
        # Per-oscillator variation
        self.lambda_0 += 0.15 * self.rng.standard_normal(self.N)

        # Make alpha-range oscillators less subcritical (easier to excite)
        freq_hz = self.natural_freqs / (2.0 * np.pi)
        alpha_mask = (freq_hz >= 7.0) & (freq_hz <= 14.0)
        self.lambda_0[alpha_mask] += 0.3 * alpha_scale

        # Delta-range oscillators (slow-wave generators) are near-critical.
        # They don't self-sustain strongly on their own; the TC feedback
        # (high H → excitability boost) is what activates them during sleep.
        # This ensures the progressive advantage comes from TC priming,
        # not from intrinsic delta power.
        delta_mask = freq_hz < 4.0
        self.lambda_0[delta_mask] = -0.05  # subcritical; TC feedback activates

        # TC coupling scaling: higher baseline delta -> stronger TC coupling
        delta_power = band_powers.get('delta_power', 0.25)
        delta_scale = delta_power / 0.25
        self.alpha_TC = self._alpha_TC_base * delta_scale

        # Non-responder mask
        n_non = int(self.N * non_responder_fraction)
        self.forcing_mask = np.ones(self.N)
        if n_non > 0:
            idx = self.rng.choice(self.N, size=n_non, replace=False)
            self.forcing_mask[idx] = 0.0

        # Sleep input current
        if sleep_stage_fractions is not None:
            self.I_sleep = 0.0
            for stage, frac in sleep_stage_fractions.items():
                weight = SLEEP_STAGE_WEIGHTS.get(stage, 0.0)
                self.I_sleep += weight * frac
        else:
            self.I_sleep = 0.0

        # Initialize complex state with small random amplitudes
        self.z = (
            0.1 * self.rng.standard_normal(self.N)
            + 0.1j * self.rng.standard_normal(self.N)
        )

        # Reset thalamic variable, history, adaptation, and time
        self.T = 0.0
        self.H = 0.0
        self.A_fast = 0.0
        self.A_slow = 0.0
        self._last_forcing_freq = -1.0
        self._baseline_swa = None
        self.so_phase = 0.0
        self.t = 0.0
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0
        # Reset SO buffer
        self._so_buffer[:] = 0.0
        self._so_buf_idx = 0
        self._so_buf_filled = False
        self._so_sample_counter = 0

    def get_state(self) -> Dict:
        """Save complete ensemble state for later restoration."""
        return {
            'z': self.z.copy(),
            'natural_freqs': self.natural_freqs.copy(),
            'lambda_0': self.lambda_0.copy(),
            'forcing_mask': self.forcing_mask.copy(),
            'T': self.T,
            'H': self.H,
            'A_fast': self.A_fast,
            'A_slow': self.A_slow,
            '_last_forcing_freq': self._last_forcing_freq,
            '_baseline_swa': self._baseline_swa,
            'so_phase': self.so_phase,
            'I_sleep': self.I_sleep,
            'alpha_TC': self.alpha_TC,
            't': self.t,
            '_mf_buffer': self._mf_buffer.copy(),
            '_mf_idx': self._mf_idx,
            '_step_counter': self._step_counter,
            '_so_buffer': self._so_buffer.copy(),
            '_so_buf_idx': self._so_buf_idx,
            '_so_buf_filled': self._so_buf_filled,
            '_so_sample_counter': self._so_sample_counter,
            '_rng_state': self.rng.bit_generator.state,
        }

    def set_state(self, state: Dict) -> None:
        """Restore ensemble state from a saved snapshot."""
        self.z = state['z'].copy()
        self.natural_freqs = state['natural_freqs'].copy()
        self.lambda_0 = state['lambda_0'].copy()
        self.forcing_mask = state['forcing_mask'].copy()
        self.T = state['T']
        self.H = state.get('H', 0.0)
        self.A_fast = state.get('A_fast', state.get('A_hab', 0.0))
        self.A_slow = state.get('A_slow', 0.0)
        self._last_forcing_freq = state.get('_last_forcing_freq', -1.0)
        self._baseline_swa = state.get('_baseline_swa', None)
        self.so_phase = state.get('so_phase', 0.0)
        self.I_sleep = state['I_sleep']
        self.alpha_TC = state['alpha_TC']
        self.t = state['t']
        self._mf_buffer = state['_mf_buffer'].copy()
        self._mf_idx = state['_mf_idx']
        self._step_counter = state['_step_counter']
        if '_so_buffer' in state:
            self._so_buffer = state['_so_buffer'].copy()
            self._so_buf_idx = state['_so_buf_idx']
            self._so_buf_filled = state['_so_buf_filled']
            self._so_sample_counter = state['_so_sample_counter']
        else:
            self._so_buffer[:] = 0.0
            self._so_buf_idx = 0
            self._so_buf_filled = False
            self._so_sample_counter = 0
        self.rng.bit_generator.state = state['_rng_state']

    def _reset_for_scan(self) -> None:
        """Reset oscillator state and buffers for a new frequency scan trial."""
        self.z = (
            0.1 * self.rng.standard_normal(self.N)
            + 0.1j * self.rng.standard_normal(self.N)
        )
        self.T = 0.0
        self.H = 0.0
        self.A_fast = 0.0
        self.A_slow = 0.0
        self._last_forcing_freq = -1.0
        self._baseline_swa = None
        self.so_phase = 0.0
        self.t = 0.0
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0
        self._so_buffer[:] = 0.0
        self._so_buf_idx = 0
        self._so_buf_filled = False
        self._so_sample_counter = 0

    def _effective_params(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute effective omega_i(t), lambda_i(t), and SO gate.

        Two-timescale feedback:
        - T (fast, tau=10s) drives frequency shift: immediate TC effect that
          tracks the current forcing. Reverses quickly when forcing changes.
        - H (slow, tau=600s) drives excitability boost: neuromodulatory
          accumulation from sustained TC engagement. Progressive builds H
          faster via early alpha resonance; the boost persists into later phases.

        SO-phase gating (when so_modulation > 0):
        - so_gate cycles at ~0.75 Hz: 1.0 at SO up-state, 0.0 at down-state
        - Modulates lambda_eff: neurons are more excitable during up-states
        - Also used by callers to modulate forcing effectiveness
        """
        sig_T = _sigmoid(
            np.array([self.T - self.T_half]), self.kappa
        )[0]
        sig_H = _sigmoid(
            np.array([self.H - self.T_half]), self.kappa
        )[0]

        # Frequency shift from T (fast): T high -> frequencies shift toward delta
        omega_eff = self.natural_freqs * (1.0 - self.gamma * sig_T)

        # Excitability boost from H (slow): H high -> more excitable
        lambda_eff = self.lambda_0 + self.delta_lambda * sig_H
        lambda_eff = np.minimum(lambda_eff, self.lambda_cap)  # always subcritical

        # SO-phase excitability gating
        # so_gate: 1.0 at up-state (phase=0), 0.0 at down-state (phase=pi)
        so_gate = 0.5 * (1.0 + np.cos(self.so_phase))
        if self.so_modulation > 0:
            # Modulate excitability: boost during up-state, suppress during down-state
            # so_gate - 0.5 ranges from -0.5 (down) to +0.5 (up), zero-mean
            lambda_eff = lambda_eff * (1.0 + self.so_modulation * (so_gate - 0.5))

        return omega_eff, lambda_eff, so_gate

    def _update_emergent_so_phase(self, mf_real: float) -> None:
        """
        Update emergent SO phase from low-pass filtered mean-field via Hilbert.

        Samples the mean-field into a circular buffer, and every
        _so_update_interval samples applies:
        1. 4th-order Butterworth low-pass at 1.5 Hz (SO band)
        2. Hilbert transform to extract instantaneous phase
        3. Updates self.so_phase with the analytic signal phase

        This replaces the free-running clock (so_phase += 2π * 0.75Hz * dt)
        with a phase that emerges from the collective oscillator dynamics.
        """
        # Store sample in circular buffer
        self._so_buffer[self._so_buf_idx % self._so_buffer_size] = mf_real
        self._so_buf_idx += 1
        if self._so_buf_idx >= self._so_buffer_size:
            self._so_buf_filled = True

        self._so_sample_counter += 1

        # Only update phase every _so_update_interval samples to amortize cost
        if self._so_sample_counter >= self._so_update_interval:
            self._so_sample_counter = 0

            n_valid = self._so_buffer_size if self._so_buf_filled else self._so_buf_idx
            if n_valid < 128:
                # Not enough data yet — keep current phase
                return

            # Get ordered buffer contents
            if self._so_buf_filled:
                start = self._so_buf_idx % self._so_buffer_size
                buf = np.roll(self._so_buffer, -start)
            else:
                buf = self._so_buffer[:n_valid]

            # Apply low-pass filter (SO band extraction)
            try:
                filtered = sp_signal.sosfiltfilt(self._so_sos, buf)
            except ValueError:
                return  # filter failed, keep current phase

            # Hilbert transform for instantaneous phase
            analytic = sp_signal.hilbert(filtered)
            self.so_phase = float(np.angle(analytic[-1])) % (2.0 * np.pi)

    def run_epoch(
        self,
        duration_sec: float,
        external_freq_hz: float,
        forcing_strength: float,
    ) -> None:
        """
        Run the ensemble for a duration using split-step Euler-Maruyama.

        Uses exponential integrator for the rotation part (i*omega*z) to
        avoid numerical instability at high frequencies, then Euler-Maruyama
        for the remaining amplitude dynamics + coupling + forcing.
        """
        n_steps = int(duration_sec / self.dt)
        omega_ext = 2.0 * np.pi * external_freq_hz
        sqrt_dt = np.sqrt(self.dt)
        dt = self.dt

        # Pre-generate complex noise
        noise_real = self.sigma * sqrt_dt * self.rng.standard_normal((n_steps, self.N))
        noise_imag = self.sigma * sqrt_dt * self.rng.standard_normal((n_steps, self.N))

        # Graded SSA: frequency-distance recovery on freq change
        if forcing_strength > 0 and self._last_forcing_freq > 0:
            delta_f = abs(external_freq_hz - self._last_forcing_freq)
            if delta_f > 0.1:  # meaningful frequency change
                novelty = 1.0 - np.exp(-delta_f / self.f_scale)
                self.A_fast *= (1.0 - novelty)  # graded recovery (distance-dependent)
                self.A_slow *= (1.0 - self.slow_recovery_frac)  # fixed partial recovery on any context change
        if forcing_strength > 0:
            self._last_forcing_freq = external_freq_hz

        # Amplitude clamp to prevent overflow (physically: hard saturation)
        AMP_MAX = 2.0

        for step_i in range(n_steps):
            # Effective parameters from thalamocortical feedback
            omega_eff, lambda_eff, so_gate = self._effective_params()

            # Mean field (complex order parameter)
            z_mean = np.mean(self.z)
            R = float(np.abs(z_mean))

            # Split-step: first apply exact rotation
            # z -> z * exp(i * omega_eff * dt)
            rotation = np.exp(1j * omega_eff * dt)
            self.z *= rotation

            # Then apply amplitude dynamics + coupling + forcing with Euler
            abs_z_sq = np.abs(self.z) ** 2

            # Amplitude part: lambda*z - |z|^2*z
            amplitude = lambda_eff * self.z - abs_z_sq * self.z

            # Coupling: K * (z_mean_rotated - z_i)
            z_mean_new = np.mean(self.z)
            coupling = self.K * (z_mean_new - self.z)

            # Dual-timescale SSA: both A_fast and A_slow grow toward 1
            # under sustained forcing. A_fast recovers on freq change
            # (graded by frequency distance); A_slow never recovers.
            if forcing_strength > 0:
                self.A_fast += (1.0 - self.A_fast) / self.tau_fast * dt
                self.A_slow += (1.0 - self.A_slow) / self.tau_slow * dt
            effective_F = forcing_strength * (
                1.0 - self.eta_fast * self.A_fast - self.eta_slow * self.A_slow
            )

            # External forcing with frequency-selective resonance filter.
            # Each oscillator responds preferentially to forcing near its
            # natural frequency: g(detuning) = 1 / (1 + (detuning/bw)^2)
            # This models spectral tuning of neural populations.
            resonance_bw = 2.0 * 2.0 * np.pi  # 2 Hz bandwidth in rad/s
            detuning = omega_eff - omega_ext
            resonance_gain = 1.0 / (1.0 + (detuning / resonance_bw) ** 2)

            # SO-phase gating of forcing: stimuli during down-states are
            # less effective. so_gate is 1.0 at up-state, 0.0 at down-state.
            so_forcing_gate = 1.0
            if self.so_modulation > 0:
                so_forcing_gate = 1.0 + self.so_modulation * (so_gate - 0.5)

            forcing = (
                effective_F * self.forcing_mask * resonance_gain
                * so_forcing_gate
                * np.exp(1j * omega_ext * self.t)
            )

            dz = (amplitude + coupling + forcing) * dt
            dz += noise_real[step_i] + 1j * noise_imag[step_i]

            self.z += dz

            # Clamp amplitudes for numerical safety
            amp = np.abs(self.z)
            overflow = amp > AMP_MAX
            if overflow.any():
                self.z[overflow] *= AMP_MAX / amp[overflow]

            # Thalamic ODE driven by resonance overlap.
            # Resonance overlap measures how well the forcing frequency
            # matches the current effective oscillator frequency distribution.
            # High overlap → strong cortical activation → thalamic engagement.
            # This creates the key frequency-selective TC feedback:
            #   Alpha forcing (near distribution center) → high overlap → T rises
            #   Delta forcing (far from center) → low overlap → T stays low
            if omega_ext > 0 and forcing_strength > 0:
                overlap_bw = 2.0 * 2.0 * np.pi  # 2 Hz bandwidth in rad/s
                freq_match = np.exp(
                    -(omega_eff - omega_ext) ** 2 / (2.0 * overlap_bw ** 2)
                )
                resonance_overlap = float(np.mean(freq_match))
            else:
                resonance_overlap = 0.0

            mean_amp = float(np.mean(np.abs(self.z)))
            thalamic_drive = (
                -self.T
                + self.alpha_TC * resonance_overlap * mean_amp
                + self.I_sleep
                + self.beta_ext * forcing_strength
            )
            self.T += (thalamic_drive / self.tau_T) * dt
            self.T = np.clip(self.T, 0.0, 1.5)  # bounded thalamic variable

            # Slow neuromodulatory history: H tracks T on a 5-minute timescale.
            # Progressive protocol builds high T early (alpha resonance),
            # so H accumulates faster → lasting frequency shift advantage.
            # Homeostatic decay prevents unbounded H accumulation.
            self.H += (self.T - self.H) / self.tau_H * dt - self.H * self.homeo_rate * dt

            self.t += dt

            # Sample mean-field signal for PSD computation and SO extraction
            self._step_counter += 1
            if self._step_counter % self._mf_sample_interval == 0:
                mf = float(np.mean(self.z.real))
                self._mf_buffer[self._mf_idx % self._mf_buffer_size] = mf
                self._mf_idx += 1
                # Update emergent SO phase from mean-field dynamics
                self._update_emergent_so_phase(mf)

    def run_epoch_pulsed(
        self,
        duration_sec: float,
        external_freq_hz: float,
        forcing_strength: float,
    ) -> Dict:
        """
        Run a pulsed stimulation epoch: forcing is gated by mean-field phase.

        Same ODE integration as run_epoch(), but forcing is only delivered
        when the mean-field phase is within pulse_phase_window of the
        target phase (SO up-state). Between pulses, oscillators evolve freely.

        Returns:
            Dict with n_pulses and pulse_duty_cycle for this epoch.
        """
        n_steps = int(duration_sec / self.dt)
        omega_ext = 2.0 * np.pi * external_freq_hz
        sqrt_dt = np.sqrt(self.dt)
        dt = self.dt

        # Pre-generate complex noise
        noise_real = self.sigma * sqrt_dt * self.rng.standard_normal((n_steps, self.N))
        noise_imag = self.sigma * sqrt_dt * self.rng.standard_normal((n_steps, self.N))

        # Graded SSA: frequency-distance recovery on freq change
        if forcing_strength > 0 and self._last_forcing_freq > 0:
            delta_f = abs(external_freq_hz - self._last_forcing_freq)
            if delta_f > 0.1:
                novelty = 1.0 - np.exp(-delta_f / self.f_scale)
                self.A_fast *= (1.0 - novelty)
                self.A_slow *= (1.0 - self.slow_recovery_frac)  # fixed partial recovery
        if forcing_strength > 0:
            self._last_forcing_freq = external_freq_hz

        AMP_MAX = 2.0

        n_pulses = 0
        active_steps = 0

        for step_i in range(n_steps):
            omega_eff, lambda_eff, so_gate = self._effective_params()
            z_mean = np.mean(self.z)

            # Determine if we should deliver a pulse based on emergent SO phase
            # Gate on SO phase: deliver during up-state (phase near 0)
            so_phase_diff = abs(self.so_phase - self.pulse_phase_target)
            if so_phase_diff > np.pi:
                so_phase_diff = 2.0 * np.pi - so_phase_diff
            in_phase_window = so_phase_diff <= self.pulse_phase_window
            past_refractory = (self.t - self._last_pulse_time) >= self._pulse_refractory_sec

            pulse_active = False
            if forcing_strength > 0 and in_phase_window and past_refractory:
                if self._last_pulse_time < 0 or (self.t - self._last_pulse_time) >= self._pulse_refractory_sec:
                    if (self.t - self._last_pulse_time) >= self._pulse_refractory_sec or self._last_pulse_time < 0:
                        self._last_pulse_time = self.t
                        n_pulses += 1
                pulse_active = True
            elif forcing_strength > 0 and self._last_pulse_time >= 0:
                if (self.t - self._last_pulse_time) < self.pulse_duration_sec:
                    pulse_active = True

            if pulse_active:
                active_steps += 1

            # Split-step: exact rotation
            rotation = np.exp(1j * omega_eff * dt)
            self.z *= rotation

            abs_z_sq = np.abs(self.z) ** 2
            amplitude = lambda_eff * self.z - abs_z_sq * self.z

            z_mean_new = np.mean(self.z)
            coupling = self.K * (z_mean_new - self.z)

            # Dual-timescale SSA accumulates only during active pulses
            if pulse_active and forcing_strength > 0:
                self.A_fast += (1.0 - self.A_fast) / self.tau_fast * dt
                self.A_slow += (1.0 - self.A_slow) / self.tau_slow * dt
            effective_F = forcing_strength * (
                1.0 - self.eta_fast * self.A_fast - self.eta_slow * self.A_slow
            )

            # Forcing: gated by pulse state and SO phase
            if pulse_active:
                resonance_bw = 2.0 * 2.0 * np.pi
                detuning = omega_eff - omega_ext
                resonance_gain = 1.0 / (1.0 + (detuning / resonance_bw) ** 2)
                so_forcing_gate = 1.0
                if self.so_modulation > 0:
                    so_forcing_gate = 1.0 + self.so_modulation * (so_gate - 0.5)
                forcing = (
                    effective_F * self.forcing_mask * resonance_gain
                    * so_forcing_gate
                    * np.exp(1j * omega_ext * self.t)
                )
            else:
                forcing = np.zeros(self.N, dtype=complex)

            dz = (amplitude + coupling + forcing) * dt
            dz += noise_real[step_i] + 1j * noise_imag[step_i]
            self.z += dz

            amp = np.abs(self.z)
            overflow = amp > AMP_MAX
            if overflow.any():
                self.z[overflow] *= AMP_MAX / amp[overflow]

            # Thalamic ODE runs continuously (T responds to pulse-evoked synchrony)
            if omega_ext > 0 and forcing_strength > 0:
                overlap_bw = 2.0 * 2.0 * np.pi
                freq_match = np.exp(
                    -(omega_eff - omega_ext) ** 2 / (2.0 * overlap_bw ** 2)
                )
                resonance_overlap = float(np.mean(freq_match))
            else:
                resonance_overlap = 0.0

            mean_amp = float(np.mean(np.abs(self.z)))
            thalamic_drive = (
                -self.T
                + self.alpha_TC * resonance_overlap * mean_amp
                + self.I_sleep
                + self.beta_ext * (forcing_strength if pulse_active else 0.0)
            )
            self.T += (thalamic_drive / self.tau_T) * dt
            self.T = np.clip(self.T, 0.0, 1.5)  # bounded thalamic variable

            self.H += (self.T - self.H) / self.tau_H * dt - self.H * self.homeo_rate * dt

            self.t += dt

            # Sample mean-field signal and update emergent SO phase
            self._step_counter += 1
            if self._step_counter % self._mf_sample_interval == 0:
                mf = float(np.mean(self.z.real))
                self._mf_buffer[self._mf_idx % self._mf_buffer_size] = mf
                self._mf_idx += 1
                self._update_emergent_so_phase(mf)

        pulse_duty_cycle = active_steps / max(n_steps, 1)
        return {
            'n_pulses': n_pulses,
            'pulse_duty_cycle': pulse_duty_cycle,
        }

    def compute_order_parameter(self) -> Tuple[float, float]:
        """Compute order parameter R = |mean(z)| and mean phase."""
        z_mean = np.mean(self.z)
        R = float(np.abs(z_mean))
        psi = float(np.angle(z_mean))
        return R, psi

    def compute_band_powers(self) -> Dict[str, float]:
        """
        Derive band powers from the mean-field signal PSD.

        Returns both normalized (sum=1) and absolute power values.
        The mean field Re(mean(z_i(t))) is a synthetic signal whose PSD
        reflects the collective oscillator dynamics.

        Returns dict with keys:
        - delta_power, theta_power, alpha_power, beta_power (normalized)
        - delta_power_abs, theta_power_abs, alpha_power_abs, beta_power_abs (absolute)
        """
        n_valid = min(self._mf_idx, self._mf_buffer_size)
        if n_valid < 16:
            return {
                'delta_power': 0.25,
                'theta_power': 0.25,
                'alpha_power': 0.25,
                'beta_power': 0.25,
                'delta_power_abs': 0.0,
                'theta_power_abs': 0.0,
                'alpha_power_abs': 0.0,
                'beta_power_abs': 0.0,
            }

        mf_signal = self._mf_buffer[:n_valid]
        fs = 1.0 / (self._mf_sample_interval * self.dt)

        nperseg = min(256, n_valid)
        freqs, psd = sp_signal.welch(
            mf_signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
        )

        # Absolute band powers (unnormalized PSD integration)
        abs_powers = {}
        for band, (lo, hi) in BAND_EDGES.items():
            mask = (freqs >= lo) & (freqs < hi)
            abs_powers[f'{band}_power_abs'] = (
                float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
            )

        # Normalized band powers (backward compat)
        total = sum(abs_powers.values())
        band_powers = {}
        if total > 0:
            for band in BAND_EDGES:
                band_powers[f'{band}_power'] = abs_powers[f'{band}_power_abs'] / total
        else:
            band_powers = {
                'delta_power': 0.25,
                'theta_power': 0.25,
                'alpha_power': 0.25,
                'beta_power': 0.25,
            }

        band_powers.update(abs_powers)
        return band_powers

    def compute_plv(self, external_freq_hz: float) -> float:
        """
        Phase-locking value to external drive.

        PLV = |<exp(i * (angle(z_i) - Omega_ext * t))>|
        Averaged over responder oscillators only.
        """
        omega_ext = 2.0 * np.pi * external_freq_hz
        responder_mask = self.forcing_mask > 0
        if not responder_mask.any():
            return 0.0

        phases = np.angle(self.z[responder_mask])
        phase_diff = phases - omega_ext * self.t
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

        Same interface as KuramotoEnsemble.frequency_scan().
        """
        self.initialize_from_baseline(
            baseline_powers,
            non_responder_fraction=non_responder_fraction,
        )

        saved_natural_freqs = self.natural_freqs.copy()
        saved_lambda_0 = self.lambda_0.copy()
        saved_forcing_mask = self.forcing_mask.copy()
        saved_alpha_TC = self.alpha_TC

        # Baseline SDR and SWA (no forcing)
        self._reset_for_scan()
        self.run_epoch(warmup_sec, 1.0, 0.0)
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0
        self.run_epoch(measurement_sec, 1.0, 0.0)
        baseline_bp = self.compute_band_powers()
        baseline_sdr = compute_sdr(baseline_bp)
        baseline_swa = compute_swa(baseline_bp)

        results = []
        for freq in test_frequencies:
            self.natural_freqs = saved_natural_freqs.copy()
            self.lambda_0 = saved_lambda_0.copy()
            self.forcing_mask = saved_forcing_mask.copy()
            self.alpha_TC = saved_alpha_TC
            self._reset_for_scan()

            self.run_epoch(warmup_sec, freq, forcing_strength)

            self._mf_buffer[:] = 0.0
            self._mf_idx = 0
            self._step_counter = 0

            self.run_epoch(measurement_sec, freq, forcing_strength)

            bp = self.compute_band_powers()
            r, _ = self.compute_order_parameter()
            plv = self.compute_plv(freq)
            sdr = compute_sdr(bp)
            sdre = sdr - baseline_sdr
            swa = compute_swa(bp)
            swa_enh = compute_swa_enhancement(swa, baseline_swa)

            results.append({
                'frequency': freq,
                'delta_power': bp['delta_power'],
                'theta_power': bp['theta_power'],
                'alpha_power': bp['alpha_power'],
                'beta_power': bp['beta_power'],
                'delta_power_abs': bp['delta_power_abs'],
                'theta_power_abs': bp['theta_power_abs'],
                'alpha_power_abs': bp['alpha_power_abs'],
                'beta_power_abs': bp['beta_power_abs'],
                'plv': plv,
                'order_parameter': r,
                'sdr': sdr,
                'sdre': sdre,
                'baseline_sdr': baseline_sdr,
                'swa': swa,
                'swa_enhancement': swa_enh,
                'baseline_swa': baseline_swa,
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
        sleep_stage_fractions: Optional[Dict[str, float]] = None,
        stim_mode: str = 'continuous',
    ) -> pd.DataFrame:
        """
        Simulate a multi-phase entrainment session with continuous oscillator state.

        Same interface as KuramotoEnsemble.run_progressive_session(), with additional
        TSLE-specific output columns: thalamic_T, mean_amplitude, mean_omega_hz.

        Args:
            baseline_powers: Subject's baseline band powers for initialization.
            protocol_phases: List of phase dicts with 'freq', 'duration_sec', 'name'.
            forcing_strength: External forcing amplitude F.
            epoch_sec: Epoch length for measurements (default 30s).
            non_responder_fraction: Fraction of non-responder oscillators.
            baseline_sdr: Pre-computed baseline SDR (for within-subject design).
            skip_init: If True, skip initialization (caller restored state).
            sleep_stage_fractions: Sleep stage distribution for I_sleep.
            stim_mode: 'continuous' (default) or 'pulsed' (SO phase-locked).

        Returns:
            DataFrame with per-epoch rows including all standard columns plus
            thalamic_T, mean_amplitude, mean_omega_hz.
            If stim_mode='pulsed', also includes n_pulses, pulse_duty_cycle.
        """
        if not skip_init:
            self.initialize_from_baseline(
                baseline_powers,
                non_responder_fraction=non_responder_fraction,
                sleep_stage_fractions=sleep_stage_fractions,
            )

        if baseline_sdr is None:
            self.run_epoch(epoch_sec, 1.0, 0.0)
            baseline_bp = self.compute_band_powers()
            baseline_sdr = compute_sdr(baseline_bp)

        # Compute baseline SWA for enhancement calculation
        if not hasattr(self, '_baseline_swa') or self._baseline_swa is None:
            # Re-measure baseline SWA if not already set
            self._baseline_swa = compute_swa(self.compute_band_powers())
        baseline_swa = self._baseline_swa

        # Reset buffer for session measurement (keep z, T, state)
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

                # Run epoch (continuous state)
                epoch_freq = freq if freq > 0 else 1.0
                epoch_forcing = forcing_strength if freq > 0 else 0.0

                pulse_info = {'n_pulses': 0, 'pulse_duty_cycle': 0.0}
                if stim_mode == 'pulsed' and epoch_forcing > 0:
                    pulse_info = self.run_epoch_pulsed(
                        epoch_sec, epoch_freq, epoch_forcing,
                    )
                else:
                    self.run_epoch(epoch_sec, epoch_freq, epoch_forcing)

                # Compute metrics
                bp = self.compute_band_powers()
                r, _ = self.compute_order_parameter()
                plv = self.compute_plv(freq) if freq > 0 else 0.0
                sdr = compute_sdr(bp)
                sdre = sdr - baseline_sdr
                swa = compute_swa(bp)
                swa_enh = compute_swa_enhancement(swa, baseline_swa)

                # TSLE-specific metrics
                thalamic_T = self.T
                thalamic_H = self.H
                adaptation_fast = self.A_fast
                adaptation_slow = self.A_slow
                mean_amplitude = float(np.mean(np.abs(self.z)))
                omega_eff, _, so_gate_val = self._effective_params()
                mean_omega_hz = float(np.mean(np.abs(omega_eff)) / (2.0 * np.pi))

                cumulative_time += epoch_sec

                row = {
                    'epoch_idx': epoch_idx,
                    'time_sec': cumulative_time,
                    'phase_name': phase_name,
                    'frequency': freq,
                    'delta_power': bp['delta_power'],
                    'theta_power': bp['theta_power'],
                    'alpha_power': bp['alpha_power'],
                    'beta_power': bp['beta_power'],
                    'delta_power_abs': bp['delta_power_abs'],
                    'theta_power_abs': bp['theta_power_abs'],
                    'alpha_power_abs': bp['alpha_power_abs'],
                    'beta_power_abs': bp['beta_power_abs'],
                    'plv': plv,
                    'order_parameter': r,
                    'sdr': sdr,
                    'sdre': sdre,
                    'baseline_sdr': baseline_sdr,
                    'swa': swa,
                    'swa_enhancement': swa_enh,
                    'baseline_swa': baseline_swa,
                    # TSLE-specific
                    'thalamic_T': thalamic_T,
                    'thalamic_H': thalamic_H,
                    'adaptation': adaptation_fast,
                    'adaptation_fast': adaptation_fast,
                    'adaptation_slow': adaptation_slow,
                    'mean_amplitude': mean_amplitude,
                    'mean_omega_hz': mean_omega_hz,
                    'so_phase': self.so_phase,
                    'so_gate': so_gate_val,
                }
                if stim_mode == 'pulsed':
                    row['n_pulses'] = pulse_info['n_pulses']
                    row['pulse_duty_cycle'] = pulse_info['pulse_duty_cycle']

                results.append(row)
                epoch_idx += 1

        return pd.DataFrame(results)
