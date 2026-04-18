"""
Wilson-Cowan Thalamocortical Ensemble Model for Sleep Entrainment.

A neural mass model combining Wilson-Cowan cortical dynamics with a
thalamocortical relay circuit for simulating auditory closed-loop
stimulation during sleep (CLAS).

Architecture:
    N=32 cortical E/I populations with steep sigmoid transfer function:
        tau_E * dE_i/dt = -E_i + S(w_EE*E_i - w_EI*I_i - g_adapt*A_i
                                    + I_tonic_i + I_thal + I_ext + I_sleep)
        tau_I * dI_i/dt = -I_i + S(w_IE*E_i - w_II*I_i)
        tau_A * dA_i/dt = E_i - A_i

    where S(x) = 1 / (1 + exp(-a * (x - theta)))
    with a = 1.56 (steeper than standard WC for sharper UP/DOWN)
    and theta = 3.0.

    Shared thalamocortical relay (TC) and reticular nucleus (TRN):
        tau_TC * dTC/dt = -TC + S(w_CT*E_mean - w_RT*TRN + I_T + I_sleep_thal)
        tau_TRN * dTRN/dt = -TRN + S(w_CT_R*E_mean + w_TC_R*TC - w_RR*TRN)

    Heterogeneous adaptation time constants (Gaussian, spread 0.20)
    produce frequency diversity across the ensemble for smooth forced
    response (Kuramoto desynchronization).

    Responder-weighted mean-field readout models Fz electrode sensitivity.

    SSA via calcium-dependent AHP (BiologicalSSA module).

Key features:
    - Steep sigmoid (a=1.56) for sharper UP/DOWN transitions
    - N=32 populations with heterogeneous tau_adapt
    - Direct auditory forcing into the WC integrator
    - Phase-locked (pulsed) and open-loop (continuous) stimulation modes
    - Responder-weighted EEG proxy (Fz electrode model)

References:
- Wilson HR, Cowan JD (1972). Biophys J 12:1-24.
- Ngo HVV et al. (2013). Neuron.
- Besedovsky L et al. (2017). Nature Communications.
- Levenstein D et al. (2019). Nature Communications.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from analysis.ssa_module import BiologicalSSA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────

# EEG frequency band edges (Hz) — matches existing pipeline
BAND_EDGES = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
}

# Sleep stage weights for I_sleep (cortical excitability drive)
SLEEP_STAGE_WEIGHTS = {
    'W': 0.0, 'Wake': 0.0,
    'N1': 0.2, '1': 0.2,
    'N2': 0.5, '2': 0.5,
    'N3': 0.8, '3': 0.8, '4': 0.8,
    'REM': 0.3, 'R': 0.3,
}

# Auditory gain per sleep stage (Campbell & Colrain 2002, adjusted)
AUDITORY_GAIN = {
    'Wake': 1.0, 'W': 1.0,
    'N1': 0.8, '1': 0.8,
    'N2': 0.6, '2': 0.6,
    'N3': 0.45, '3': 0.45, '4': 0.45,
    'REM': 0.7, 'R': 0.7,
}

# K-complex parameters (Halasz 2005; Cash et al. 2009)
# KC_DELTA_BOOST was 1.5 in the hybrid-architecture days when it only
# fed the post-hoc readout multiplier. Reduced to 0.005
# now that kc_boost is passed directly into the WC integrator. At 1.5
# it produced +850% phantom enhancements; at 0.05 it still produced
# +120%; at 0.005 it gives a modest +5-10% contribution that stacks
# with the direct cortical forcing path.
KC_BASE_PROBABILITY = 0.5
KC_DELTA_BOOST = 0.005
KC_DURATION_SEC = 0.5
KC_HABITUATION_RATE = 0.02

# Process S defaults (Achermann & Borbely 2003)
PROCESS_S_TAU_RISE_HR = 18.2
PROCESS_S_TAU_DECAY_HR = 4.2

# NREM-REM cycle period (Feinberg & Floyd 1979)
ULTRADIAN_CYCLE_MIN = 90.0


# ─── Helper functions ─────────────────────────────────────────────────

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


def compute_sdr(band_powers: Dict[str, float]) -> float:
    """
    Compute Sleep Depth Ratio from band powers (backward compat).

    SDR = (delta + theta) / (alpha + beta + eps)
    """
    delta = band_powers.get('delta_power', 0.0)
    theta = band_powers.get('theta_power', 0.0)
    alpha = band_powers.get('alpha_power', 0.0)
    beta = band_powers.get('beta_power', 0.0)
    eps = 0.05
    return (delta + theta) / (alpha + beta + eps)


def compute_swa(band_powers: Dict[str, float]) -> float:
    """
    Compute absolute Slow-Wave Activity (SWA).

    SWA = absolute integrated power in delta band (0.5-4 Hz).
    Standard CLAS outcome (Ngo et al. 2013; Besedovsky et al. 2017).
    """
    return band_powers.get('delta_power_abs', band_powers.get('delta_power', 0.0))


def compute_swa_enhancement(stim_swa: float, baseline_swa: float) -> float:
    """
    Compute SWA enhancement as percent change from baseline.

    Literature target: ~18-22% for active vs sham (Besedovsky et al. 2017).

    IMPORTANT: baseline_swa must be the no-stim (sham) SWA from a matched
    epoch window, NOT the initial-state SWA. For an N-population ensemble
    with desynchronized initial phases, the mean-field delta power at t=0
    is A^2/N, which ramps up to ~A^2 as thalamic coupling synchronizes the
    populations. Using the desynchronized-state power as baseline produces
    a massive spurious enhancement (~39% for no_stim alone). The correct
    baseline is the steady-state no-stim SWA after the thalamocortical
    synchronization transient has settled (burn-in >= 3 epochs = 90 sec).
    """
    if baseline_swa < 1e-10:
        return 0.0
    return 100.0 * (stim_swa - baseline_swa) / baseline_swa


# Number of 30-sec epochs to run before measurement begins, allowing
# the thalamocortical loop to reach its natural partial-synchrony
# attractor. With w_CT*w_TC_ctx/(tau_TC*tau_E) ~ O(10^3), the
# effective Kuramoto coupling is far above critical; transients
# settle in ~30-90 sec.
#
# We use 5 epochs (150 sec) to be conservative:
#   - Fast variables (tau_TC=20ms, tau_adapt=350ms) settle within 1 epoch
#   - Ensemble partial-synchrony attractor needs ~3-5 epochs
#   - Slow homeostatic variable (tau_H=600s) is INTENTIONALLY left
#     unsettled -- it should evolve during the session
# The burn-in is run WITHOUT forcing (no-stim) and the LAST 3 epochs
# of the burn-in are used to compute baseline SDR and SWA (Option A).
# This paired-baseline approach matches real CLAS within-subject design
# where each subject's pre-stim NREM serves as their own control.
BURN_IN_EPOCHS = 14  # 7 minutes of settling (system reaches SS by ~ep10)
BURN_IN_BASELINE_EPOCHS = 8  # last 4 minutes used for baseline averaging


def _sigmoid(x: float, a: float = 1.3, theta: float = 4.0) -> float:
    """
    Sigmoidal transfer function for Wilson-Cowan populations.

    S(x) = 1 / (1 + exp(-a * (x - theta)))

    Numerically stable with clipping.
    """
    arg = -a * (x - theta)
    arg = np.clip(arg, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(arg))


def _sigmoid_vec(x: np.ndarray, a: float = 1.3, theta: float = 4.0) -> np.ndarray:
    """Vectorized sigmoid for array inputs — operates element-wise on numpy arrays."""
    arg = -a * (x - theta)
    arg = np.clip(arg, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(arg))

def _wc_sigmoid(x, a=1.56, theta=3.0):
    """Steep sigmoid transfer function.
    
    20% steeper than standard Wilson-Cowan sigmoid (a=1.56 vs 1.3),
    Steeper than standard Wilson-Cowan (a=1.56 vs 1.3) for sharper
    UP/DOWN state transitions matching cortical slow oscillation data.
    """
    arg = -a * (x - theta)
    arg = np.clip(arg, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(arg))


def _wc_sigmoid_vec(x: np.ndarray, a: float = 1.56, theta: float = 3.0) -> np.ndarray:
    """Vectorized steep sigmoid for array inputs."""
    arg = -a * (x - theta)
    arg = np.clip(arg, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(arg))




# ─── Default literature-constrained parameters ───────────────────────

DEFAULT_PARAMS = {
    # ── Cortical E-I Circuit ──────────────────────────────────────
    # Wilson & Cowan (1972) time constants
    'tau_E': 0.010,       # 10 ms excitatory time constant
    'tau_I': 0.020,       # 20 ms inhibitory time constant

    # Cortical connection weights (Liley et al. 2002)
    'w_EE': 12.0,        # E->E recurrent excitation
    'w_EI': 4.0,         # I->E inhibition
    'w_IE': 13.0,        # E->I excitation
    'w_II': 2.0,         # I->I recurrent inhibition

    # Cortical sigmoid parameters
    'a_ctx': 1.56,       # Steep sigmoid steepness (20% steeper)
    'theta_ctx': 3.0,    # sigmoid threshold

    # Spike-frequency adaptation (Sanchez-Vives & McCormick 2000;
    # Compte et al. 2003; Destexhe 2009)
    #  tau_adapt extended from 350 -> 500 ms and
    # g_adapt raised from 6.0 -> 7.0 to deepen Down states and bring
    # the system closer to the excitable bistable regime where weak
    # forcing can bias noise-driven Up/Down transitions
    # (Levenstein 2019; Jercog 2017).
    'g_adapt': 7.0,
    'tau_adapt': 0.500,

    # Tonic drive to cortex
    # lowered from 1.5 -> 1.30 to bring the operating point
    # closer to the saddle node so noise + adaptation can produce
    # genuine Up/Down switching rather than a stiff square-wave
    # limit cycle.
    'I_tonic': 1.30,

    # ── Ensemble parameters ───────────────────────────────────────
    # N_pop=32: doubled from 16 to give more frequency
    # diversity in the heterogeneous tau_adapt ensemble. 16 populations
    # only sampled 16 distinct natural-SO frequencies, leaving sharp
    # lock-in tongues; 32 populations halve the spacing between
    # neighbouring tongues and produce a smoother forced response.
    'N_pop': 32,
    # K_ensemble: direct inter-cortical mean-field coupling between
    # populations. Represents horizontal cortico-cortical connections
    # (spanning ~3-5mm, fast, strong) that enforce local coherence
    # during SWS (Massimini et al. 2004). Set to 1.5 after
    # the review found that K=0 produced full 1/√N phase
    # cancellation between pulses — populations drifted apart faster
    # than the 1 Hz pulse train could re-synchronize them. K=1.5
    # maintains partial inter-pulse coherence, matching the biological
    # reality that slow waves propagate as coherent traveling waves
    # across local cortical patches (Torres et al. 2021 used spatial
    # propagation; we use mean-field coupling as the analogous term).
    # K_ensemble: K=1.5 was tried but immediately saturated the ensemble
    # (baseline SWA 28x higher, no room for CLAS improvement). The
    # responder-weighted EEG proxy (Fix 1) is the primary fix for the
    # 1/sqrt(N) dilution; coupling should be weak enough to allow
    # independent population dynamics while slightly retarding phase
    # drift. K=0.05 gives effective coupling rate ~5 Hz, comparable to
    # the SO frequency, allowing partial coherence without lock-in.
    'K_ensemble': 0.05,
    'I_tonic_spread': 0.12,  # fractional spread of I_tonic across populations
                              # Creates ~15% variation in intrinsic SO frequency
                              # across populations (Nir et al. 2011), ensuring
                              # moderate desynchronization that forcing must overcome

    #  heterogeneous adaptation time constant across
    # populations. I_tonic_spread alone only shifts excitability — it
    # does NOT decorrelate natural SO frequencies (which are set by
    # tau_adapt and w_EE*g_adapt). Without frequency heterogeneity the
    # ensemble was in a sharp 1:1 Arnold tongue producing either full
    # +200% lock-in or no lock at all. Kuramoto theory: frequency
    # spread breaks sharp mode-locking, producing a smooth graded
    # cohort response matching Ngo/Besedovsky literature.
    # Final calibration: tau_adapt_spread=0.20 with TRUE
    # Gaussian draw and N_pop=32 — gives 32 random natural SO frequencies
    # (vs the previous 16 deterministic linspace values shuffled in
    # order, which produced an artificially regular spacing that the
    # click train could resonate with).
    'tau_adapt_spread': 0.20,

    # ── Self-limiting forcing mechanism ────────────────────────────
    # Prevents runaway synchronization by attenuating forcing as the
    # ensemble coherence R increases. This implements cortical gain
    # control: as populations synchronize, lateral inhibition from
    # surrounding synchronized columns reduces responsiveness to
    # additional external input (Haider et al. 2006 J Neurosci;
    # Destexhe & Contreras 2006 Science).
    #
    # F_eff_i = F * gain_i * (1 - R^coherence_power)
    #
    # where R = ensemble coherence (0-1), gain_i ~ Beta distribution.
    # This creates a self-consistent equation for steady-state R:
    #   R* is the fixed point where forcing-induced sync = heterogeneity-
    #   induced desync. The fixed point yields 10-25% SWA enhancement
    #   (matching Ngo 2013; Besedovsky 2017).
    'coherence_power': 2.0,     # exponent for coherence attenuation
                                 # Higher values -> sharper cutoff near R=1
                                 # 2.0 gives smooth self-limiting
    'gain_alpha': 2.0,          # Beta distribution alpha parameter
    'gain_beta': 3.0,           # Beta distribution beta parameter
                                 # Beta(2,3) -> mean=0.4, moderate heterogeneity
                                 # Reflects variable thalamocortical innervation
                                 # density across cortical columns (Jones 2001)
    'click_jitter_sec': 0.050,  # per-population temporal jitter on clicks (s)
                                 # 50 ms reflects auditory pathway dispersion
                                 # across cortical columns (different distances
                                 # from A1; Kaas & Hackett 2000)
    'non_responder_frac': 0.30, # fraction of populations that don't respond

    # ── Thalamocortical Circuit ───────────────────────────────────
    'tau_TC': 0.020,     # 20 ms TC relay time constant
    'tau_TRN': 0.020,    # 20 ms reticular nucleus time constant

    # Thalamocortical connection weights
    'w_CT': 1.5,         # cortex (E) -> TC relay
    'w_RT': 2.5,         # TRN -> TC inhibition
    'w_CT_R': 1.0,       # cortex (E) -> TRN
    'w_TC_R': 3.0,       # TC -> TRN (recurrent loop)
    'w_RR': 0.5,         # TRN -> TRN self-inhibition
    'w_TC_ctx': 1.0,     # TC -> cortex (E) thalamocortical projection

    # Thalamic sigmoid parameters
    'a_thal': 3.0,       # sigmoid steepness (TC/TRN)
    'theta_thal': 3.0,   # sigmoid threshold (TC/TRN)

    # T-current parameters (Huguenard & McCormick 1992)
    # g_T raised from 0.5 → 2.0 to enable thalamic rebound
    # bursts that amplify evoked responses. Costa 2016 uses g_T=3.0;
    # our 0.5 was preventing the TC relay from producing the rebound-
    # spindle response that is the primary amplification pathway for
    # CLAS-evoked cortical potentials.
    'g_T': 2.0,          # T-current conductance
    'tau_h': 0.080,      # 80 ms de-inactivation time constant
    'theta_h': 0.3,      # de-inactivation threshold

    # ── Noise ─────────────────────────────────────────────────────
    #  raised from 0.008 -> 0.020 to provide noise-
    # aided Up/Down crossings near the bistable fold. Reduced again
    # 0.020 -> 0.012 after the 6-subject validation showed cohort
    # responses 10x larger than the calibrated single-subject result.
    # The 6-subject CV on pulsed was 27% (good), but the mean was
    # 254% (way above the Ngo/Besedovsky 20-25% range). Reducing
    # sigma compresses the inter-subject variance via the Kramers
    # escape time scaling exp(-DeltaU/sigma^2): going 0.020 -> 0.012
    # changes sigma^2 by 0.36x, dramatically narrowing the response
    # spread without removing the stochastic-resonance mechanism.
    'sigma': 0.012,

    'tau_fast': 60.0,
    'tau_slow': 600.0,
    'eta_fast': 0.4,
    'eta_slow': 0.3,
    'f_scale': 2.0,
    'slow_recovery_frac': 0.5,

    # Phase efficacy gain for pulsed delivery.
    # With tau_adapt heterogeneity + Ngo 2015 refractory gain decay
    # (both added), the refractory decay alone absorbs ~30%
    # of the base pulse amplitude at steady state, and the tau_adapt
    # heterogeneity prevents the cortex from locking 1:1. To compensate
    # for both smoothing effects and land at ~+20-25% pulsed effect,
    # we bump phase_efficacy_gain to 3.5 (which was the original value
    # calibrated for the Ngo 2013 +25% effect).
    'phase_efficacy_gain': 3.5,

    # Ngo 2015 refractory gain decay (empirically fit to the self-limiting
    # 2-click saturation finding). After each pulse, a decay variable r
    # rises by alpha_ref and then relaxes with tau_r. Effective pulse
    # amplitude is multiplied by (1 - r). Re-tuned to a light
    # touch: alpha=0.08, tau=1.5 → steady state r ≈ 0.11, ~89% effective
    # pulse amplitude. The Kuramoto desync from N_pop=32 + Gaussian
    # tau_adapt is the primary smoothing mechanism; Ngo refractory is
    # a secondary modest brake.
    # Ngo 2015 refractory gain decay DISABLED (alpha=0,).
    # At any nonzero alpha the refractory absorbs the already-small
    # per-pulse drive before it can accumulate across pulses. With
    # tau_adapt heterogeneity (Gaussian N=32 spread=0.20) providing
    # the primary smoothing of the Arnold tongue, and the sigmoid
    # operating above threshold limiting the marginal gain, the Ngo
    # refractory is no longer needed and was actively killing the
    # SWA effect. The "self-limiting" behavior is now produced
    # naturally by adaptation (g_adapt=7.0, tau_adapt=0.500) which
    # terminates UP states after each pulse-evoked transition.
    'refractory_alpha': 0.0,
    'refractory_tau': 1.5,
    # Adaptation modulation during stim — reduces g_adapt by this fraction
    # while click envelope is active. Reduced from 0.10 -> 0.005 in Apr
    # 2026 because adapt_mod is now actually passed into the integrator
    # (it was zeroed out in the hybrid architecture). Even 0.02 produced
    # +57% paired SWA on its own; 0.005 keeps it as a small modulator
    # rather than the dominant forcing path.
    'entrainment_boost': 0.005,

    # Forcing gain: scales TSLE-range forcing_strength (0-0.10) to
    # Wilson-Cowan input scale. With self-limiting coherence feedback,
    # the forcing saturates at partial synchronization regardless of gain.
    'forcing_gain': 4.0,   # Calibrated
                            # (Ngo 2013 ~11%, Besedovsky 2017 ~20%)

    # Pulsed stimulation parameters
    # NOTE: phase_window tightened from pi/2 to pi/4 (±45° around SO peak)
    # to match real CLAS targeting (Choi 2020; Schreiner 2018).
    # refractory lengthened from 0.3 s to 1.0 s — one SO period — to
    # match the natural spindle/SWA refractoriness (Ngo 2015).
    # tau_response shortened from 0.3 s to 0.06 s — closer to a 50–80 ms
    # pulse envelope rather than a 300 ms exponential blur (Costa 2016;
    # Mendoza-Halliday 2022).
    'pulse_phase_window': np.pi / 4,
    'pulse_refractory_sec': 1.0,
    # tau_response: returned to 0.15 s from 0.06 s. The
    # 60ms pulse was too brief — each pulse delivered forcing for only
    # 3.6% of the epoch, so the time-averaged cortical input was
    # negligible even at high amplitude. At 0.15 s the pulse covers
    # ~10% of each inter-pulse interval, giving 5x more cumulative
    # drive. This is still shorter than the Costa 2016 80ms square
    # pulse but allows the exponential envelope to deposit more energy.
    'pulse_tau_response': 0.15,

    # ── WC-direct forcing scale ────────────────────────────────────
    # Multiplier on the WC-bound forcing path. Final calibration (v7):
    # With N_pop=32, Gaussian tau_adapt heterogeneity spread=0.20, and
    # light Ngo refractory (alpha=0.08), the sharp Arnold-tongue lock-in
    # is broken and pulses produce additive (not multiplicative) effects.
    # 0.25 gives each pulse enough cortical drive to measurably bias
    # the Up/Down transitions without triggering full lock-in.
    # wc_forcing_scale: raised to 1.0 (, final calibration).
    # At 0.25 the per-pulse cortical input was ~0.12, which shifts
    # sigmoid output by only 0.033 (sigmoid is above threshold at
    # the operating point, so marginal gain is small). At 1.0 the
    # per-pulse input is ~0.50, producing a 0.10 sigmoid shift —
    # enough to transiently bias the Up/Down transition. The Arnold-
    # tongue lock-in is now prevented by tau_adapt heterogeneity
    # (N=32, Gaussian spread 0.20) and the Ngo refractory.
    'wc_forcing_scale': 1.0,
}


# ─── Main Model Class ────────────────────────────────────────────────

class WCThalamocorticalEnsemble:
    """
    Thalamocortical Wilson-Cowan (TCWC) model with N-population cortical
    ensemble for sleep entrainment.

    N cortical E/I populations with heterogeneous I_tonic share a single
    thalamocortical circuit (TC relay + TRN). Mean-field coupling between
    populations (Kuramoto-like) enables synchronization by external forcing,
    which is the mechanism for SWA enhancement.

    Drop-in replacement for ThalamocorticalEnsemble (TSLE) in the existing
    simulation pipeline. All public methods and return formats are preserved.
    """

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        # Legacy TSLE-compatible constructor arguments
        rng_seed: Optional[int] = None,
        n_oscillators: int = 64,
        coupling_strength: float = 4.0,
        noise_sigma: float = 0.15,
        dt: float = 0.005,
        tau_T: float = 10.0,
        alpha_TC: float = 1.5,
        gamma: float = 0.5,
        kappa: float = 3.0,
        T_half: float = 0.3,
        delta_lambda: float = 0.30,
        beta_ext: float = 1.5,
        lambda_base: float = 0.1,
        so_freq_hz: float = 0.75,
        so_modulation: float = 0.0,
        so_phase_init: float = 0.0,
        **legacy_kwargs,
    ):
        # Merge parameters: explicit params dict overrides defaults
        self.p = dict(DEFAULT_PARAMS)
        if params is not None:
            self.p.update(params)

        # Map legacy noise_sigma if provided and no explicit sigma in params
        if params is None or 'sigma' not in params:
            if noise_sigma != 0.15:
                self.p['sigma'] = noise_sigma * 0.33

        # Integration step
        self.dt = dt

        # RNG — separate generators for WC noise and auxiliary randomness.
        # This ensures the WC noise sequence is identical regardless of
        # forcing condition (KC checks and z-array consume RNG differently
        # for stim vs no-stim, which would shift the WC noise and create
        # spurious enhancement if a single RNG were shared).
        effective_seed = seed if seed is not None else rng_seed
        self.rng = np.random.default_rng(effective_seed)
        # Auxiliary RNG for KC probability, z-array noise, etc.
        self._aux_rng = np.random.default_rng(
            effective_seed + 1000 if effective_seed is not None else None
        )

        # Legacy compatibility attributes
        self.N = n_oscillators
        self.K = coupling_strength
        self.sigma = self.p['sigma']
        self._alpha_TC_base = alpha_TC
        self.alpha_TC = alpha_TC
        self.gamma = gamma
        self.kappa = kappa
        self.T_half = T_half
        self.delta_lambda = delta_lambda
        self.beta_ext = beta_ext
        self.lambda_base = lambda_base
        self.so_modulation = so_modulation
        self.so_phase: float = so_phase_init

        # ─── Ensemble parameters ─────────────────────────────────
        self.N_pop: int = int(self.p.get('N_pop', 16))

        # Heterogeneous I_tonic across populations
        I_tonic_base = self.p['I_tonic']
        spread = self.p.get('I_tonic_spread', 0.12)
        self.I_tonic_pop: np.ndarray = I_tonic_base * (
            1.0 + spread * np.linspace(-1, 1, self.N_pop)
        )

        # Heterogeneous tau_adapt across populations — breaks sharp
        # Arnold-tongue lock-in by giving each population a slightly
        # different natural SO frequency (Kuramoto desynchronization).
        # Use TRUE Gaussian draw (not linspace+shuffle) so the natural
        # frequencies are genuinely random across the ensemble — this
        # is what Kuramoto theory needs to break 1:1 mode locking.
        tau_adapt_base = self.p['tau_adapt']
        tau_spread = self.p.get('tau_adapt_spread', 0.20)
        tau_offsets = self.rng.standard_normal(self.N_pop)
        self.tau_adapt_pop: np.ndarray = np.clip(
            tau_adapt_base * (1.0 + tau_spread * tau_offsets),
            tau_adapt_base * 0.3,  # don't let tau go below 30% of base
            tau_adapt_base * 2.5,  # or above 250%
        )

        # Forcing gain distribution: Beta-distributed continuous gains
        # instead of binary mask. Reflects heterogeneous thalamocortical
        # innervation density and auditory pathway effectiveness across
        # cortical columns (Jones 2001 Phil Trans R Soc).
        # Non-responders (gain=0) are set explicitly; responders draw
        # from Beta(alpha, beta) for graded responsiveness.
        nr_frac = self.p.get('non_responder_frac', 0.30)
        n_non = max(0, int(self.N_pop * nr_frac))
        ga = self.p.get('gain_alpha', 2.0)
        gb = self.p.get('gain_beta', 3.0)
        self.forcing_gain_pop: np.ndarray = self.rng.beta(ga, gb, size=self.N_pop)
        # Set non-responders to zero gain (last n_non populations)
        if n_non > 0:
            self.forcing_gain_pop[-n_non:] = 0.0

        # Per-population click arrival jitter (seconds).
        # Each population has a fixed temporal offset for auditory click
        # arrival, reflecting different path lengths from primary auditory
        # cortex (Kaas & Hackett 2000 PNAS). This prevents all populations
        # from receiving the same phase-reset impulse simultaneously.
        jitter_sigma = self.p.get('click_jitter_sec', 0.050)
        self.click_jitter_pop: np.ndarray = (
            jitter_sigma * self.rng.standard_normal(self.N_pop)
        )

        # Legacy binary mask (kept for backward compat with pipeline)
        self.forcing_mask_pop: np.ndarray = (self.forcing_gain_pop > 0).astype(float)

        # ─── State variables (ensemble) ──────────────────────────
        # Cortical populations: arrays of shape (N_pop,)
        self.E_pop: np.ndarray = np.full(self.N_pop, 0.1)
        self.I_pop: np.ndarray = np.full(self.N_pop, 0.05)
        self.A_pop: np.ndarray = np.zeros(self.N_pop)

        # Scalar mean-field E for backward compatibility
        self.E: float = float(np.mean(self.E_pop))
        self.I: float = float(np.mean(self.I_pop))
        self.A_adapt: float = float(np.mean(self.A_pop))

        # Thalamic populations (shared, scalar)
        self.TC: float = 0.1
        self.TRN: float = 0.05

        # T-current de-inactivation variable
        self.h_T: float = 0.5

        # Process S (sleep homeostasis)
        self.S: float = 0.6
        self._S_init: float = 0.6  # preserved across resets
        self._is_sleeping: bool = True

        # NREM-REM cycling
        self._session_time: float = 0.0
        self._ultradian_phase_offset: float = 0.0

        # Stage-aware initialization metadata
        self._onset_sleep_stage: str = ''

        # Thalamic feedback variables (legacy-compatible names)
        self.T: float = 0.0
        self.H: float = 0.0
        self.tau_H: float = 600.0
        self.homeo_rate: float = 0.001

        # Stimulus-specific adaptation — delegated to BiologicalSSA
        self.biological_ssa = BiologicalSSA()
        # Legacy A_fast / A_slow kept for LOGGING only
        self.A_fast: float = 0.0
        self.A_slow: float = 0.0
        # Deprecated instance attrs — kept so old serialization still loads
        self.tau_fast: float = self.p['tau_fast']
        self.tau_slow: float = self.p['tau_slow']
        self.eta_fast: float = self.p['eta_fast']
        self.eta_slow: float = self.p['eta_slow']
        self.f_scale: float = self.p['f_scale']
        self.slow_recovery_frac: float = self.p['slow_recovery_frac']
        self._last_forcing_freq: float = -1.0

        # Sleep input current
        self.I_sleep: float = 0.0
        self._sleep_stage: str = 'N2'
        self._sleep_stage_fractions: Optional[Dict[str, float]] = None

        # K-complex state
        self._kc_habituation: float = 0.0
        self._kc_active: bool = False
        self._kc_remaining_sec: float = 0.0

        # Time
        self.t: float = 0.0

        # Pulsed stimulation state
        self._last_pulse_time: float = -1.0
        self._pulse_envelope: float = 0.0
        self._pulse_onset_amplitude: float = 0.0

        # Continuous-click state (periodic DC pulses at fixed interval)
        self._last_click_time: float = -1.0
        self._click_envelope: float = 0.0
        self._click_onset_amplitude: float = 0.0

        # Diagnostic: log SO phase at every delivered pulse so we can
        # build a phase-histogram and verify closed-loop targeting
        # actually concentrates pulses near the SO peak.
        self._pulse_phase_log: List[float] = []
        self._pulse_count: int = 0

        # K-complex event log: timestamps of each KC trigger
        self._kc_event_times: List[float] = []
        # Pulse delivery event times (for ERP-locked analysis)
        self._pulse_event_times: List[float] = []

        # Ngo 2015 refractory gain: rises on each pulse by alpha, decays
        # with tau_r. Effective pulse amplitude is (1 - r) * base.
        self._refractory_gain: float = 0.0

        # Track previous forcing phase for stimulus-event detection
        self._prev_forcing_phase: float = 0.0

        # Absolute start time for mean-field buffer (PLV phase alignment)
        self._mf_start_time: float = 0.0

        # Running coherence estimator for self-limiting forcing.
        # Tracks exponentially-weighted mean-field variance over a ~2s window.
        # Higher variance = more synchronized UP/DOWN transitions.
        # tau_coherence ~ 2 SO cycles = 2s at 1 Hz.
        self._mf_ema: float = 0.5       # exponential moving average of mf
        self._mf_emvar: float = 0.01    # exponential moving variance of mf
        self._coherence_tau: float = 2.0  # seconds, smoothing timescale
        self._R_smooth: float = 0.08     # smoothed Kuramoto coherence

        # Legacy-compatible arrays
        self.natural_freqs: np.ndarray = np.zeros(n_oscillators)
        self.lambda_0: np.ndarray = np.ones(n_oscillators) * lambda_base
        self.forcing_mask: np.ndarray = np.ones(n_oscillators)
        self.z: np.ndarray = np.zeros(n_oscillators, dtype=complex)

        # Mean-field buffer for PSD computation (sampled at ~256 Hz)
        self._mf_buffer_size = 8192
        self._mf_buffer: np.ndarray = np.zeros(self._mf_buffer_size)
        self._mf_idx: int = 0
        self._mf_sample_interval: int = max(1, int(1.0 / (256.0 * dt)))
        self._step_counter: int = 0
        self._mf_fs: float = 1.0 / (self._mf_sample_interval * dt)

        # SO phase extraction buffer (for pulsed stimulation)
        self._so_buffer_size = 2048
        self._so_buffer: np.ndarray = np.zeros(self._so_buffer_size)
        self._so_buf_idx: int = 0
        self._so_buf_filled: bool = False
        # Update SO phase every 32 mean-field samples (~125 ms at 256 Hz),
        # not every 256 samples (~1 s) — the previous interval allowed
        # the trigger to use a phase frozen up to a full SO cycle ago.
        self._so_update_interval: int = 32
        self._so_sample_counter: int = 0
        # Butterworth filter for SO extraction (< 1.5 Hz). We still use
        # forward-backward filtering on a buffer of past data so the
        # detected phase is the *current* phase of past samples — this
        # is the closest analog to a real CLAS PLL (Santostasi 2016)
        # which forecasts ~80 ms ahead from past data. To honestly model
        # the ~80 ms total system latency of a real device we shift the
        # detected phase forward by `phase_lead_sec * 2 * pi * f_so`,
        # which corresponds to predicting where the SO phase WILL be
        # when the click hits. Default 0 ms (perfect oracle); set to
        # 0.08 s for realistic CLAS device modeling.
        nyq = self._mf_fs / 2.0
        cutoff = min(1.5, nyq * 0.9)
        if cutoff > 0 and nyq > 0:
            self._so_sos = sp_signal.butter(
                4, cutoff, btype='low', fs=self._mf_fs, output='sos'
            )
        else:
            self._so_sos = None
        self._so_phase_lead_sec: float = 0.0  # set per-experiment

        # Baseline SWA for enhancement calculations
        self._baseline_swa: Optional[float] = None

    # ─── Initialization from subject data ─────────────────────────

    def initialize_from_baseline(
        self,
        band_powers: Dict[str, float],
        non_responder_fraction: float = 0.30,
        sleep_stage_fractions: Optional[Dict[str, float]] = None,
        S_init: Optional[float] = None,
        baseline_delta_at_onset: Optional[float] = None,
        sleep_stage: Optional[str] = None,
        process_s_init: Optional[float] = None,
        night_fraction: float = 0.0,
    ) -> None:
        """
        Initialize model state from a subject's baseline EEG band powers.

        Maps baseline spectral characteristics to initial conditions for
        all N cortical populations and the shared thalamic circuit.

        Stage-aware initialization (v2)
        --------------------------------
        When *sleep_stage* is provided (e.g. 'N2' or 'N3'), the model sets
        I_sleep directly from the SLEEP_STAGE_WEIGHTS for that stage rather
        than averaging over the whole-night stage distribution.  This gives
        each subject a starting I_sleep that matches the depth at which
        they enter consolidated NREM sleep — the same operating point that
        a real CLAS experimenter would target.

        When *process_s_init* is provided it overrides both the legacy
        S_init parameter and the delta-based heuristic.  It is computed
        from the number of epochs between recording start and stable
        N2/N3 onset: S = 0.90 * exp(-onset_time / tau_decay).

        *night_fraction* (0-1) indicates how far into the recording the
        NREM onset occurs.  This is used to set an initial ultradian
        phase offset so subjects starting in later cycles begin at a
        corresponding phase of the ~90 min NREM-REM cycle.

        Parameters
        ----------
        band_powers : dict
            Baseline EEG band powers (delta_power, theta_power, etc.).
        non_responder_fraction : float
            Fraction of cortical populations that do not respond to forcing.
        sleep_stage_fractions : dict, optional
            Sleep stage distribution for I_sleep computation.
        S_init : float, optional
            Per-subject Process S at CLAS onset (legacy parameter).
        baseline_delta_at_onset : float, optional
            Delta power at N2 onset for ceiling effect modelling.
        sleep_stage : str, optional
            The sleep stage at the stable NREM onset epoch ('N2', 'N3').
            When set, overrides sleep_stage_fractions for I_sleep.
        process_s_init : float, optional
            Per-subject Process S computed from NREM onset latency.
            Takes precedence over S_init and the delta heuristic.
        night_fraction : float
            Fraction through the recording at NREM onset (0-1).
            Used to offset the ultradian cycle phase.
        """
        delta = band_powers.get('delta_power', 0.25)
        theta = band_powers.get('theta_power', 0.25)
        alpha = band_powers.get('alpha_power', 0.25)
        beta = band_powers.get('beta_power', 0.25)

        # Initial cortical state: higher delta -> more time in UP state
        E_init = 0.05 + 0.15 * delta
        I_init = 0.02 + 0.08 * delta
        self.E_pop = np.full(self.N_pop, E_init)
        self.I_pop = np.full(self.N_pop, I_init)
        self.A_pop = np.zeros(self.N_pop)
        self.E = float(np.mean(self.E_pop))
        self.I = float(np.mean(self.I_pop))
        self.A_adapt = 0.0

        # Heterogeneous I_tonic for ensemble
        I_tonic_base = self.p['I_tonic']
        spread = self.p.get('I_tonic_spread', 0.12)
        self.I_tonic_pop = I_tonic_base * (
            1.0 + spread * np.linspace(-1, 1, self.N_pop)
        )

        # Heterogeneous tau_adapt across populations (breaks sharp
        # Arnold-tongue lock-in — Kuramoto desynchronization).
        # True Gaussian draw, not linspace+shuffle.
        tau_adapt_base = self.p['tau_adapt']
        tau_spread = self.p.get('tau_adapt_spread', 0.20)
        tau_offsets = self.rng.standard_normal(self.N_pop)
        self.tau_adapt_pop = np.clip(
            tau_adapt_base * (1.0 + tau_spread * tau_offsets),
            tau_adapt_base * 0.3,
            tau_adapt_base * 2.5,
        )

        # Forcing gain distribution: Beta-distributed continuous gains.
        # Replaces binary mask with graded responsiveness reflecting
        # heterogeneous thalamocortical innervation (Jones 2001).
        # Non-responders are set to zero; responders have graded gains.
        n_non = max(0, int(self.N_pop * non_responder_fraction))
        ga = self.p.get('gain_alpha', 2.0)
        gb = self.p.get('gain_beta', 3.0)
        self.forcing_gain_pop = self.rng.beta(ga, gb, size=self.N_pop)
        if n_non > 0:
            self.forcing_gain_pop[-n_non:] = 0.0
        # Legacy binary mask (backward compat)
        self.forcing_mask_pop = (self.forcing_gain_pop > 0).astype(float)

        # ── Ceiling effect on forcing gain (Papalambros et al. 2017) ──
        # Subjects with high baseline delta at N2 onset are already near
        # maximal SWA. Their percentage enhancement is attenuated because
        # there is less room to grow. We scale the per-population forcing
        # gains by (1 - baseline_delta_at_onset), so a subject with
        # delta=0.8 at onset retains only 20% of forcing effectiveness,
        # while delta=0.3 retains 70%.
        if baseline_delta_at_onset is not None:
            ceiling_factor = max(0.1, 1.0 - baseline_delta_at_onset)
            self.forcing_gain_pop *= ceiling_factor
            # Update legacy binary mask to reflect attenuated gains
            self.forcing_mask_pop = (self.forcing_gain_pop > 0).astype(float)

        # Per-population click jitter (fixed per subject, structural)
        jitter_sigma = self.p.get('click_jitter_sec', 0.050)
        self.click_jitter_pop = (
            jitter_sigma * self.rng.standard_normal(self.N_pop)
        )

        # Initial thalamic state
        self.TC = 0.05 + 0.10 * alpha
        self.TRN = 0.03 + 0.05 * alpha

        # ── Process S: per-subject homeostatic sleep pressure ──
        # Priority: process_s_init > S_init > delta heuristic.
        #
        # process_s_init is computed from the actual NREM onset latency
        # in the subject's recording: S = 0.90 * exp(-onset_time / tau).
        # Subjects who reach N2/N3 quickly retain high S (strong
        # homeostatic drive); those with delayed onset have already
        # partially dissipated S during light NREM.
        #
        # S_init is a legacy parameter (kept for backward compatibility).
        # The delta heuristic (0.4 + 0.4*delta) is the final fallback.
        if process_s_init is not None:
            self.S = float(np.clip(process_s_init, 0.0, 1.0))
        elif S_init is not None:
            self.S = float(np.clip(S_init, 0.0, 1.0))
        else:
            self.S = 0.4 + 0.4 * delta
        # Store the initialized S value so reset methods can restore it
        # instead of falling back to the generic 0.6 default.
        self._S_init = self.S
        self._is_sleeping = True

        # TC coupling scaling
        delta_scale = delta / 0.25
        self.alpha_TC = self._alpha_TC_base * delta_scale

        # Sleep input current: stage-aware override vs distribution average.
        #
        # When a specific sleep_stage is given (from the stable NREM onset
        # finder), I_sleep is set directly from that stage's weight.
        # This is more accurate than averaging over the whole-night
        # distribution because it reflects the *actual* cortical state
        # at the moment stimulation would begin in a real CLAS experiment.
        self._sleep_stage_fractions = sleep_stage_fractions
        self._onset_sleep_stage = sleep_stage  # store for diagnostics
        if sleep_stage and sleep_stage in SLEEP_STAGE_WEIGHTS:
            self.I_sleep = SLEEP_STAGE_WEIGHTS[sleep_stage]
        elif sleep_stage_fractions is not None:
            self.I_sleep = 0.0
            for stage, frac in sleep_stage_fractions.items():
                weight = SLEEP_STAGE_WEIGHTS.get(stage, 0.0)
                self.I_sleep += weight * frac
        else:
            self.I_sleep = 0.5

        # ── Ultradian cycle phase offset ──
        # Subjects who reach stable NREM at different points in the
        # recording are at different phases of the ~90 min ultradian
        # cycle.  We encode this as an initial _session_time offset
        # so that the NREM gate (if reactivated for cycle modeling)
        # and the SSA/habituation dynamics start at the correct phase.
        if night_fraction > 0:
            ultradian_period_sec = ULTRADIAN_CYCLE_MIN * 60.0
            # Map night_fraction to a phase within the first ultradian cycle.
            # Most subjects reach N2/N3 within the first cycle (0-90 min).
            self._ultradian_phase_offset = (
                night_fraction * ultradian_period_sec
            )
        else:
            self._ultradian_phase_offset = 0.0

        # Legacy-compatible arrays for pipeline
        center_freq = _dominant_frequency(band_powers)
        spread_freq = max(0.5, np.sqrt(sum(
            band_powers.get(f'{b}_power', 0.25) *
            ((lo + hi) / 2 - center_freq) ** 2
            for b, (lo, hi) in BAND_EDGES.items()
        )))

        n_delta_osc = int(0.15 * self.N)
        n_main = self.N - n_delta_osc
        main_freqs = center_freq + spread_freq * self.rng.standard_normal(n_main)
        main_freqs = np.clip(main_freqs, 1.0, 30.0)
        delta_freqs = 1.0 + 2.0 * self.rng.random(n_delta_osc)
        self.natural_freqs = np.concatenate([main_freqs, delta_freqs])
        self.rng.shuffle(self.natural_freqs)
        self.natural_freqs *= 2.0 * np.pi

        # Base excitability
        self.lambda_0 = np.full(self.N, self.lambda_base)
        self.lambda_0 += 0.15 * self.rng.standard_normal(self.N)
        freq_hz = self.natural_freqs / (2.0 * np.pi)
        alpha_mask = (freq_hz >= 7.0) & (freq_hz <= 14.0)
        alpha_scale = alpha / 0.25
        self.lambda_0[alpha_mask] += 0.3 * alpha_scale
        delta_mask = freq_hz < 4.0
        self.lambda_0[delta_mask] = -0.05

        # Non-responder mask (legacy)
        n_non = int(self.N * non_responder_fraction)
        self.forcing_mask = np.ones(self.N)
        if n_non > 0:
            idx = self.rng.choice(self.N, size=n_non, replace=False)
            self.forcing_mask[idx] = 0.0

        # Legacy complex state (z)
        self.z = (
            0.1 * self.rng.standard_normal(self.N)
            + 0.1j * self.rng.standard_normal(self.N)
        )

        # Reset dynamics
        self.h_T = 0.5
        self.T = 0.0
        self.H = 0.0
        self.A_fast = 0.0
        self.A_slow = 0.0
        self._last_forcing_freq = -1.0
        self._baseline_swa = None
        self.so_phase = 0.0
        self.t = 0.0
        self._session_time = 0.0
        self._kc_habituation = 0.0
        self._kc_active = False
        self._kc_remaining_sec = 0.0
        self._last_pulse_time = -1.0
        self._pulse_envelope = 0.0
        self._pulse_onset_amplitude = 0.0
        self._last_click_time = -1.0
        self._click_envelope = 0.0
        self._click_onset_amplitude = 0.0
        self._prev_forcing_phase = 0.0
        self._mf_start_time = 0.0
        self._mf_ema = 0.5
        self._mf_emvar = 0.01
        self._R_smooth = 0.08
        self._pulse_phase_log = []
        self._pulse_count = 0
        self._kc_event_times = []
        self._pulse_event_times = []
        self._refractory_gain = 0.0
        self.biological_ssa.reset()

        # Reset buffers
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0
        self._so_buffer[:] = 0.0
        self._so_buf_idx = 0
        self._so_buf_filled = False
        self._so_sample_counter = 0

    def initialize_from_subject(self, subject_data: Dict[str, float]) -> None:
        """Alias for initialize_from_baseline for the new API."""
        self.initialize_from_baseline(subject_data)

    # ─── State management ─────────────────────────────────────────

    def get_state(self) -> Dict:
        """Return complete model state for checkpointing."""
        return {
            'E': self.E,
            'I': self.I,
            'A_adapt': self.A_adapt,
            # Ensemble arrays
            'E_pop': self.E_pop.copy(),
            'I_pop': self.I_pop.copy(),
            'A_pop': self.A_pop.copy(),
            'I_tonic_pop': self.I_tonic_pop.copy(),
            'forcing_mask_pop': self.forcing_mask_pop.copy(),
            'forcing_gain_pop': self.forcing_gain_pop.copy(),
            'click_jitter_pop': self.click_jitter_pop.copy(),
            # Thalamus
            'TC': self.TC,
            'TRN': self.TRN,
            'h_T': self.h_T,
            'S': self.S,
            '_S_init': self._S_init,
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
            '_session_time': self._session_time,
            '_kc_habituation': self._kc_habituation,
            '_last_pulse_time': self._last_pulse_time,
            '_pulse_envelope': self._pulse_envelope,
            '_pulse_onset_amplitude': self._pulse_onset_amplitude,
            '_last_click_time': self._last_click_time,
            '_click_envelope': self._click_envelope,
            '_click_onset_amplitude': self._click_onset_amplitude,
            '_prev_forcing_phase': self._prev_forcing_phase,
            '_mf_start_time': self._mf_start_time,
            '_mf_ema': self._mf_ema,
            '_mf_emvar': self._mf_emvar,
            '_R_smooth': self._R_smooth,
            '_biological_ssa_state': self.biological_ssa.get_state(),
            # Legacy arrays
            'z': self.z.copy(),
            'natural_freqs': self.natural_freqs.copy(),
            'lambda_0': self.lambda_0.copy(),
            'forcing_mask': self.forcing_mask.copy(),
            # Buffers
            '_mf_buffer': self._mf_buffer.copy(),
            '_mf_idx': self._mf_idx,
            '_step_counter': self._step_counter,
            '_so_buffer': self._so_buffer.copy(),
            '_so_buf_idx': self._so_buf_idx,
            '_so_buf_filled': self._so_buf_filled,
            '_so_sample_counter': self._so_sample_counter,
            '_rng_state': self.rng.bit_generator.state,
            '_aux_rng_state': self._aux_rng.bit_generator.state,
        }

    def set_state(self, state: Dict) -> None:
        """Restore model state from a saved snapshot."""
        self.E = state['E']
        self.I = state['I']
        self.A_adapt = state.get('A_adapt', 0.0)
        # Ensemble arrays
        if 'E_pop' in state:
            self.E_pop = state['E_pop'].copy()
            self.I_pop = state['I_pop'].copy()
            self.A_pop = state['A_pop'].copy()
        else:
            # Backward compat: old state without ensemble
            self.E_pop = np.full(self.N_pop, self.E)
            self.I_pop = np.full(self.N_pop, self.I)
            self.A_pop = np.full(self.N_pop, self.A_adapt)
        if 'I_tonic_pop' in state:
            self.I_tonic_pop = state['I_tonic_pop'].copy()
        if 'forcing_mask_pop' in state:
            self.forcing_mask_pop = state['forcing_mask_pop'].copy()
        if 'forcing_gain_pop' in state:
            self.forcing_gain_pop = state['forcing_gain_pop'].copy()
        if 'click_jitter_pop' in state:
            self.click_jitter_pop = state['click_jitter_pop'].copy()
        # Thalamus
        self.TC = state['TC']
        self.TRN = state['TRN']
        self.h_T = state.get('h_T', 0.5)
        self.S = state.get('S', 0.6)
        self._S_init = state.get('_S_init', self.S)
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
        self._session_time = state.get('_session_time', 0.0)
        self._kc_habituation = state.get('_kc_habituation', 0.0)
        self._last_pulse_time = state.get('_last_pulse_time', -1.0)
        self._pulse_envelope = state.get('_pulse_envelope', 0.0)
        self._pulse_onset_amplitude = state.get('_pulse_onset_amplitude', 0.0)
        self._last_click_time = state.get('_last_click_time', -1.0)
        self._click_envelope = state.get('_click_envelope', 0.0)
        self._click_onset_amplitude = state.get('_click_onset_amplitude', 0.0)
        self._prev_forcing_phase = state.get('_prev_forcing_phase', 0.0)
        self._mf_ema = state.get('_mf_ema', 0.5)
        self._mf_emvar = state.get('_mf_emvar', 0.01)
        self._R_smooth = state.get('_R_smooth', 0.08)
        self._mf_start_time = state.get('_mf_start_time', 0.0)
        # Restore BiologicalSSA state if available
        if '_biological_ssa_state' in state:
            bio_st = state['_biological_ssa_state']
            if 'x_channels' in bio_st:
                self.biological_ssa.x = bio_st['x_channels'].copy()
            self.biological_ssa.H = bio_st.get('H', 0.0)
            self.biological_ssa.Ca = bio_st.get('Ca', 0.0)
            self.biological_ssa.KC_hab = bio_st.get('KC_hab', 0.0)
            self.biological_ssa._last_f_stim = bio_st.get('last_f_stim', -1.0)
        # Legacy arrays
        self.z = state['z'].copy()
        self.natural_freqs = state['natural_freqs'].copy()
        self.lambda_0 = state['lambda_0'].copy()
        self.forcing_mask = state['forcing_mask'].copy()
        # Buffers
        self._mf_buffer = state['_mf_buffer'].copy()
        self._mf_idx = state['_mf_idx']
        self._step_counter = state['_step_counter']
        if '_so_buffer' in state:
            self._so_buffer = state['_so_buffer'].copy()
            self._so_buf_idx = state['_so_buf_idx']
            self._so_buf_filled = state['_so_buf_filled']
            self._so_sample_counter = state['_so_sample_counter']
        self.rng.bit_generator.state = state['_rng_state']
        if '_aux_rng_state' in state:
            self._aux_rng.bit_generator.state = state['_aux_rng_state']

    def _reset_for_scan(self) -> None:
        """Reset state and buffers for a new frequency scan trial."""
        # Ensemble arrays
        self.E_pop = np.full(self.N_pop, 0.1)
        self.I_pop = np.full(self.N_pop, 0.05)
        self.A_pop = np.zeros(self.N_pop)
        # Scalar backward-compat
        self.E = 0.1
        self.I = 0.05
        self.A_adapt = 0.0
        # Thalamus
        self.TC = 0.1
        self.TRN = 0.05
        self.h_T = 0.5
        # Restore per-subject Process S (set by initialize_from_baseline)
        # instead of generic 0.6, preserving individual differences.
        self.S = self._S_init
        self.T = 0.0
        self.H = 0.0
        self.A_fast = 0.0
        self.A_slow = 0.0
        self._last_forcing_freq = -1.0
        self._baseline_swa = None
        self.so_phase = 0.0
        self.t = 0.0
        self._session_time = 0.0
        self._kc_habituation = 0.0
        self._kc_active = False
        self._kc_remaining_sec = 0.0
        self._last_pulse_time = -1.0
        self._pulse_envelope = 0.0
        self._pulse_onset_amplitude = 0.0
        self._last_click_time = -1.0
        self._click_envelope = 0.0
        self._click_onset_amplitude = 0.0
        self._prev_forcing_phase = 0.0
        self._mf_start_time = 0.0
        self._mf_ema = 0.5
        self._mf_emvar = 0.01
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0
        self._so_buffer[:] = 0.0
        self._so_buf_idx = 0
        self._so_buf_filled = False
        self._so_sample_counter = 0
        self.biological_ssa.reset()
        # NOTE: Do NOT re-randomize forcing_mask_pop here.
        # The forcing mask is a structural property of the subject (which
        # cortical columns respond to stimulation). Re-randomizing it on
        # every frequency trial introduces spurious within-subject variance
        # that inflates individual-differences SD. The mask is set once in
        # initialize_from_baseline() and preserved across scan trials.
        # Reset legacy z
        self.z = (
            0.1 * self.rng.standard_normal(self.N)
            + 0.1j * self.rng.standard_normal(self.N)
        )

    def _reset_dynamic_state(self) -> None:
        """Reset dynamic integration state (buffers, timers) without
        re-randomizing structural arrays (I_tonic_pop, forcing_mask_pop).

        Used by frequency_scan to ensure each trial starts from the same
        structural configuration while clearing transient dynamics.
        """
        # Ensemble arrays — reset to default initial conditions
        self.E_pop = np.full(self.N_pop, 0.1)
        self.I_pop = np.full(self.N_pop, 0.05)
        self.A_pop = np.zeros(self.N_pop)
        # Scalar backward-compat
        self.E = 0.1
        self.I = 0.05
        self.A_adapt = 0.0
        # Thalamus
        self.TC = 0.1
        self.TRN = 0.05
        self.h_T = 0.5
        # Restore per-subject Process S (set by initialize_from_baseline)
        self.S = self._S_init
        self.T = 0.0
        self.H = 0.0
        self.A_fast = 0.0
        self.A_slow = 0.0
        self._last_forcing_freq = -1.0
        self._baseline_swa = None
        self.so_phase = 0.0
        self.t = 0.0
        self._session_time = 0.0
        self._kc_active = False
        self._kc_remaining_sec = 0.0
        self._last_pulse_time = -1.0
        self._pulse_envelope = 0.0
        self._pulse_onset_amplitude = 0.0
        self._last_click_time = -1.0
        self._click_envelope = 0.0
        self._click_onset_amplitude = 0.0
        self._prev_forcing_phase = 0.0
        self._mf_start_time = 0.0
        self._mf_ema = 0.5
        self._mf_emvar = 0.01
        # Buffers
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0
        self._so_buffer[:] = 0.0
        self._so_buf_idx = 0
        self._so_buf_filled = False
        self._so_sample_counter = 0
        self.biological_ssa.reset()

    def reset(self) -> None:
        """Reset to initial conditions for next session."""
        self._reset_for_scan()

    # ─── NREM-REM gate ────────────────────────────────────────────

    def _nrem_gate(self) -> float:
        """
        NREM gate for the stimulation-active portion of a NREM period.

        Design rationale (Ngo 2013, Besedovsky 2017):
        -----------------------------------------------
        In real CLAS experiments the experimenter waits for stable N2/N3
        (typically 15-30 min after lights-off) and then delivers auditory
        stimulation throughout the remainder of the first NREM period
        (60-90 min). Stimulation is only briefly paused during micro-
        arousals or stage transitions (~10-15% of epochs).

        The 60-minute simulation therefore represents the *stimulation-
        active* window of a single NREM period.  The gate should be
        near 1.0 for the entire session.  Homeostatic decline is already
        modelled by Process S (tau = 4.2 h, ~21% decay over 60 min),
        so the NREM gate must NOT impose a second, redundant decline.

        Implementation:
        - Constant gate = 1.0 for the full session.
        - Process S handles the gentle homeostatic sleep-pressure decay.
        - Micro-arousal pauses are handled by BiologicalSSA's arousal
          logic, not by this gate.
        """
        return 1.0

    # ─── Process S dynamics ───────────────────────────────────────

    def _update_process_s(self, dt_sec: float) -> None:
        """Update Process S (homeostatic sleep pressure)."""
        if self._is_sleeping:
            tau_s = PROCESS_S_TAU_DECAY_HR * 3600.0
            s_inf = 0.0
        else:
            tau_s = PROCESS_S_TAU_RISE_HR * 3600.0
            s_inf = 1.0
        self.S += -(self.S - s_inf) / tau_s * dt_sec

    # ─── SO phase extraction ─────────────────────────────────────

    def _update_emergent_so_phase(self, mf_real: float) -> None:
        """Update emergent SO phase from low-pass filtered mean-field signal."""
        self._so_buffer[self._so_buf_idx % self._so_buffer_size] = mf_real
        self._so_buf_idx += 1
        if self._so_buf_idx >= self._so_buffer_size:
            self._so_buf_filled = True

        self._so_sample_counter += 1

        if self._so_sample_counter >= self._so_update_interval:
            self._so_sample_counter = 0

            n_valid = self._so_buffer_size if self._so_buf_filled else self._so_buf_idx
            if n_valid < 128 or self._so_sos is None:
                return

            if self._so_buf_filled:
                start = self._so_buf_idx % self._so_buffer_size
                buf = np.roll(self._so_buffer, -start)
            else:
                buf = self._so_buffer[:n_valid]

            try:
                filtered = sp_signal.sosfiltfilt(self._so_sos, buf)
            except ValueError:
                return

            analytic = sp_signal.hilbert(filtered)
            self.so_phase = float(np.angle(analytic[-1])) % (2.0 * np.pi)

    # ─── K-complex generation ─────────────────────────────────────

    def _maybe_trigger_kcomplex(
        self, forcing_strength: float, nrem_gate: float, dt_sec: float,
        sleep_stage: str = 'N2',
    ) -> float:
        """Probabilistically trigger a K-complex evoked response."""
        kc_boost = 0.0

        if self._kc_active:
            self._kc_remaining_sec -= dt_sec
            if self._kc_remaining_sec <= 0:
                self._kc_active = False
                self._kc_remaining_sec = 0.0
            else:
                progress = 1.0 - self._kc_remaining_sec / KC_DURATION_SEC
                kc_boost = KC_DELTA_BOOST * np.exp(-3.0 * progress)

        if (not self._kc_active and forcing_strength > 0.01
                and nrem_gate > 0.3):
            p_kc = self.biological_ssa.get_kc_probability(sleep_stage) * nrem_gate
            if self.rng.random() < p_kc * dt_sec * 10.0:
                self._kc_active = True
                self._kc_remaining_sec = KC_DURATION_SEC
                kc_boost = KC_DELTA_BOOST
                self.biological_ssa.update_kc_habituation(True, dt_sec)

        return kc_boost

    # ─── Core ensemble derivatives (vectorized) ───────────────────

    def _derivatives_ensemble(
        self,
        E: np.ndarray, I: np.ndarray, A_ad: np.ndarray,
        TC: float, TRN: float, h_T: float,
        I_ext_ctx: np.ndarray, I_sleep_ctx: float,
        I_ext_thal: float, I_sleep_thal: float,
        kc_boost: float,
        I_tonic_eff_pop: np.ndarray,
        adapt_modulation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """
        Vectorized derivatives for N cortical populations + shared thalamus.

        E, I, A_ad: arrays of shape (N_pop,)
        TC, TRN, h_T: scalars (shared thalamus)
        I_ext_ctx: array of shape (N_pop,) — per-population forcing
        I_tonic_eff_pop: array of shape (N_pop,) — heterogeneous tonic drive
        Returns: (dE[N], dI[N], dA[N], dTC, dTRN, dh_T)
        """
        p = self.p
        E_mean = np.mean(E)

        # Thalamocortical input (same for all populations)
        I_thal = p['w_TC_ctx'] * TC

        # Mean-field coupling (Kuramoto-like)
        K_ens = p.get('K_ensemble', 0.0)
        coupling = K_ens * (E_mean - E)  # shape (N_pop,)

        # Cortical E (vectorized across populations)
        g_eff = p['g_adapt'] * (1.0 - adapt_modulation)
        input_E = (p['w_EE'] * E - p['w_EI'] * I - g_eff * A_ad
                   + I_tonic_eff_pop + I_thal + I_ext_ctx + I_sleep_ctx
                   + kc_boost + coupling)
        dE = (-E + _wc_sigmoid_vec(input_E, p['a_ctx'], p['theta_ctx'])) / p['tau_E']

        # Cortical I (vectorized)
        input_I = p['w_IE'] * E - p['w_II'] * I
        dI = (-I + _wc_sigmoid_vec(input_I, p['a_ctx'], p['theta_ctx'])) / p['tau_I']

        # Spike-frequency adaptation — per-population time constant
        # breaks sharp 1:1 Arnold-tongue lock-in (Kuramoto theory).
        dA = (E - A_ad) / self.tau_adapt_pop

        # T-current de-inactivation (shared thalamus, scalar)
        h_inf = 1.0 / (1.0 + np.exp(5.0 * (TC - p['theta_h'])))
        dh_T = (h_inf - h_T) / p['tau_h']
        I_T = p['g_T'] * h_T * TC

        # TC relay driven by E_mean (convergent corticothalamic projection)
        input_TC = (p['w_CT'] * E_mean - p['w_RT'] * TRN
                    + I_T + I_sleep_thal + I_ext_thal)
        dTC = (-TC + _sigmoid(input_TC, p['a_thal'], p['theta_thal'])) / p['tau_TC']

        # Reticular nucleus driven by E_mean
        input_TRN = (p['w_CT_R'] * E_mean + p['w_TC_R'] * TC
                     - p['w_RR'] * TRN)
        dTRN = (-TRN + _sigmoid(input_TRN, p['a_thal'], p['theta_thal'])) / p['tau_TRN']

        return dE, dI, dA, dTC, dTRN, dh_T

    def _rk4_step_ensemble(
        self,
        E: np.ndarray, I: np.ndarray, A_ad: np.ndarray,
        TC: float, TRN: float, h_T: float,
        dt: float,
        I_ext_ctx: np.ndarray, I_sleep_ctx: float,
        I_ext_thal: float, I_sleep_thal: float,
        kc_boost: float,
        I_tonic_eff_pop: np.ndarray,
        adapt_modulation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """
        Single RK4 integration step for the ensemble system.

        E, I, A_ad: arrays of shape (N_pop,)
        TC, TRN, h_T: scalars
        Returns updated (E, I, A_ad, TC, TRN, h_T).
        """
        args = (I_ext_ctx, I_sleep_ctx, I_ext_thal, I_sleep_thal,
                kc_boost, I_tonic_eff_pop, adapt_modulation)

        # k1
        k1_E, k1_I, k1_A, k1_TC, k1_TRN, k1_h = self._derivatives_ensemble(
            E, I, A_ad, TC, TRN, h_T, *args)

        # k2
        k2_E, k2_I, k2_A, k2_TC, k2_TRN, k2_h = self._derivatives_ensemble(
            E + 0.5 * dt * k1_E,
            I + 0.5 * dt * k1_I,
            A_ad + 0.5 * dt * k1_A,
            TC + 0.5 * dt * k1_TC,
            TRN + 0.5 * dt * k1_TRN,
            h_T + 0.5 * dt * k1_h,
            *args)

        # k3
        k3_E, k3_I, k3_A, k3_TC, k3_TRN, k3_h = self._derivatives_ensemble(
            E + 0.5 * dt * k2_E,
            I + 0.5 * dt * k2_I,
            A_ad + 0.5 * dt * k2_A,
            TC + 0.5 * dt * k2_TC,
            TRN + 0.5 * dt * k2_TRN,
            h_T + 0.5 * dt * k2_h,
            *args)

        # k4
        k4_E, k4_I, k4_A, k4_TC, k4_TRN, k4_h = self._derivatives_ensemble(
            E + dt * k3_E,
            I + dt * k3_I,
            A_ad + dt * k3_A,
            TC + dt * k3_TC,
            TRN + dt * k3_TRN,
            h_T + dt * k3_h,
            *args)

        # Combine
        E_new = E + (dt / 6.0) * (k1_E + 2.0 * k2_E + 2.0 * k3_E + k4_E)
        I_new = I + (dt / 6.0) * (k1_I + 2.0 * k2_I + 2.0 * k3_I + k4_I)
        A_new = A_ad + (dt / 6.0) * (k1_A + 2.0 * k2_A + 2.0 * k3_A + k4_A)
        TC_new = TC + (dt / 6.0) * (k1_TC + 2.0 * k2_TC + 2.0 * k3_TC + k4_TC)
        TRN_new = TRN + (dt / 6.0) * (k1_TRN + 2.0 * k2_TRN + 2.0 * k3_TRN + k4_TRN)
        h_T_new = h_T + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)

        # Soft clip: only prevent gross blow-up, not natural variability.
        #  widened from [0,1] to [-0.05, 1.05] so the
        # firing-rate variables retain a small differentiable margin
        # near the saturation boundaries (the previous tight clip
        # destroyed the smoothness on which any phase-response
        # analysis depends).
        np.clip(E_new, -0.05, 1.05, out=E_new)
        np.clip(I_new, -0.05, 1.05, out=I_new)
        np.clip(A_new, 0.0, 2.0, out=A_new)
        TC_new = float(np.clip(TC_new, 0.0, 1.0))
        TRN_new = float(np.clip(TRN_new, 0.0, 1.0))
        h_T_new = float(np.clip(h_T_new, 0.0, 1.0))

        return E_new, I_new, A_new, TC_new, TRN_new, h_T_new

    # ─── Legacy scalar derivatives (kept for backward compat) ─────

    def _derivatives(
        self,
        E: float, I: float, A_ad: float, TC: float, TRN: float,
        h_T: float,
        I_ext_ctx: float, I_sleep_ctx: float,
        I_ext_thal: float, I_sleep_thal: float,
        kc_boost: float,
        adapt_modulation: float = 0.0,
    ) -> Tuple[float, float, float, float, float, float]:
        """Legacy scalar derivatives (single population). Kept for compat."""
        p = self.p
        I_thal = p['w_TC_ctx'] * TC
        g_adapt_eff = p['g_adapt'] * (1.0 - adapt_modulation)
        input_E = (p['w_EE'] * E - p['w_EI'] * I
                   - g_adapt_eff * A_ad
                   + p['I_tonic']
                   + I_thal + I_ext_ctx + I_sleep_ctx + kc_boost)
        dE = (-E + _wc_sigmoid(input_E, p['a_ctx'], p['theta_ctx'])) / p['tau_E']
        input_I = p['w_IE'] * E - p['w_II'] * I
        dI = (-I + _wc_sigmoid(input_I, p['a_ctx'], p['theta_ctx'])) / p['tau_I']
        dA_ad = (E - A_ad) / p['tau_adapt']
        h_inf = 1.0 / (1.0 + np.exp(5.0 * (TC - p['theta_h'])))
        dh_T = (h_inf - h_T) / p['tau_h']
        I_T = p['g_T'] * h_T * TC
        input_TC = (p['w_CT'] * E - p['w_RT'] * TRN
                    + I_T + I_sleep_thal + I_ext_thal)
        dTC = (-TC + _sigmoid(input_TC, p['a_thal'], p['theta_thal'])) / p['tau_TC']
        input_TRN = (p['w_CT_R'] * E + p['w_TC_R'] * TC
                     - p['w_RR'] * TRN)
        dTRN = (-TRN + _sigmoid(input_TRN, p['a_thal'], p['theta_thal'])) / p['tau_TRN']
        return dE, dI, dA_ad, dTC, dTRN, dh_T

    def _rk4_step(
        self,
        E: float, I: float, A_ad: float, TC: float, TRN: float,
        h_T: float,
        dt: float,
        I_ext_ctx: float, I_sleep_ctx: float,
        I_ext_thal: float, I_sleep_thal: float,
        kc_boost: float,
        adapt_modulation: float = 0.0,
    ) -> Tuple[float, float, float, float, float, float]:
        """Legacy scalar RK4 step. Kept for backward compat."""
        k1_E, k1_I, k1_A, k1_TC, k1_TRN, k1_h = self._derivatives(
            E, I, A_ad, TC, TRN, h_T,
            I_ext_ctx, I_sleep_ctx, I_ext_thal, I_sleep_thal, kc_boost,
            adapt_modulation)
        k2_E, k2_I, k2_A, k2_TC, k2_TRN, k2_h = self._derivatives(
            E + 0.5 * dt * k1_E, I + 0.5 * dt * k1_I,
            A_ad + 0.5 * dt * k1_A, TC + 0.5 * dt * k1_TC,
            TRN + 0.5 * dt * k1_TRN, h_T + 0.5 * dt * k1_h,
            I_ext_ctx, I_sleep_ctx, I_ext_thal, I_sleep_thal, kc_boost,
            adapt_modulation)
        k3_E, k3_I, k3_A, k3_TC, k3_TRN, k3_h = self._derivatives(
            E + 0.5 * dt * k2_E, I + 0.5 * dt * k2_I,
            A_ad + 0.5 * dt * k2_A, TC + 0.5 * dt * k2_TC,
            TRN + 0.5 * dt * k2_TRN, h_T + 0.5 * dt * k2_h,
            I_ext_ctx, I_sleep_ctx, I_ext_thal, I_sleep_thal, kc_boost,
            adapt_modulation)
        k4_E, k4_I, k4_A, k4_TC, k4_TRN, k4_h = self._derivatives(
            E + dt * k3_E, I + dt * k3_I,
            A_ad + dt * k3_A, TC + dt * k3_TC,
            TRN + dt * k3_TRN, h_T + dt * k3_h,
            I_ext_ctx, I_sleep_ctx, I_ext_thal, I_sleep_thal, kc_boost,
            adapt_modulation)
        E_new = E + (dt / 6.0) * (k1_E + 2.0 * k2_E + 2.0 * k3_E + k4_E)
        I_new = I + (dt / 6.0) * (k1_I + 2.0 * k2_I + 2.0 * k3_I + k4_I)
        A_new = A_ad + (dt / 6.0) * (k1_A + 2.0 * k2_A + 2.0 * k3_A + k4_A)
        TC_new = TC + (dt / 6.0) * (k1_TC + 2.0 * k2_TC + 2.0 * k3_TC + k4_TC)
        TRN_new = TRN + (dt / 6.0) * (k1_TRN + 2.0 * k2_TRN + 2.0 * k3_TRN + k4_TRN)
        h_T_new = h_T + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
        E_new = float(np.clip(E_new, 0.0, 1.0))
        I_new = float(np.clip(I_new, 0.0, 1.0))
        A_new = float(np.clip(A_new, 0.0, 2.0))
        TC_new = float(np.clip(TC_new, 0.0, 1.0))
        TRN_new = float(np.clip(TRN_new, 0.0, 1.0))
        h_T_new = float(np.clip(h_T_new, 0.0, 1.0))
        return E_new, I_new, A_new, TC_new, TRN_new, h_T_new

    # ─── Vectorized epoch simulation ──────────────────────────────

    def run_epoch(
        self,
        duration_sec: float,
        external_freq_hz: float,
        forcing_strength: float,
        sleep_stage: str = 'N2',
        pulsed: bool = False,
        pulse_params: Optional[Dict] = None,
    ) -> None:
        """
        Simulate one epoch of the TCWC ensemble model.

        Integrates N cortical Wilson-Cowan populations + shared thalamus
        using RK4 with additive Gaussian noise. The mean-field signal
        mean(E_pop) is sampled into a buffer for PSD / band power / PLV.
        """
        dt = self.dt
        n_steps = int(duration_sec / dt)
        sigma = self.p['sigma']
        sqrt_dt = np.sqrt(dt)
        N = self.N_pop

        omega_ext = 2.0 * np.pi * external_freq_hz

        # Auditory gain based on sleep stage
        aud_gain = AUDITORY_GAIN.get(sleep_stage, 0.4)

        # NREM gate
        nrem = self._nrem_gate()

        # Track last forcing frequency
        if forcing_strength > 0:
            self._last_forcing_freq = external_freq_hz

        # Phase efficacy gain for pulsed delivery
        phase_efficacy_gain = self.p.get('phase_efficacy_gain', 3.5)

        # Pulse parameters
        phase_window = self.p['pulse_phase_window']
        refractory = self.p['pulse_refractory_sec']
        tau_response = self.p['pulse_tau_response']
        if pulse_params is not None:
            phase_window = pulse_params.get('phase_window', phase_window)
            refractory = pulse_params.get('refractory_sec', refractory)
            tau_response = pulse_params.get('tau_response', tau_response)

        # Process S modulates I_tonic
        I_tonic_eff_base = self.p['I_tonic'] * (1.0 + 1.0 * self.S)
        spread = self.p.get('I_tonic_spread', 0.12)
        I_tonic_eff_pop = I_tonic_eff_base * (
            1.0 + spread * np.linspace(-1, 1, N)
        )

        # Sleep-dependent inputs
        I_sleep_ctx = 0.3 * self.I_sleep * self.S * nrem
        I_sleep_thal = 0.2 * self.I_sleep * self.S * nrem

        # Pre-generate ALL noise to ensure WC and z-array use deterministic
        # noise sequences regardless of forcing condition. This prevents
        # the shared RNG from creating spurious differences between
        # baseline and stim conditions (the z-array consumes different
        # amounts of randomness with/without forcing, which would shift
        # the WC noise sequence and create fake enhancement).
        noise_E = sigma * sqrt_dt * self.rng.standard_normal((n_steps, N))
        noise_I = sigma * sqrt_dt * self.rng.standard_normal((n_steps, N))
        noise_TC = sigma * sqrt_dt * self.rng.standard_normal(n_steps)
        noise_TRN = sigma * sqrt_dt * self.rng.standard_normal(n_steps)

        # Pre-generate z-array noise (Stuart-Landau phase diffusion).
        # Uses auxiliary RNG to avoid contaminating WC noise sequence.
        n_z_osc = max(1, self.N - self.N_pop)
        z_update_interval = self._mf_sample_interval * 4
        n_z_updates = n_steps // z_update_interval + 1
        noise_z_real = self.sigma * self._aux_rng.standard_normal((n_z_updates, n_z_osc))
        noise_z_imag = self.sigma * self._aux_rng.standard_normal((n_z_updates, n_z_osc))
        _z_noise_idx = 0

        # Forcing gain
        forcing_gain = self.p.get('forcing_gain', 2.0)

        # Self-limiting forcing parameters
        coherence_power = self.p.get('coherence_power', 2.0)
        forcing_gain_pop = self.forcing_gain_pop  # Beta-distributed gains
        click_jitter_pop = self.click_jitter_pop  # per-pop temporal offsets

        # Local copies for hot loop
        E_pop = self.E_pop.copy()
        I_pop = self.I_pop.copy()
        A_pop = self.A_pop.copy()
        TC = self.TC
        TRN_val = self.TRN
        h_T = self.h_T
        forcing_mask_pop = self.forcing_mask_pop

        for step_i in range(n_steps):
            # ── BiologicalSSA for effective forcing ──
            E_mean = float(np.mean(E_pop))
            forcing_freq = external_freq_hz if forcing_strength > 0 else 0.0
            F_eff = self.biological_ssa.update(
                f_stim=forcing_freq,
                F=forcing_strength * nrem,
                neural_activity=E_mean,
                sleep_stage=sleep_stage,
                dt=dt,
            )
            # Update legacy A_fast/A_slow from biological_ssa for LOGGING
            bio_state = self.biological_ssa.get_state()
            self.A_fast = 1.0 - bio_state.get('x_mean', 1.0)
            self.A_slow = bio_state.get('Ca', 0.0)

            # ── Track forcing phase (used for PLV computation) ──
            current_forcing_phase = omega_ext * self.t if omega_ext > 0 else 0.0
            self._prev_forcing_phase = current_forcing_phase

            pulse_just_delivered = False

            # ── Update running coherence estimator ──
            # Track exponentially-weighted variance of mean-field signal.
            # High variance = synchronized UP/DOWN transitions across
            # populations (large collective oscillation amplitude).
            # Low variance = desynchronized (flat mean-field).
            # This is more informative than instantaneous E_pop std
            # because individual E_i are binary (0 or 1) regardless
            # of synchronization state.
            alpha_ema = min(1.0, dt / self._coherence_tau)
            self._mf_ema += alpha_ema * (E_mean - self._mf_ema)
            dev_sq = (E_mean - self._mf_ema) ** 2
            self._mf_emvar += alpha_ema * (dev_sq - self._mf_emvar)

            # Map running variance to coherence R in [0, 1].
            # Fully synchronized N=8: mf oscillates ~0.0-1.0, var ~ 0.12
            # Fully desynchronized: mf ~ 0.5 +/- noise, var ~ 0.003
            # Normalize so R=1 at max sync, R~0.15 at desync.
            mf_var_max = 0.125  # theoretical max for square-wave 0-1
            R_ensemble = float(np.clip(
                np.sqrt(self._mf_emvar / mf_var_max), 0.0, 1.0
            ))
            # Soft cap: coherence_atten only kicks in once R is large
            # (>0.7) to prevent runaway lock-in, but does NOT throttle
            # forcing under normal operating regimes the way the old
            # (1 - R^p) feedback did. Adaptation provides the main
            # self-limiting mechanism (Compte 2003; Levenstein 2019).
            if R_ensemble > 0.7:
                coherence_atten = max(0.0, 1.0 - ((R_ensemble - 0.7) / 0.3) ** 2)
            else:
                coherence_atten = 1.0

            # ── Update Ngo 2015 refractory gain (decays toward 0) ──
            ref_tau = self.p.get('refractory_tau', 3.0)
            self._refractory_gain *= np.exp(-dt / ref_tau)

            # ── Pulsed delivery with stored onset amplitude ──
            if pulsed and forcing_strength > 0:
                # Predict phase forward by phase_lead_sec to model real
                # CLAS device latency (~80 ms; Santostasi 2016, Garcia-
                # Molina 2018). Default 0 ms = perfect oracle.
                f_so_est = self._last_forcing_freq if self._last_forcing_freq > 0 else 0.75
                phase_advance = (2.0 * np.pi * f_so_est * self._so_phase_lead_sec)
                effective_phase = (self.so_phase + phase_advance) % (2.0 * np.pi)
                so_phase_diff = abs(effective_phase) if effective_phase <= np.pi \
                    else 2.0 * np.pi - effective_phase
                in_phase = so_phase_diff <= phase_window
                past_refrac = (
                    (self.t - self._last_pulse_time) >= refractory
                    or self._last_pulse_time < 0
                )

                if in_phase and past_refrac:
                    # Apply Ngo 2015 refractory gain decay to the pulse
                    # onset amplitude. The refractory gain saturates
                    # on cumulative pulse trains, matching the
                    # empirical finding that 2-click trains ≈ full
                    # driving stimulation.
                    eff_ngo = max(0.0, 1.0 - self._refractory_gain)
                    self._pulse_onset_amplitude = (
                        F_eff * phase_efficacy_gain * eff_ngo
                    )
                    self._last_pulse_time = self.t
                    pulse_just_delivered = True
                    # Increment refractory gain (saturates at 1)
                    ref_alpha = self.p.get('refractory_alpha', 0.50)
                    self._refractory_gain = min(
                        1.0, self._refractory_gain + ref_alpha
                    )
                    # Log phase + time at delivery for diagnostic
                    # histogram and ERP-locked analysis
                    self._pulse_phase_log.append(float(self.so_phase))
                    self._pulse_event_times.append(float(self._session_time))
                    self._pulse_count += 1

                if self._last_pulse_time >= 0:
                    t_since = self.t - self._last_pulse_time
                    if t_since < 5.0 * tau_response:
                        self._pulse_envelope = (
                            self._pulse_onset_amplitude
                            * np.exp(-t_since / tau_response)
                        )
                    else:
                        self._pulse_envelope = 0.0
                F_eff = self._pulse_envelope
            elif not pulsed and forcing_strength > 0:
                # CONTINUOUS: periodic DC clicks at fixed interval (Ngo 2015
                # open-loop paradigm). Deliver a brief unipolar positive
                # pulse every 1/freq seconds, regardless of SO phase.
                click_interval = 1.0 / external_freq_hz
                past_click_refrac = (
                    (self.t - self._last_click_time) >= click_interval
                    or self._last_click_time < 0
                )
                if past_click_refrac:
                    # Apply Ngo 2015 refractory gain decay to continuous
                    # clicks too (same self-limiting mechanism).
                    eff_ngo = max(0.0, 1.0 - self._refractory_gain)
                    self._click_onset_amplitude = F_eff * eff_ngo
                    self._last_click_time = self.t
                    pulse_just_delivered = True  # triggers KC check
                    ref_alpha = self.p.get('refractory_alpha', 0.50)
                    self._refractory_gain = min(
                        1.0, self._refractory_gain + ref_alpha
                    )
                    self._pulse_event_times.append(float(self._session_time))

                if self._last_click_time >= 0:
                    t_since_click = self.t - self._last_click_time
                    if t_since_click < 5.0 * tau_response:
                        self._click_envelope = (
                            self._click_onset_amplitude
                            * np.exp(-t_since_click / tau_response)
                        )
                    else:
                        self._click_envelope = 0.0
                F_eff = self._click_envelope

            # ── Per-population external forcing with self-limiting ──
            # Three mechanisms prevent runaway synchronization:
            #   1. coherence_atten: (1 - R^p) scales down as ensemble syncs
            #   2. forcing_gain_pop: Beta-distributed gains (graded, not binary)
            #   3. click_jitter_pop: per-population temporal dispersion
            if external_freq_hz > 0 and F_eff > 0:
                if pulsed and self._pulse_envelope > 0.001:
                    # PULSED: DC burst phase-locked to SO up-state.
                    # Per-population jitter: each pop receives the pulse
                    # at a slightly different time (auditory path length
                    # dispersion; Kaas & Hackett 2000).
                    if self._last_pulse_time >= 0:
                        t_since_pop = (self.t - self._last_pulse_time
                                       - click_jitter_pop)
                        # Only active for populations where t_since >= 0
                        pop_envelope = np.where(
                            (t_since_pop >= 0) & (t_since_pop < 5.0 * tau_response),
                            self._pulse_onset_amplitude * np.exp(-t_since_pop / tau_response),
                            0.0,
                        )
                    else:
                        pop_envelope = np.zeros(N)
                    I_ext_ctx_pop = (forcing_gain * coherence_atten
                                     * pop_envelope * forcing_gain_pop)
                elif not pulsed and self._click_envelope > 0.001:
                    # CONTINUOUS: periodic DC click with per-population
                    # jittered arrival times.
                    if self._last_click_time >= 0:
                        t_since_pop = (self.t - self._last_click_time
                                       - click_jitter_pop)
                        pop_envelope = np.where(
                            (t_since_pop >= 0) & (t_since_pop < 5.0 * tau_response),
                            self._click_onset_amplitude * np.exp(-t_since_pop / tau_response),
                            0.0,
                        )
                    else:
                        pop_envelope = np.zeros(N)
                    I_ext_ctx_pop = (forcing_gain * coherence_atten
                                     * pop_envelope * forcing_gain_pop)
                else:
                    I_ext_ctx_pop = np.zeros(N)
            else:
                I_ext_ctx_pop = np.zeros(N)

            # Thalamic external input (attenuated, also self-limited)
            if external_freq_hz > 0 and F_eff > 0:
                if (pulsed and self._pulse_envelope > 0.001) or \
                   (not pulsed and self._click_envelope > 0.001):
                    I_ext_thal = 0.3 * forcing_gain * F_eff * coherence_atten
                else:
                    I_ext_thal = 0.0
            else:
                I_ext_thal = 0.0

            # No down-state penalty needed: DC clicks are unipolar and
            # brief, so clicks landing during DOWN states are simply
            # ineffective (cortex is refractory) rather than actively
            # disruptive. This matches the biological reality that
            # auditory-evoked potentials are attenuated but not inverted
            # during cortical DOWN states (Ngo et al. 2015).

            # ── K-complex triggering ──
            # NOTE: In the hybrid architecture, KC effects are tracked for
            # the z-array's forcing dynamics (adapt_mod, kc_boost for z-array)
            # but do NOT modify the WC ensemble's A_pop. This ensures the
            # WC produces identical SO waveforms for stim and no-stim
            # conditions, with all enhancement coming from the z-array
            # Kuramoto coherence.
            kc_boost = 0.0
            if self._kc_active:
                self._kc_remaining_sec -= dt
                if self._kc_remaining_sec <= 0:
                    self._kc_active = False
                    self._kc_remaining_sec = 0.0
                else:
                    progress = 1.0 - self._kc_remaining_sec / KC_DURATION_SEC
                    kc_boost = KC_DELTA_BOOST * np.exp(-3.0 * progress)

            is_stimulus_event = pulse_just_delivered
            if is_stimulus_event and not self._kc_active and nrem > 0.3:
                p_kc = self.biological_ssa.get_kc_probability(sleep_stage) * nrem
                if pulsed:
                    p_kc *= 2.0
                if self._aux_rng.random() < p_kc:
                    self._kc_active = True
                    self._kc_remaining_sec = KC_DURATION_SEC
                    kc_boost = KC_DELTA_BOOST
                    self._kc_event_times.append(float(self._session_time))
                    # KC adaptation reset is handled by the z-array via
                    # adapt_mod, not by modifying WC A_pop directly.
                    self.biological_ssa.update_kc_habituation(True, dt)

            # ── Entrainment boost: adaptation modulation ──
            adapt_mod = 0.0
            entrainment_boost = self.p.get('entrainment_boost', 0.25)
            if F_eff > 0.001 and entrainment_boost > 0:
                if pulsed and self._pulse_envelope > 0.01:
                    adapt_mod = entrainment_boost * min(1.0, self._pulse_envelope / 0.05)
                elif not pulsed and self._click_envelope > 0.01:
                    # Continuous clicks: moderate adaptation modulation
                    # during the click envelope (regardless of SO phase,
                    # since clicks are not phase-locked).
                    adapt_mod = entrainment_boost * 0.5 * min(1.0, self._click_envelope / 0.05)

            # ── RK4 integration step (ensemble) ──
            # FIXED ARCHITECTURE: External forcing is now passed
            # directly into the WC ensemble integrator. The previous
            # "hybrid" architecture zeroed out I_ext_ctx and I_ext_thal
            # before the RK4 step, leaving forcing only in the parallel
            # Stuart-Landau z-array — but since the z-array never feeds
            # back into the WC dynamics, no closed-loop, phase, or
            # frequency targeting could affect the SO generator.
            #
            # The "binary synchronization problem" that motivated the
            # hybrid bypass is now controlled by:
            #   1. spike-frequency adaptation (g_adapt) — natural Up/Down
            #      self-limiting (Compte 2003; Levenstein 2019)
            #   2. heterogeneous I_tonic_pop — populations cannot trivially
            #      lock because they have different intrinsic frequencies
            #   3. Beta-distributed forcing_gain_pop — not all populations
            #      receive the full drive
            #   4. coherence_atten (kept as a soft cap, not the load
            #      bearing mechanism)
            #
            # Forcing is reduced by `wc_forcing_scale` to keep the
            # cortical depolarization in a physiological range (Costa
            # 2016: ~0.05–0.15 above baseline E, mimicking ~1 mV EPSP).
            wc_forcing_scale = self.p.get('wc_forcing_scale', 0.10)
            I_ext_ctx_to_wc = I_ext_ctx_pop * wc_forcing_scale
            I_ext_thal_to_wc = I_ext_thal * wc_forcing_scale
            E_new, I_new, A_new, TC_new, TRN_new, h_T_new = self._rk4_step_ensemble(
                E_pop, I_pop, A_pop, TC, TRN_val, h_T,
                dt,
                I_ext_ctx_to_wc, I_sleep_ctx,
                I_ext_thal_to_wc, I_sleep_thal,
                kc_boost,
                I_tonic_eff_pop,
                adapt_mod,
            )

            # Add noise (per-population for E, I; scalar for TC, TRN)
            E_new = E_new + noise_E[step_i]
            I_new = I_new + noise_I[step_i]
            TC_new = TC_new + noise_TC[step_i]
            TRN_new = TRN_new + noise_TRN[step_i]

            # Soft clip — see _rk4_step_ensemble for rationale.
            np.clip(E_new, -0.05, 1.05, out=E_new)
            np.clip(I_new, -0.05, 1.05, out=I_new)
            np.clip(A_new, 0.0, 2.0, out=A_new)
            TC_new = float(np.clip(TC_new, 0.0, 1.0))
            TRN_new = float(np.clip(TRN_new, 0.0, 1.0))
            h_T_new = float(np.clip(h_T_new, 0.0, 1.0))

            E_pop = E_new
            I_pop = I_new
            A_pop = A_new
            TC = TC_new
            TRN_val = TRN_new
            h_T = h_T_new

            # ── Update slow variables ──
            self.T = float(TC)

            self.H += (self.T - self.H) / self.tau_H * dt - self.H * self.homeo_rate * dt
            self.H = max(0.0, self.H)

            self._update_process_s(dt)

            # Update I_tonic_eff_pop as S evolves
            I_tonic_eff_base = self.p['I_tonic'] * (1.0 + 1.0 * self.S)
            I_tonic_eff_pop = I_tonic_eff_base * (
                1.0 + spread * np.linspace(-1, 1, N)
            )

            self._session_time += dt
            if step_i % 1000 == 0:
                nrem = self._nrem_gate()
                I_sleep_ctx = 0.3 * self.I_sleep * self.S * nrem
                I_sleep_thal = 0.2 * self.I_sleep * self.S * nrem

            self.t += dt

            # ── Sample mean-field signal into buffer ──
            # HYBRID ARCHITECTURE: The mean-field signal combines:
            #   1. Wilson-Cowan SO waveform (biophysically correct UP/DOWN,
            #      adaptation, spindles) from mean(E_pop)
            #   2. Kuramoto spatial coherence (graded synchronization,
            #      frequency-selective entrainment) from the z-array
            #
            # The WC ensemble provides the local SO dynamics of one
            # representative cortical column. The Kuramoto z-array provides
            # the spatial coherence factor R that modulates how much of
            # the local SO appears in the scalp EEG (mean-field).
            #
            # Signal model: mf(t) = E_wc(t) * R_spatial(t)
            #   where R_spatial = R_baseline + (R_kuramoto - R_baseline)
            # This is biophysically justified because scalp EEG amplitude
            # is proportional to the product of local dipole strength
            # (from WC) and spatial coherence (from Kuramoto).
            #
            # Enhancement mechanism: forcing increases R_kuramoto from
            # ~R_baseline to R_baseline + dR, yielding:
            #   SWA_enh = (R_stim/R_base)^2 - 1  (power is amplitude^2)
            # For 15% SWA enhancement: R_stim/R_base = sqrt(1.15) = 1.072
            # So dR ~ 0.07 * R_base ~ 0.03 (very modest R increase).
            self._step_counter += 1
            if self._step_counter % self._mf_sample_interval == 0:
                if self._mf_idx == 0:
                    self._mf_start_time = self._session_time

                # Mean-field signal is now the raw WC ensemble mean E.
                # Previously this was multiplied by a hand-calibrated
                # (1 + alpha_R * (R_smooth - R_baseline_est)) gain that
                # was the only path by which forcing influenced SWA in
                # the bypassed-WC architecture. With forcing now routed
                # directly into the WC integrator, the SWA enhancement
                # emerges from real cortical dynamics, not a multiplier.
                #
                # The Stuart–Landau z-array is still updated below as a
                # diagnostic (PLV, ensemble coherence), but no longer
                # contributes to the mean-field readout.
                #
                # RESPONDER-WEIGHTED EEG PROXY (, Fix 1):
                # Real scalp EEG at Fz/Cz is NOT a global average over
                # all cortex — it is a weighted spatial integral dominated
                # by frontal sources within ~6 cm of the electrode
                # (Nunez & Srinivasan 2006). Our forcing_gain_pop array
                # indexes each population's coupling to the auditory
                # pathway: high-gain populations represent columns near
                # the auditory/frontal target region (close to Fz),
                # while zero-gain populations (non-responders) represent
                # distant regions. Weighting by forcing_gain_pop models
                # the electrode sensitivity profile.
                #
                # This fixes the 1/√N ensemble-averaging cancellation
                # that diluted the SWA effect from +20% per-population
                # to +0.7% on the unweighted mean (,
                #). Every published CLAS model that validates
                # against experimental SWA uses either a single cortical
                # node or a spatially-coupled field — none use an
                # unweighted mean of uncoupled heterogeneous populations.
                _fg = forcing_gain_pop
                _fg_sum = float(np.sum(_fg))
                if _fg_sum > 1e-10:
                    E_wc = float(np.dot(_fg, E_pop) / _fg_sum)
                else:
                    E_wc = float(np.mean(E_pop))
                mf = E_wc

                # Track Kuramoto R from the z-array as a diagnostic only.
                z_mean = np.mean(self.z)
                R_kuramoto = float(np.abs(z_mean))
                z_amp_mean = float(np.mean(np.abs(self.z)))
                if z_amp_mean > 1e-8:
                    R_instant = R_kuramoto / z_amp_mean
                else:
                    R_instant = 0.0
                R_instant = min(1.0, R_instant)
                tau_R_smooth = 5.0
                alpha_R_ema = min(1.0, (1.0 / self._mf_fs) / tau_R_smooth)
                self._R_smooth += alpha_R_ema * (R_instant - self._R_smooth)

                self._mf_buffer[self._mf_idx % self._mf_buffer_size] = mf
                self._mf_idx += 1
                # Update SO phase from mean-field dynamics
                self._update_emergent_so_phase(mf)

            # ── Update z-array: full Stuart-Landau dynamics ──
            # The z-array implements N=64 Stuart-Landau oscillators with:
            #   dz_i/dt = (lambda_i + i*omega_i)*z_i - |z_i|^2*z_i
            #             + K/N * sum(z_j)
            #             + F_i * exp(i*forcing_phase) * so_gate
            #
            # This provides the spatial coherence R that modulates the
            # mean-field signal. The WC model provides:
            #   - lambda_i modulated by thalamic state H (sleep depth)
            #   - so_gate from the WC SO phase (forcing only during UP)
            #   - E_pop amplitude for the first N_pop oscillators
            #
            # The full Stuart-Landau dynamics (not just phase) enable
            # amplitude growth near resonance, which is essential for
            # the cascade synchronization mechanism (Pikovsky 2001).
            if self._step_counter % (self._mf_sample_interval * 4) == 0:
                dt_block = dt * (self._mf_sample_interval * 4)

                # First N_pop entries: driven by WC E_pop
                self.z[:self.N_pop] = self.E_pop * np.exp(
                    1j * (self.natural_freqs[:self.N_pop] * self.t)
                )

                # Remaining oscillators: full Stuart-Landau dynamics
                if self.N > self.N_pop:
                    z_sub = self.z[self.N_pop:]
                    omega_sub = self.natural_freqs[self.N_pop:]
                    lambda_sub = self.lambda_0[self.N_pop:].copy()

                    # WC modulation of lambda: thalamic state deepens
                    # NREM excitability (more positive lambda = larger
                    # limit cycle amplitude)
                    sig_H = 1.0 / (1.0 + np.exp(-self.kappa * (self.H - self.T_half)))
                    lambda_sub += self.delta_lambda * sig_H

                    # Mean-field coupling
                    z_bar = np.mean(self.z)
                    K_mf = self.K

                    # Stuart-Landau ODE: dz/dt = (lambda+i*omega)*z - |z|^2*z + K*z_bar + F
                    dz = ((lambda_sub + 1j * omega_sub) * z_sub
                          - np.abs(z_sub)**2 * z_sub
                          + K_mf * z_bar)

                    # External forcing with frequency-selective resonance
                    if F_eff > 0.001 and omega_ext > 0:
                        forcing_phase = omega_ext * self.t
                        detuning = omega_sub - omega_ext
                        resonance_bw = 2.0 * 2 * np.pi
                        resonance_gain = 1.0 / (1.0 + (detuning / resonance_bw) ** 2)

                        # SO gate: forcing only effective during UP state
                        # (WC-derived phase gating)
                        so_gate = 0.5 * (1.0 + np.cos(self.so_phase))

                        f_coupling = (self.beta_ext * forcing_strength
                                      * resonance_gain
                                      * self.forcing_mask[self.N_pop:]
                                      * so_gate)
                        # Pulsed delivery has higher effective coupling
                        # (concentrated in UP phase)
                        if pulsed and self._pulse_envelope > 0.01:
                            f_coupling *= phase_efficacy_gain
                        elif not pulsed and self._click_envelope > 0.01:
                            # Continuous: clicks arrive at random SO phases
                            # so effective coupling is lower than pulsed
                            pass

                        f_coupling *= (1.0 + adapt_mod * 3.0)
                        dz += f_coupling * np.exp(1j * forcing_phase)

                    # Noise (phase diffusion) — from pre-generated arrays
                    # to ensure identical WC noise sequence across conditions
                    zi = min(_z_noise_idx, n_z_updates - 1)
                    noise_z = np.sqrt(dt_block) * (
                        noise_z_real[zi, :len(z_sub)]
                        + 1j * noise_z_imag[zi, :len(z_sub)]
                    )
                    dz += noise_z / dt_block  # noise intensity per unit time
                    _z_noise_idx += 1

                    # Euler step
                    self.z[self.N_pop:] = z_sub + dz * dt_block
                    # Clamp amplitude to prevent blowup
                    amp = np.abs(self.z[self.N_pop:])
                    too_big = amp > 2.0
                    if too_big.any():
                        self.z[self.N_pop:][too_big] *= 2.0 / amp[too_big]

        # Store final ensemble state back to self
        self.E_pop = E_pop
        self.I_pop = I_pop
        self.A_pop = A_pop
        self.TC = TC
        self.TRN = TRN_val
        self.h_T = h_T

        # Update scalar backward-compat attributes from ensemble mean
        self.E = float(np.mean(self.E_pop))
        self.I = float(np.mean(self.I_pop))
        self.A_adapt = float(np.mean(self.A_pop))

    def run_epoch_pulsed(
        self,
        duration_sec: float,
        external_freq_hz: float,
        forcing_strength: float,
    ) -> Dict:
        """
        Run a pulsed stimulation epoch (SO phase-locked delivery).

        Returns:
            Dict with n_pulses and pulse_duty_cycle.
        """
        initial_pulse_time = self._last_pulse_time

        self.run_epoch(
            duration_sec, external_freq_hz, forcing_strength,
            pulsed=True,
        )

        so_freq_est = 0.75
        n_pulses_est = max(1, int(duration_sec * so_freq_est * 0.8))
        duty_cycle_est = 0.30

        if forcing_strength <= 0:
            n_pulses_est = 0
            duty_cycle_est = 0.0

        return {
            'n_pulses': n_pulses_est,
            'pulse_duty_cycle': duty_cycle_est,
        }

    # ─── Output metrics ──────────────────────────────────────────

    def compute_band_powers(self) -> Dict[str, float]:
        """
        Derive band powers from the mean-field signal PSD via Welch's method.
        """
        n_valid = min(self._mf_idx, self._mf_buffer_size)
        if n_valid < 16:
            return {
                'delta_power': 0.25, 'theta_power': 0.25,
                'alpha_power': 0.25, 'beta_power': 0.25,
                'delta_power_abs': 0.0, 'theta_power_abs': 0.0,
                'alpha_power_abs': 0.0, 'beta_power_abs': 0.0,
            }

        # Read buffer in time-order. The buffer is circular: the most
        # recent sample is at (_mf_idx - 1) % buffer_size and the
        # oldest valid sample is at _mf_idx % buffer_size when the
        # buffer is full. Reading the raw buffer directly creates a
        # discontinuity that contaminates the PSD.
        if self._mf_idx >= self._mf_buffer_size:
            # Buffer has wrapped — roll to time-order
            start = self._mf_idx % self._mf_buffer_size
            mf_signal = np.roll(self._mf_buffer, -start)
        else:
            mf_signal = self._mf_buffer[:n_valid]
        fs = self._mf_fs

        # Use a larger nperseg for finer frequency resolution in the
        # narrow 0.5-4 Hz delta band. With fs=200 Hz, nperseg=1024 gives
        # ~0.2 Hz resolution (17 frequency bins in delta) vs the
        # previous 0.78 Hz (4 bins). Finer resolution dramatically
        # reduces epoch-to-epoch variance of integrated power, which
        # is what makes the baseline_swa estimate stable.
        nperseg = min(1024, n_valid)
        freqs, psd = sp_signal.welch(
            mf_signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
        )

        abs_powers = {}
        for band, (lo, hi) in BAND_EDGES.items():
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                abs_powers[f'{band}_power_abs'] = float(
                    np.trapezoid(psd[mask], freqs[mask])
                )
            else:
                abs_powers[f'{band}_power_abs'] = 0.0

        total = sum(abs_powers.values())
        band_powers = {}
        if total > 0:
            for band in BAND_EDGES:
                band_powers[f'{band}_power'] = abs_powers[f'{band}_power_abs'] / total
        else:
            band_powers = {
                'delta_power': 0.25, 'theta_power': 0.25,
                'alpha_power': 0.25, 'beta_power': 0.25,
            }

        band_powers.update(abs_powers)
        return band_powers

    def compute_plv(self, external_freq_hz: float) -> float:
        """
        Phase-locking value between mean-field signal and external drive.

        Uses a narrow bandpass (+/-0.5 Hz) around the forcing frequency to
        isolate the entrained component, then computes circular PLV against
        the reference phase of the external drive.
        """
        if external_freq_hz <= 0:
            return 0.0

        n_valid = min(self._mf_idx, self._mf_buffer_size)
        if n_valid < 64:
            return 0.0

        mf_signal = self._mf_buffer[:n_valid]
        fs = self._mf_fs

        # Trim initial transient (~2 s) to avoid startup phase artifacts
        n_trim = min(int(2.0 * fs), n_valid // 3)
        mf_signal = mf_signal[n_trim:]
        n_valid = len(mf_signal)
        if n_valid < 64:
            return 0.0

        # Narrow bandpass: +/-0.5 Hz around forcing frequency
        # This isolates the entrained component from the intrinsic SO
        bw = 0.5
        lo = max(0.3, external_freq_hz - bw)
        hi = min(fs / 2 - 0.1, external_freq_hz + bw)
        if lo >= hi:
            return 0.0

        try:
            sos = sp_signal.butter(3, [lo, hi], btype='band', fs=fs, output='sos')
            filtered = sp_signal.sosfiltfilt(sos, mf_signal)
        except (ValueError, np.linalg.LinAlgError):
            return 0.0

        # Remove residual DC before Hilbert transform
        filtered = filtered - np.mean(filtered)

        # Skip if filtered signal has negligible amplitude (no entrainment)
        if np.std(filtered) < 1e-10:
            return 0.0

        analytic = sp_signal.hilbert(filtered)
        signal_phase = np.angle(analytic)

        # Reference phase: must match the forcing phase omega_ext * t
        # used in the integration loop (lines 1267, 1277)
        t_samples = (self._mf_start_time + (n_trim + np.arange(n_valid)) / fs)
        ext_phase = 2.0 * np.pi * external_freq_hz * t_samples

        phase_diff = signal_phase - ext_phase
        plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))

        return plv

    def compute_clas_outcome_metrics(self) -> Dict[str, float]:
        """
        Compute CLAS-relevant outcome metrics from the most recent
        mean-field buffer and the event logs.

        Metrics returned:
            erp_so_amplitude_uv : trough-to-peak SO ERP amplitude averaged
                over [-0.5, +1.5] s windows around each pulse event.
            erp_so_n_trials : number of trials averaged.
            spindle_rms : RMS of 12-15 Hz bandpassed signal in
                [+0.5, +1.5] s post-pulse.
            slow_wave_slope : mean descending-limb slope (uV/s) of detected
                negative-going SOs.
            kc_density_per_min : K-complex events per minute over the
                most recent buffer window.
            pulse_phase_concentration : circular mean resultant length of
                logged pulse phases (1.0 = perfectly concentrated, 0 = uniform).
            pulse_phase_mean : circular mean of logged pulse phases (rad).
            n_pulses_total : total pulses logged this run.

        Returns zero-filled dict if the buffer is too short or no events
        have been logged. The metrics are computed against the
        mean-field buffer at its current sampling rate (~256 Hz).
        """
        out = {
            'erp_so_amplitude_uv': 0.0,
            'erp_so_n_trials': 0,
            'spindle_rms': 0.0,
            'slow_wave_slope': 0.0,
            'kc_density_per_min': 0.0,
            'pulse_phase_concentration': 0.0,
            'pulse_phase_mean': 0.0,
            'n_pulses_total': int(self._pulse_count),
        }

        n_valid = min(self._mf_idx, self._mf_buffer_size)
        if n_valid < 64:
            return out

        mf_signal = self._mf_buffer[:n_valid].copy()
        mf_signal = mf_signal - float(np.mean(mf_signal))
        fs = self._mf_fs

        # ── ERP-locked SO amplitude ────────────────────────────────
        # Bandpass 0.5-4 Hz, then average a [-0.5, +1.5] s window around
        # every pulse-event time that falls inside the buffer's time
        # range. Buffer covers from `_mf_start_time` (an absolute
        # session time) to `_mf_start_time + n_valid/fs`.
        if len(self._pulse_event_times) > 0 and self._so_sos is not None:
            try:
                # Bandpass for SO
                sos_so = sp_signal.butter(
                    3, [0.5, 4.0], btype='band', fs=fs, output='sos'
                )
                so_filt = sp_signal.sosfiltfilt(sos_so, mf_signal)
                # Window for ERP: -0.5 to +1.5 s
                pre_samp = int(0.5 * fs)
                post_samp = int(1.5 * fs)
                erp_acc = []
                for t_pulse in self._pulse_event_times:
                    rel = t_pulse - self._mf_start_time
                    center = int(round(rel * fs))
                    if (center - pre_samp) >= 0 and (center + post_samp) < n_valid:
                        seg = so_filt[center - pre_samp:center + post_samp]
                        erp_acc.append(seg)
                if erp_acc:
                    erp = np.mean(erp_acc, axis=0)
                    # Trough-to-peak amplitude (proxy for ERP magnitude)
                    out['erp_so_amplitude_uv'] = float(np.max(erp) - np.min(erp))
                    out['erp_so_n_trials'] = int(len(erp_acc))

                    # Slow-wave slope: avg descending-limb slope from
                    # ERP peak to following trough
                    pk_idx = int(np.argmax(erp))
                    tr_idx = int(np.argmin(erp[pk_idx:])) + pk_idx
                    if tr_idx > pk_idx:
                        dt_seg = (tr_idx - pk_idx) / fs
                        if dt_seg > 0:
                            out['slow_wave_slope'] = float(
                                (erp[pk_idx] - erp[tr_idx]) / dt_seg
                            )
            except (ValueError, np.linalg.LinAlgError):
                pass

            # ── Spindle RMS in [+0.5, +1.5] s post-pulse ────────────
            try:
                sos_sp = sp_signal.butter(
                    3, [12.0, 15.0], btype='band', fs=fs, output='sos'
                )
                sp_filt = sp_signal.sosfiltfilt(sos_sp, mf_signal)
                start = int(0.5 * fs)
                end = int(1.5 * fs)
                rms_acc = []
                for t_pulse in self._pulse_event_times:
                    rel = t_pulse - self._mf_start_time
                    center = int(round(rel * fs))
                    if (center + start) >= 0 and (center + end) < n_valid:
                        seg = sp_filt[center + start:center + end]
                        rms_acc.append(float(np.sqrt(np.mean(seg ** 2))))
                if rms_acc:
                    out['spindle_rms'] = float(np.mean(rms_acc))
            except (ValueError, np.linalg.LinAlgError):
                pass

        # ── K-complex density ──────────────────────────────────────
        # Convert to events / minute over the buffer window length.
        buffer_window_sec = n_valid / fs
        if buffer_window_sec > 0:
            recent_kcs = [
                t for t in self._kc_event_times
                if t >= self._mf_start_time
            ]
            out['kc_density_per_min'] = float(
                60.0 * len(recent_kcs) / buffer_window_sec
            )

        # ── Pulse phase concentration (circular statistics) ────────
        if len(self._pulse_phase_log) > 0:
            phases = np.asarray(self._pulse_phase_log)
            mean_vec = np.mean(np.exp(1j * phases))
            out['pulse_phase_concentration'] = float(np.abs(mean_vec))
            out['pulse_phase_mean'] = float(np.angle(mean_vec))

        return out

    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Compute synchronization order parameter from the ensemble.

        R measures the coherence across N cortical populations.
        When populations are synchronized (by forcing), R is high.
        """
        # Primary: order parameter from E_pop ensemble
        # Map E_pop to complex phases using their relative deviations
        E_mean = np.mean(self.E_pop)
        # Compute instantaneous phase-like variable from each population
        # Use the z-array for proper Kuramoto order parameter
        z_mean = np.mean(self.z)
        R_z = float(np.abs(z_mean))
        psi = float(np.angle(z_mean))

        # Ensemble coherence: standard deviation of E_pop
        # When synchronized: std is low, all E_i ~ E_mean
        E_std = float(np.std(self.E_pop))
        # Map to coherence: 1.0 when std=0 (perfect sync), lower when desync
        ensemble_coherence = float(np.exp(-10.0 * E_std))

        # TC-TRN coherence measure
        tc_trn_coherence = float(np.sqrt(self.TC * self.TRN))

        # Blend: 40% z-order param + 30% ensemble coherence + 30% TC-TRN
        R = 0.4 * R_z + 0.3 * ensemble_coherence + 0.3 * tc_trn_coherence

        return R, psi

    # ─── Legacy-compatible effective params ───────────────────────

    def _effective_params(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute effective omega, lambda, and SO gate (legacy compat)."""
        sig_T = _sigmoid(self.T, a=self.kappa, theta=self.T_half)
        omega_eff = self.natural_freqs * (1.0 - self.gamma * sig_T)

        sig_H = _sigmoid(self.H, a=self.kappa, theta=self.T_half)
        lambda_eff = self.lambda_0 + self.delta_lambda * sig_H

        so_gate = 0.5 * (1.0 + np.cos(self.so_phase))

        return omega_eff, lambda_eff, float(so_gate)

    # ─── Frequency scan ──────────────────────────────────────────

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
        Same interface as TSLE.frequency_scan().

        Baseline is measured AFTER a burn-in period (BURN_IN_EPOCHS * 30 sec)
        that lets the ensemble reach steady-state partial synchrony through
        the shared thalamocortical loop. This prevents the desynchronized
        initial state (mean-field power ~ A^2/N) from inflating all
        enhancement values by ~39%.
        """
        self.initialize_from_baseline(
            baseline_powers,
            non_responder_fraction=non_responder_fraction,
        )

        saved_state = self.get_state()

        # Baseline (no forcing) — with burn-in for thalamocortical sync
        self._reset_for_scan()
        # Burn-in: let thalamocortical coupling reach partial-synchrony
        # attractor before measuring baseline. The shared TC/TRN loop
        # mediates indirect coupling with K_eff >> K_critical, so the
        # transient settles in ~30-90 sec. We run BURN_IN_EPOCHS epochs.
        burn_in_sec = BURN_IN_EPOCHS * 30.0
        total_warmup = warmup_sec + burn_in_sec
        self.run_epoch(total_warmup, 1.0, 0.0)
        self._mf_buffer[:] = 0.0
        self._mf_idx = 0
        self._step_counter = 0
        self.run_epoch(measurement_sec, 1.0, 0.0)
        baseline_bp = self.compute_band_powers()
        baseline_sdr = compute_sdr(baseline_bp)
        baseline_swa = compute_swa(baseline_bp)

        results = []
        for freq in test_frequencies:
            self.set_state(saved_state)
            self._reset_dynamic_state()

            # Same burn-in for each test frequency so baseline and stim
            # conditions start from the same synchronization state
            self.run_epoch(total_warmup, freq, forcing_strength)

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

    # ─── Progressive session ─────────────────────────────────────

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
        Simulate a multi-phase entrainment session with continuous state.
        Exact same interface as TSLE.run_progressive_session().

        Burn-in: runs BURN_IN_EPOCHS epochs of no-stimulation before
        measurement begins. This lets the N-population ensemble reach
        the thalamocortical partial-synchrony attractor, so the baseline
        SWA reflects steady-state (not the desynchronized initial condition
        where mean-field power = A^2/N instead of ~A^2).

        Burn-in and baseline procedure (matches real CLAS protocol):
        ------------------------------------------------------------
        1. Initialize cortical/thalamic state from subject band powers.
        2. Run BURN_IN_EPOCHS (5) no-stim epochs to let fast transients
           settle (tau_adapt=350ms, tau_TC=20ms fully settled; ensemble
           partial-synchrony attractor reached in ~90s).
        3. Compute baseline SDR and SWA as the MEAN over the last
           BURN_IN_BASELINE_EPOCHS (3) of the burn-in.  This is Option A
           (within-subject paired baseline) -- the most methodologically
           sound approach because:
           - It matches each subject's own pre-stim NREM sleep
           - It captures the natural partial-synchrony steady state
           - It avoids the ~39% spurious enhancement from desynchronized
             initial conditions
        4. Reset _session_time to 0 so the stimulation epoch clock starts
           fresh (the burn-in represents the pre-stim sleep period).
        5. Run protocol phases with stimulation.

        The NREM gate is constant (1.0) throughout because the 60-min
        session represents the stimulation-active window within a single
        NREM period. Homeostatic decline is handled by Process S
        (tau=4.2h, ~21% decay over 60 min).
        """
        if not skip_init:
            self.initialize_from_baseline(
                baseline_powers,
                non_responder_fraction=non_responder_fraction,
                sleep_stage_fractions=sleep_stage_fractions,
            )

        # ── Burn-in: reach thalamocortical partial-synchrony attractor ──
        # The ensemble starts with desynchronized phases. The shared
        # thalamic loop (TC/TRN) mediates indirect mean-field coupling
        # with effective K ~ w_CT * w_TC_ctx * sigmoid_gain^2 / (tau_TC * tau_E).
        # This far exceeds the critical coupling for N=8, spread=0.12,
        # so the transient settles in ~30-90 sec.
        #
        # We split BURN_IN_EPOCHS into two phases:
        #   Phase 1: (BURN_IN_EPOCHS - BURN_IN_BASELINE_EPOCHS) settling
        #            epochs -- output discarded, just for transient decay.
        #   Phase 2: BURN_IN_BASELINE_EPOCHS measurement epochs -- averaged
        #            to produce the paired no-stim baseline (SDR, SWA).
        if baseline_sdr is None:
            # Phase 1: settling (no measurement) — no buffer resets
            # between epochs so the PSD always sees a continuous buffer
            # of steady-state samples, never a A^2/N -> A^2 transient.
            n_settle = max(0, BURN_IN_EPOCHS - BURN_IN_BASELINE_EPOCHS)
            for _ in range(n_settle):
                self.run_epoch(epoch_sec, 1.0, 0.0)

            # Phase 2: baseline measurement (no-stim, post-settling).
            # We average band powers across BURN_IN_BASELINE_EPOCHS
            # WITHOUT resetting the mean-field buffer between epochs.
            # This was the source of the +150% phantom enhancement —
            # each buffer reset forced the Welch PSD to start at the
            # desynchronized mean-field power and ramp back up.
            baseline_sdr_accum = []
            baseline_swa_accum = []
            for _ in range(BURN_IN_BASELINE_EPOCHS):
                self.run_epoch(epoch_sec, 1.0, 0.0)
                bp = self.compute_band_powers()
                baseline_sdr_accum.append(compute_sdr(bp))
                baseline_swa_accum.append(compute_swa(bp))

            baseline_sdr = float(np.mean(baseline_sdr_accum))
            baseline_swa = float(np.mean(baseline_swa_accum))
            self._baseline_swa = baseline_swa
        else:
            # Caller provided baseline_sdr; still run burn-in for settling
            # without buffer resets so band powers reflect steady state.
            for _ in range(BURN_IN_EPOCHS):
                self.run_epoch(epoch_sec, 1.0, 0.0)
            if self._baseline_swa is None:
                self._baseline_swa = compute_swa(self.compute_band_powers())
            baseline_swa = self._baseline_swa

        # ── Reset session clock: stimulation starts at t=0 ──
        # The burn-in represents pre-stim NREM sleep. Reset so the
        # NREM gate and session-time tracking start from the stim onset.
        # Do NOT reset the mean-field buffer — keep continuous so PSD
        # operates on steady-state samples rather than restart transients.
        self._session_time = 0.0

        results = []
        epoch_idx = 0
        cumulative_time = 0.0

        for phase in protocol_phases:
            freq = phase['freq']
            phase_duration = phase['duration_sec']
            phase_name = phase['name']
            n_epochs = max(1, int(phase_duration / epoch_sec))

            for _ in range(n_epochs):
                # No buffer reset: PSD reads the most-recent buffer-full
                # of mean-field samples (~32 s at 256 Hz). Each epoch
                # advances the rolling buffer rather than discarding it.

                epoch_freq = freq if freq > 0 else 1.0
                epoch_forcing = forcing_strength if freq > 0 else 0.0

                pulse_info = {'n_pulses': 0, 'pulse_duty_cycle': 0.0}
                if stim_mode == 'pulsed' and epoch_forcing > 0:
                    pulse_info = self.run_epoch_pulsed(
                        epoch_sec, epoch_freq, epoch_forcing,
                    )
                else:
                    self.run_epoch(epoch_sec, epoch_freq, epoch_forcing)

                bp = self.compute_band_powers()
                r, _ = self.compute_order_parameter()
                plv = self.compute_plv(freq) if freq > 0 else 0.0
                sdr = compute_sdr(bp)
                sdre = sdr - baseline_sdr
                swa = compute_swa(bp)
                swa_enh = compute_swa_enhancement(swa, baseline_swa)
                # CLAS outcome metrics (ERP, KC density, etc.) are now
                # computed once per session at the end (see below) instead
                # of once per epoch — ERP averaging needs many trials so
                # per-epoch values are unstable AND the computation cost
                # was 5-10x the per-epoch cost. Stored zero placeholders
                # here so the row schema stays consistent.
                clas_metrics = {
                    'erp_so_amplitude_uv': 0.0,
                    'erp_so_n_trials': 0,
                    'spindle_rms': 0.0,
                    'slow_wave_slope': 0.0,
                    'kc_density_per_min': 0.0,
                    'pulse_phase_concentration': 0.0,
                    'pulse_phase_mean': 0.0,
                    'n_pulses_total': int(self._pulse_count),
                }

                thalamic_T = self.T
                thalamic_H = self.H
                adaptation_fast = self.A_fast
                adaptation_slow = self.A_slow
                mean_amplitude = float(self.E)
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
                    'thalamic_T': thalamic_T,
                    'thalamic_H': thalamic_H,
                    'adaptation': adaptation_fast,
                    'adaptation_fast': adaptation_fast,
                    'adaptation_slow': adaptation_slow,
                    'mean_amplitude': mean_amplitude,
                    'mean_omega_hz': mean_omega_hz,
                    'so_phase': self.so_phase,
                    'so_gate': so_gate_val,
                    'E': self.E,
                    'I': self.I,
                    'TC': self.TC,
                    'TRN': self.TRN,
                    'h_T': self.h_T,
                    'process_S': self.S,
                    'nrem_gate': self._nrem_gate(),
                }
                if stim_mode == 'pulsed':
                    row['n_pulses'] = pulse_info['n_pulses']
                    row['pulse_duty_cycle'] = pulse_info['pulse_duty_cycle']

                # Tier-4 CLAS outcome metrics (ERP, KC density, spindle,
                # phase concentration). These are per-epoch snapshots
                # of metrics that are themselves windowed over the
                # rolling mean-field buffer.
                row['erp_so_amplitude_uv'] = clas_metrics['erp_so_amplitude_uv']
                row['erp_so_n_trials'] = clas_metrics['erp_so_n_trials']
                row['spindle_rms'] = clas_metrics['spindle_rms']
                row['slow_wave_slope'] = clas_metrics['slow_wave_slope']
                row['kc_density_per_min'] = clas_metrics['kc_density_per_min']
                row['pulse_phase_concentration'] = clas_metrics['pulse_phase_concentration']
                row['pulse_phase_mean'] = clas_metrics['pulse_phase_mean']
                row['n_pulses_total'] = clas_metrics['n_pulses_total']

                results.append(row)
                epoch_idx += 1

        # ── Final session-level CLAS metrics ──
        # Compute ERP-locked outcomes ONCE at the end of the session
        # (after enough pulses have been delivered to average over).
        # Write the values to the LAST row only — compute_extended_metrics
        # picks them up via the tail mean.
        final_clas = self.compute_clas_outcome_metrics()
        if results:
            results[-1].update({
                'erp_so_amplitude_uv': final_clas['erp_so_amplitude_uv'],
                'erp_so_n_trials': final_clas['erp_so_n_trials'],
                'spindle_rms': final_clas['spindle_rms'],
                'slow_wave_slope': final_clas['slow_wave_slope'],
                'kc_density_per_min': final_clas['kc_density_per_min'],
                'pulse_phase_concentration': final_clas['pulse_phase_concentration'],
                'pulse_phase_mean': final_clas['pulse_phase_mean'],
                'n_pulses_total': final_clas['n_pulses_total'],
            })

        return pd.DataFrame(results)


# ─── Module-level aliases for pipeline compatibility ──────────────────

# Backward-compatibility aliases
MPRThalamocorticalEnsemble = WCThalamocorticalEnsemble
ThalamocorticalEnsemble = WCThalamocorticalEnsemble


# ─── Self-test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 70)
    print("WC Thalamocortical Model — Calibration Test")
    print("  Wilson-Cowan transfer function + adaptation + thalamocortical")
    print("=" * 70)

    # Quick calibration test with burn-in for correct baseline.
    # Uses absolute SWA (delta_power_abs) which is the standard
    # metric in CLAS literature (Ngo 2013; Besedovsky 2017).
    bp_init = {'delta': 0.45, 'theta': 0.25, 'alpha': 0.20, 'beta': 0.10}

    # Baseline
    m = WCThalamocorticalEnsemble(seed=42)
    m.initialize_from_baseline(bp_init)
    print(f"Burn-in: {BURN_IN_EPOCHS} epochs ({BURN_IN_EPOCHS * 30}s)")
    for _ in range(BURN_IN_EPOCHS):
        m.run_epoch(30.0, 0.0, 0.0, sleep_stage='N2')
    m._mf_buffer[:] = 0.0
    m._mf_idx = 0
    m._step_counter = 0
    for _ in range(3):
        m.run_epoch(30.0, 0.0, 0.0, sleep_stage='N2')
    bp_base = m.compute_band_powers()
    swa_base = compute_swa(bp_base)

    # Continuous (1 Hz clicks)
    m_c = WCThalamocorticalEnsemble(seed=42)
    m_c.initialize_from_baseline(bp_init)
    for _ in range(BURN_IN_EPOCHS):
        m_c.run_epoch(30.0, 0.0, 0.0, sleep_stage='N2')
    m_c._mf_buffer[:] = 0.0
    m_c._mf_idx = 0
    m_c._step_counter = 0
    for _ in range(3):
        m_c.run_epoch(30.0, 1.0, 0.10, sleep_stage='N2', pulsed=False)
    bp_cont = m_c.compute_band_powers()
    swa_cont = compute_swa(bp_cont)
    enh_cont = compute_swa_enhancement(swa_cont, swa_base)
    plv_cont = m_c.compute_plv(1.0)

    # Pulsed (SO phase-locked)
    m_p = WCThalamocorticalEnsemble(seed=42)
    m_p.initialize_from_baseline(bp_init)
    for _ in range(BURN_IN_EPOCHS):
        m_p.run_epoch(30.0, 0.0, 0.0, sleep_stage='N2')
    m_p._mf_buffer[:] = 0.0
    m_p._mf_idx = 0
    m_p._step_counter = 0
    for _ in range(3):
        m_p.run_epoch(30.0, 1.0, 0.10, sleep_stage='N2', pulsed=True)
    bp_pulsed = m_p.compute_band_powers()
    swa_pulsed = compute_swa(bp_pulsed)
    enh_pulsed = compute_swa_enhancement(swa_pulsed, swa_base)
    plv_pulsed = m_p.compute_plv(1.0)

    print(f"Baseline SWA (abs): {swa_base:.6f}")
    print(f"Continuous: SWA enh={enh_cont:+.1f}%, PLV={plv_cont:.3f}")
    print(f"Pulsed:     SWA enh={enh_pulsed:+.1f}%, PLV={plv_pulsed:.3f}")
    print(f"Pulsed > Continuous: {enh_pulsed > enh_cont}")
    print(f"Target range: SWA 10-25%, PLV 0.3-0.6")

    ok = (8.0 <= enh_cont <= 30.0 and
          12.0 <= enh_pulsed <= 30.0 and
          enh_pulsed > enh_cont)
    print(f"PASS: {ok}")

    sys.exit(0 if ok else 1)
