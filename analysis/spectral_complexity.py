"""
Spectral Complexity Metrics Module

Computes EEG spectral complexity features for sleep characterization:
- Power spectral density estimation (Welch / periodogram)
- Normalized spectral entropy (standard metric; cf. Inouye et al. 1991)
- Band Dominance Index (BDI): relative band power, a standard measure
  in EEG analysis (see Niedermeyer & da Silva, "Electroencephalography")

Note: Spectral entropy and BDI are individually well-established metrics.
Their novelty in this work lies in their integration within the STDI
composite score (see transition_dynamics.py), not as standalone inventions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EEG frequency band definitions
FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)  # Extended for completeness
}


class SpectralComplexityAnalyzer:
    """
    Analyzer for EEG spectral complexity metrics.

    Provides novel biomarkers based on the complexity and distribution
    of EEG spectral power, complementing traditional band power analysis.
    """

    def __init__(self, sampling_rate: float = 256.0):
        """
        Initialize the analyzer.

        Args:
            sampling_rate: EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def compute_psd(
        self,
        eeg_signal: np.ndarray,
        method: str = 'welch',
        nperseg: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density of an EEG signal.

        Args:
            eeg_signal: 1D array of EEG samples
            method: PSD estimation method ('welch' or 'periodogram')
            nperseg: Segment length for Welch method

        Returns:
            (frequencies, psd): Arrays of frequency values and PSD estimates
        """
        if method == 'welch':
            freqs, psd = signal.welch(
                eeg_signal,
                fs=self.sampling_rate,
                nperseg=min(nperseg, len(eeg_signal)),
                noverlap=nperseg // 2
            )
        else:
            freqs, psd = signal.periodogram(
                eeg_signal,
                fs=self.sampling_rate
            )

        return freqs, psd

    def compute_spectral_entropy(
        self,
        psd: np.ndarray,
        freqs: Optional[np.ndarray] = None,
        freq_range: Tuple[float, float] = (0.5, 30.0)
    ) -> float:
        """
        Compute normalized spectral entropy (Shannon entropy of PSD).

        Note: Spectral entropy is a standard information-theoretic measure
        widely used in anesthesia monitoring (e.g., Viertiö-Oja et al. 2004).

        Interpretation:
        - Low entropy: Power concentrated in narrow frequency range (e.g., deep sleep)
        - High entropy: Power distributed across many frequencies (e.g., wake/REM)

        Args:
            psd: Power spectral density values
            freqs: Corresponding frequency values (optional)
            freq_range: Frequency range to consider

        Returns:
            Normalized spectral entropy in range [0, 1]
        """
        # Apply frequency range filter if freqs provided
        if freqs is not None:
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            psd = psd[mask]

        if len(psd) == 0 or np.sum(psd) == 0:
            return 0.0

        # Normalize PSD to probability distribution
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros

        # Shannon entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(psd_norm))

        if max_entropy == 0:
            return 0.0

        return entropy / max_entropy

    def compute_band_dominance_index(
        self,
        band_powers: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute Band Dominance Index for each frequency band.

        Note: BDI is equivalent to relative band power, a standard EEG
        measure. Its value here is as a component of the STDI composite.

        Definition: BDI_band = band_power / sum(all_band_powers)

        This normalized metric shows which frequency band dominates
        the EEG spectrum at any given moment.

        Args:
            band_powers: Dictionary of band powers {'delta': x, 'theta': y, ...}

        Returns:
            Dictionary of BDI values for each band
        """
        total_power = sum(band_powers.values())

        if total_power == 0:
            return {band: 0.0 for band in band_powers}

        return {
            band: power / total_power
            for band, power in band_powers.items()
        }

    def compute_delta_dominance_ratio(
        self,
        band_powers: Dict[str, float]
    ) -> float:
        """
        Compute delta dominance ratio - a marker of deep sleep.

        Definition: DDR = delta_power / (alpha_power + beta_power)

        Higher values indicate deeper sleep states.

        Args:
            band_powers: Dictionary of band powers

        Returns:
            Delta dominance ratio
        """
        delta = band_powers.get('delta_power', band_powers.get('delta', 0))
        alpha = band_powers.get('alpha_power', band_powers.get('alpha', 0))
        beta = band_powers.get('beta_power', band_powers.get('beta', 0))

        denominator = alpha + beta
        if denominator == 0:
            return 0.0

        return delta / denominator

    def compute_slow_fast_ratio(
        self,
        band_powers: Dict[str, float]
    ) -> float:
        """
        Compute slow-to-fast wave ratio.

        Definition: SFR = (delta + theta) / (alpha + beta)

        Similar to sleep depth index but explicitly named for clarity.

        Args:
            band_powers: Dictionary of band powers

        Returns:
            Slow/fast ratio
        """
        # Handle both naming conventions
        delta = band_powers.get('delta_power', band_powers.get('delta', 0))
        theta = band_powers.get('theta_power', band_powers.get('theta', 0))
        alpha = band_powers.get('alpha_power', band_powers.get('alpha', 0))
        beta = band_powers.get('beta_power', band_powers.get('beta', 0))

        fast = alpha + beta
        if fast == 0:
            return 0.0

        return (delta + theta) / fast

    def compute_spectral_edge_frequency(
        self,
        psd: np.ndarray,
        freqs: np.ndarray,
        percentile: float = 95.0
    ) -> float:
        """
        Compute spectral edge frequency (SEF).

        SEF is the frequency below which a certain percentage of total
        power is contained. Used in anesthesia monitoring.

        Args:
            psd: Power spectral density
            freqs: Corresponding frequencies
            percentile: Cumulative power percentile (default 95%)

        Returns:
            Spectral edge frequency in Hz
        """
        if len(psd) == 0:
            return 0.0

        # Cumulative power
        cumsum = np.cumsum(psd)
        total = cumsum[-1]

        if total == 0:
            return 0.0

        # Find frequency at percentile
        threshold = total * (percentile / 100.0)
        idx = np.searchsorted(cumsum, threshold)

        if idx >= len(freqs):
            idx = len(freqs) - 1

        return freqs[idx]

    def compute_peak_frequency(
        self,
        psd: np.ndarray,
        freqs: np.ndarray,
        freq_range: Tuple[float, float] = (0.5, 30.0)
    ) -> float:
        """
        Compute peak (dominant) frequency in the spectrum.

        Args:
            psd: Power spectral density
            freqs: Corresponding frequencies
            freq_range: Frequency range to search

        Returns:
            Peak frequency in Hz
        """
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        psd_masked = psd[mask]
        freqs_masked = freqs[mask]

        if len(psd_masked) == 0:
            return 0.0

        peak_idx = np.argmax(psd_masked)
        return freqs_masked[peak_idx]

    def compute_complexity_ratio(
        self,
        current_entropy: float,
        baseline_entropy: float
    ) -> float:
        """
        Compute complexity ratio relative to wake baseline.

        Definition: CR = current_entropy / baseline_entropy

        Interpretation:
        - CR < 1: Less complex than wake (deeper sleep)
        - CR = 1: Similar to wake
        - CR > 1: More complex than wake (unusual)

        Args:
            current_entropy: Entropy of current epoch
            baseline_entropy: Entropy of wake baseline

        Returns:
            Complexity ratio
        """
        if baseline_entropy == 0:
            return 1.0

        return current_entropy / baseline_entropy

    def compute_complexity_profile(
        self,
        band_powers_series: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute spectral complexity profile for a series of epochs.

        Args:
            band_powers_series: DataFrame with columns for each band power

        Returns:
            DataFrame with complexity metrics for each epoch
        """
        results = []

        for idx, row in band_powers_series.iterrows():
            # Extract band powers from row
            band_powers = {}
            for col in row.index:
                if 'power' in col:
                    band = col.replace('_power', '')
                    band_powers[band] = row[col]

            # Compute metrics
            bdi = self.compute_band_dominance_index(band_powers)
            sfr = self.compute_slow_fast_ratio(band_powers)
            ddr = self.compute_delta_dominance_ratio(band_powers)

            result = {
                'epoch': idx,
                'slow_fast_ratio': sfr,
                'delta_dominance_ratio': ddr,
                **{f'bdi_{band}': val for band, val in bdi.items()}
            }
            results.append(result)

        return pd.DataFrame(results)

    def classify_by_complexity(
        self,
        spectral_entropy: float,
        delta_dominance: float
    ) -> str:
        """
        Classify sleep state based on complexity features.

        This provides an alternative classification approach based on
        spectral complexity rather than traditional band power ratios.

        Args:
            spectral_entropy: Normalized spectral entropy [0, 1]
            delta_dominance: Delta dominance ratio

        Returns:
            Inferred sleep stage
        """
        # Decision rules based on complexity features
        if spectral_entropy < 0.4 and delta_dominance > 2.0:
            return 'N3'  # Low complexity, high delta = deep sleep
        elif spectral_entropy < 0.6 and delta_dominance > 1.0:
            return 'N2'  # Moderate complexity
        elif spectral_entropy < 0.7 and delta_dominance > 0.5:
            return 'N1'  # Transitional
        elif spectral_entropy > 0.7 and delta_dominance < 0.5:
            return 'REM'  # High complexity, low delta (desynchronized)
        else:
            return 'Wake'  # High complexity, variable

    def compute_all_complexity_metrics(
        self,
        band_powers: Dict[str, float],
        psd: Optional[np.ndarray] = None,
        freqs: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute all spectral complexity metrics for a single epoch.

        Args:
            band_powers: Dictionary of band powers
            psd: Optional raw PSD array
            freqs: Optional frequency array

        Returns:
            Dictionary with all complexity metrics
        """
        # Band dominance indices
        bdi = self.compute_band_dominance_index(band_powers)

        # Ratios
        sfr = self.compute_slow_fast_ratio(band_powers)
        ddr = self.compute_delta_dominance_ratio(band_powers)

        result = {
            'slow_fast_ratio': sfr,
            'delta_dominance_ratio': ddr,
            'band_dominance': bdi
        }

        # If raw PSD provided, compute additional metrics
        if psd is not None and freqs is not None:
            result['spectral_entropy'] = self.compute_spectral_entropy(psd, freqs)
            result['spectral_edge_95'] = self.compute_spectral_edge_frequency(psd, freqs, 95)
            result['peak_frequency'] = self.compute_peak_frequency(psd, freqs)

            # Complexity-based classification
            result['complexity_stage'] = self.classify_by_complexity(
                result['spectral_entropy'],
                ddr
            )

        return result


def compute_epoch_complexity(
    band_powers: Dict[str, float],
    psd: Optional[np.ndarray] = None,
    freqs: Optional[np.ndarray] = None
) -> Dict:
    """
    Convenience function to compute complexity metrics for a single epoch.

    Args:
        band_powers: Band powers for the epoch
        psd: Optional PSD array
        freqs: Optional frequency array

    Returns:
        Dictionary of complexity metrics
    """
    analyzer = SpectralComplexityAnalyzer()
    return analyzer.compute_all_complexity_metrics(band_powers, psd, freqs)


if __name__ == '__main__':
    # Demo with synthetic data
    print("=" * 60)
    print("SPECTRAL COMPLEXITY METRICS DEMO")
    print("=" * 60)

    analyzer = SpectralComplexityAnalyzer()

    # Simulate different sleep states
    sleep_states = {
        'Wake': {'delta': 0.15, 'theta': 0.15, 'alpha': 0.40, 'beta': 0.30},
        'N1': {'delta': 0.25, 'theta': 0.30, 'alpha': 0.25, 'beta': 0.20},
        'N2': {'delta': 0.35, 'theta': 0.35, 'alpha': 0.15, 'beta': 0.15},
        'N3': {'delta': 0.55, 'theta': 0.30, 'alpha': 0.08, 'beta': 0.07},
        'REM': {'delta': 0.20, 'theta': 0.25, 'alpha': 0.25, 'beta': 0.30}
    }

    print("\nSpectral Complexity by Sleep State:")
    print("-" * 60)
    print(f"{'State':<8} {'SFR':>8} {'DDR':>8} {'BDI_delta':>10} {'BDI_alpha':>10}")
    print("-" * 60)

    for state, powers in sleep_states.items():
        metrics = analyzer.compute_all_complexity_metrics(powers)
        bdi = metrics['band_dominance']
        print(f"{state:<8} {metrics['slow_fast_ratio']:>8.2f} "
              f"{metrics['delta_dominance_ratio']:>8.2f} "
              f"{bdi['delta']:>10.3f} {bdi['alpha']:>10.3f}")

    # Demo entropy calculation with synthetic PSD
    print("\n\nSpectral Entropy Demo:")
    print("-" * 40)

    # Create synthetic PSDs for different states
    freqs = np.linspace(0.5, 30, 100)

    # Deep sleep: concentrated power in delta
    psd_deep = np.exp(-0.5 * ((freqs - 2) / 1) ** 2)  # Peak at 2 Hz
    entropy_deep = analyzer.compute_spectral_entropy(psd_deep, freqs)

    # Wake: distributed power
    psd_wake = np.exp(-0.5 * ((freqs - 10) / 5) ** 2)  # Broader peak
    entropy_wake = analyzer.compute_spectral_entropy(psd_wake, freqs)

    print(f"Deep sleep spectral entropy: {entropy_deep:.3f}")
    print(f"Wake spectral entropy:       {entropy_wake:.3f}")
    print(f"\nAs expected, wake has higher entropy (more distributed power)")
