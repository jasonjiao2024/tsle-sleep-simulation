"""
HMC Sleep Staging Database Processing Adapter

Processes the HMC Sleep Staging Database for sleep analysis validation.
Reference: Haaglanden Medisch Centrum, The Netherlands

Key characteristics:
- 151 subjects (we process a subset)
- 4 EEG channels (F4-M1, C4-M1, O2-M1, C3-M2) at 256 Hz
- Sleep annotations in EDF+ format (30-second epochs)
- AASM scoring: W=Wake, N1, N2, N3, R=REM
"""

import mne
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings
from collections import Counter

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')


def parse_hmc_annotations(scoring_path: str) -> pd.DataFrame:
    """
    Parse HMC sleep scoring EDF+ file.

    HMC uses AASM scoring with 30-second epochs:
    - Sleep stage W = Wake
    - Sleep stage N1 = N1
    - Sleep stage N2 = N2
    - Sleep stage N3 = N3
    - Sleep stage R = REM

    Args:
        scoring_path: Path to sleepscoring EDF+ file

    Returns:
        DataFrame with 'sleep_stage' column (30-second epochs)
    """
    stage_mapping = {
        'Sleep stage W': 'Wake',
        'Sleep stage N1': 'N1',
        'Sleep stage N2': 'N2',
        'Sleep stage N3': 'N3',
        'Sleep stage R': 'REM',
    }

    annot = mne.read_annotations(scoring_path)

    stages = []
    for desc in annot.description:
        if desc in stage_mapping:
            stages.append(stage_mapping[desc])

    return pd.DataFrame({'sleep_stage': stages})


class HMCProcessor:
    """Process HMC PSG recordings."""

    # EEG channels in HMC dataset
    EEG_CHANNELS = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']
    PREFERRED_CHANNEL = 'EEG C4-M1'  # Central channel

    def __init__(self, edf_path: str, target_sfreq: float = 100.0):
        """
        Initialize processor for HMC EDF file.

        Args:
            edf_path: Path to EDF file
            target_sfreq: Target sampling frequency for processing
        """
        self.edf_path = edf_path
        self.target_sfreq = target_sfreq

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        self.sfreq = self.raw.info['sfreq']
        self.ch_names = self.raw.ch_names

    def select_channel(self) -> str:
        """Select best EEG channel for sleep analysis."""
        if self.PREFERRED_CHANNEL in self.ch_names:
            return self.PREFERRED_CHANNEL

        for ch in self.EEG_CHANNELS:
            if ch in self.ch_names:
                return ch

        # Fallback: find any EEG channel
        for ch in self.ch_names:
            if 'EEG' in ch.upper():
                return ch

        raise ValueError(f"No suitable EEG channel found. Available: {self.ch_names}")

    def extract_band_powers(self, channel: Optional[str] = None,
                           epoch_duration: float = 30.0,
                           normalize: bool = True) -> pd.DataFrame:
        """
        Extract EEG band powers for each epoch.

        Args:
            channel: EEG channel to use (auto-selected if None)
            epoch_duration: Duration of each epoch in seconds
            normalize: If True, return relative band powers (sum to 1.0)

        Returns:
            DataFrame with delta, theta, alpha, beta power columns
        """
        if channel is None:
            channel = self.select_channel()

        print(f"  Using channel: {channel}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)

        raw.pick_channels([channel])

        # Resample if needed (HMC is at 256 Hz)
        if raw.info['sfreq'] > self.target_sfreq:
            raw.resample(self.target_sfreq)

        # Create fixed-length epochs
        epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration,
                                               preload=True, verbose=False)

        # Compute PSD
        psds = epochs.compute_psd(method='welch', fmin=0.5, fmax=30,
                                  n_fft=int(self.target_sfreq * 4),
                                  verbose=False)
        freqs = psds.freqs

        band_powers = []
        for idx in range(len(psds)):
            psd_data = psds.get_data()[idx, 0, :]

            delta = self._band_power(psd_data, freqs, 0.5, 4.0)
            theta = self._band_power(psd_data, freqs, 4.0, 8.0)
            alpha = self._band_power(psd_data, freqs, 8.0, 13.0)
            beta = self._band_power(psd_data, freqs, 13.0, 30.0)

            if normalize:
                total = delta + theta + alpha + beta
                if total > 0:
                    delta /= total
                    theta /= total
                    alpha /= total
                    beta /= total

            band_powers.append({
                'delta_power': delta,
                'theta_power': theta,
                'alpha_power': alpha,
                'beta_power': beta,
            })

        return pd.DataFrame(band_powers)

    @staticmethod
    def _band_power(psd: np.ndarray, freqs: np.ndarray,
                    fmin: float, fmax: float) -> float:
        """Calculate power in frequency band."""
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if np.sum(idx) < 2:
            return 0.0
        psd_band = psd[idx]
        freqs_band = freqs[idx]
        return np.sum((psd_band[1:] + psd_band[:-1]) / 2.0 * np.diff(freqs_band))


def process_hmc_subject(edf_path: Path, scoring_path: Path,
                        output_dir: Path) -> Optional[Path]:
    """
    Process a single HMC subject.

    Args:
        edf_path: Path to EDF file
        scoring_path: Path to sleepscoring EDF+ file
        output_dir: Output directory

    Returns:
        Path to output CSV file, or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = edf_path.stem  # e.g., "SN001"

    print(f"Processing {subject_id}...")
    print(f"  EDF: {edf_path.name}")
    print(f"  Scoring: {scoring_path.name}")

    try:
        processor = HMCProcessor(str(edf_path))
        band_powers = processor.extract_band_powers()

        stages = parse_hmc_annotations(str(scoring_path))

        print(f"  Band power epochs: {len(band_powers)}, Annotation epochs: {len(stages)}")

        # Align lengths
        min_len = min(len(band_powers), len(stages))
        if min_len == 0:
            print(f"  Error: No epochs to process")
            return None

        df = band_powers.iloc[:min_len].copy()
        df['sleep_stage'] = stages['sleep_stage'].iloc[:min_len].values

        output_file = output_dir / f'HMC_{subject_id}_processed.csv'
        df.to_csv(output_file, index=False)

        print(f"  Saved: {output_file.name} ({len(df)} epochs)")
        print(f"  Stage distribution: {dict(df['sleep_stage'].value_counts())}")

        return output_file

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_all_hmc(hmc_dir: Path, output_dir: Path, max_subjects: int = None):
    """
    Process all HMC subjects.

    Args:
        hmc_dir: Path to HMC data directory
        output_dir: Output directory
        max_subjects: Maximum number of subjects to process (None = all)
    """
    hmc_dir = Path(hmc_dir)
    output_dir = Path(output_dir)

    # Find all subject EDF files with matching scoring files
    edf_files = sorted(hmc_dir.glob('SN*.edf'))
    edf_files = [f for f in edf_files if not f.name.endswith('_sleepscoring.edf')]

    valid_subjects = []
    for edf_path in edf_files:
        scoring_path = hmc_dir / f'{edf_path.stem}_sleepscoring.edf'
        if scoring_path.exists():
            valid_subjects.append((edf_path, scoring_path))

    if max_subjects:
        valid_subjects = valid_subjects[:max_subjects]

    print(f"\n{'='*70}")
    print(f"Processing {len(valid_subjects)} HMC subjects")
    print(f"{'='*70}\n")

    processed = 0
    failed = 0

    for edf_path, scoring_path in valid_subjects:
        result = process_hmc_subject(edf_path, scoring_path, output_dir)
        if result:
            processed += 1
        else:
            failed += 1
        print()

    print(f"{'='*70}")
    print(f"Complete: {processed} processed, {failed} failed")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        hmc_dir = Path(sys.argv[1])
    else:
        hmc_dir = Path('data/raw/hmc')

    output_dir = Path('data/processed')

    process_all_hmc(hmc_dir, output_dir)
