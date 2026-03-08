"""
DREAMS Dataset Processing Adapter

Processes the DREAMS Subjects database for sleep analysis validation.
Reference: Devuyst et al., University of MONS - TCTS Laboratory

Key characteristics:
- 20 healthy subjects
- 23 channels at 200 Hz (3 EEG: CZ-A1, FP1-A2, O1-A2)
- Sleep annotations in TXT format (one stage per line, 5-second epochs)
- AASM scoring: 1=N1, 2=N2, 3=N3, 4=REM, 5=Wake
"""

import mne
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')


def parse_dreams_hypnogram(txt_path: str, source_epoch_sec: float = 5.0,
                           target_epoch_sec: float = 30.0) -> pd.DataFrame:
    """
    Parse DREAMS hypnogram TXT file.

    DREAMS uses 5-second epochs with AASM scoring:
    - 1 = N1
    - 2 = N2
    - 3 = N3
    - 4 = REM
    - 5 = Wake

    Args:
        txt_path: Path to hypnogram TXT file
        source_epoch_sec: Original epoch duration (5 seconds for DREAMS)
        target_epoch_sec: Target epoch duration for output (30 seconds)

    Returns:
        DataFrame with 'sleep_stage' column (30-second epochs)
    """
    stage_mapping = {
        '1': 'N1',
        '2': 'N2',
        '3': 'N3',
        '4': 'REM',
        '5': 'Wake',
        '0': 'Wake',  # Some files may use 0 for wake
    }

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Skip header line if present
    stages_5sec = []
    for line in lines:
        line = line.strip()
        if line.startswith('[') or not line:
            continue
        if line in stage_mapping:
            stages_5sec.append(stage_mapping[line])

    # Convert 5-second epochs to 30-second epochs (majority vote)
    epochs_per_30sec = int(target_epoch_sec / source_epoch_sec)
    stages_30sec = []

    for i in range(0, len(stages_5sec), epochs_per_30sec):
        chunk = stages_5sec[i:i + epochs_per_30sec]
        if len(chunk) >= epochs_per_30sec // 2:
            # Majority vote for the 30-second epoch
            from collections import Counter
            most_common = Counter(chunk).most_common(1)[0][0]
            stages_30sec.append(most_common)

    return pd.DataFrame({'sleep_stage': stages_30sec})


class DREAMSProcessor:
    """Process DREAMS PSG recordings."""

    # EEG channels in DREAMS dataset
    EEG_CHANNELS = ['CZ-A1', 'FP1-A2', 'O1-A2', 'FP2-A1', 'O2-A1', 'CZ2-A1']
    PREFERRED_CHANNEL = 'CZ-A1'  # Central channel, similar to Sleep-EDF

    def __init__(self, edf_path: str, target_sfreq: float = 100.0):
        """
        Initialize processor for DREAMS EDF file.

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
        # Try preferred channel first
        if self.PREFERRED_CHANNEL in self.ch_names:
            return self.PREFERRED_CHANNEL

        # Try other EEG channels
        for ch in self.EEG_CHANNELS:
            if ch in self.ch_names:
                return ch

        # Fallback: find any channel with EEG-like name
        for ch in self.ch_names:
            if any(x in ch.upper() for x in ['CZ', 'FP', 'O1', 'O2', 'C3', 'C4']):
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

        # Resample if needed
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


def process_dreams_subject(edf_path: Path, hypno_path: Path,
                           output_dir: Path) -> Optional[Path]:
    """
    Process a single DREAMS subject.

    Args:
        edf_path: Path to EDF file
        hypno_path: Path to hypnogram TXT file
        output_dir: Output directory

    Returns:
        Path to output CSV file, or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = edf_path.stem  # e.g., "subject1"

    print(f"Processing {subject_id}...")
    print(f"  EDF: {edf_path.name}")
    print(f"  Hypnogram: {hypno_path.name}")

    try:
        processor = DREAMSProcessor(str(edf_path))
        band_powers = processor.extract_band_powers()

        stages = parse_dreams_hypnogram(str(hypno_path))

        # Align lengths
        min_len = min(len(band_powers), len(stages))
        df = band_powers.iloc[:min_len].copy()
        df['sleep_stage'] = stages['sleep_stage'].iloc[:min_len].values

        output_file = output_dir / f'DREAMS_{subject_id}_processed.csv'
        df.to_csv(output_file, index=False)

        print(f"  Saved: {output_file.name} ({len(df)} epochs)")
        print(f"  Stage distribution: {dict(df['sleep_stage'].value_counts())}")

        return output_file

    except Exception as e:
        print(f"  Error: {e}")
        return None


def process_all_dreams(dreams_dir: Path, output_dir: Path, max_subjects: int = None):
    """
    Process all DREAMS subjects.

    Args:
        dreams_dir: Path to DREAMS DatabaseSubjects directory
        output_dir: Output directory
        max_subjects: Maximum number of subjects to process (None = all)
    """
    dreams_dir = Path(dreams_dir)
    output_dir = Path(output_dir)

    # Find all subject EDF files
    edf_files = sorted(dreams_dir.glob('subject*.edf'))

    if max_subjects:
        edf_files = edf_files[:max_subjects]

    print(f"\n{'='*70}")
    print(f"Processing {len(edf_files)} DREAMS subjects")
    print(f"{'='*70}\n")

    processed = 0
    failed = 0

    for edf_path in edf_files:
        subject_num = edf_path.stem.replace('subject', '')
        hypno_path = dreams_dir / f'HypnogramAASM_subject{subject_num}.txt'

        if not hypno_path.exists():
            print(f"  Hypnogram not found for {edf_path.name}")
            failed += 1
            continue

        result = process_dreams_subject(edf_path, hypno_path, output_dir)
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
        dreams_dir = Path(sys.argv[1])
    else:
        dreams_dir = Path('data/raw/dreams/DatabaseSubjects')

    output_dir = Path('data/processed')

    # Process first 5 subjects for quick validation
    process_all_dreams(dreams_dir, output_dir, max_subjects=5)
