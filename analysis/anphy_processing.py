"""
ANPHY-Sleep Dataset Processing Adapter

Processes High-Density EEG (83+ channels) from the ANPHY-Sleep dataset.
Reference: Wei et al., "ANPHY-Sleep: Open Sleep Database from Healthy Adults
           using High-Density Scalp Electroencephalogram"

Key differences from Sleep-EDF:
- 93 channels (HD-EEG + EOG + ECG) at 1000 Hz
- Sleep annotations in TXT format (tab-separated)
- Standard 10-20 channel names (Fp1, Fp2, F3, F4, C3, C4, etc.)
"""

import mne
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import warnings
import re

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')


def parse_anphy_annotations(txt_path: str, epoch_duration: float = 30.0) -> pd.DataFrame:
    """
    Parse ANPHY-Sleep TXT annotation file.

    Format: Records separated by carriage return (CR / \\r)
    Each record: Stage<TAB>Start_Time<TAB>Duration
    Stages: L (lights), W (wake), N1, N2, N3, R (REM)

    Args:
        txt_path: Path to annotation TXT file
        epoch_duration: Duration of each epoch in seconds (default 30)

    Returns:
        DataFrame with 'sleep_stage' column
    """
    stage_mapping = {
        'L': 'Wake',   # Lights on/off - treat as wake
        'W': 'Wake',
        'N1': 'N1',
        'N2': 'N2',
        'N3': 'N3',
        'R': 'REM',
    }

    with open(txt_path, 'rb') as f:
        content = f.read()

    # Split by carriage return (0x0d) to get individual records
    records = content.split(b'\r')

    stages_list = []
    for record in records:
        record = record.strip()
        if not record:
            continue

        # Split by tab to get [stage, start_time, duration]
        parts = record.split(b'\t')
        if len(parts) >= 3:
            try:
                stage = parts[0].decode('utf-8').strip()
                duration = float(parts[2].decode('utf-8'))

                # Map to standard stage names
                mapped_stage = stage_mapping.get(stage, 'Wake')

                # Calculate number of epochs for this annotation
                n_epochs = int(duration / epoch_duration)
                for _ in range(n_epochs):
                    stages_list.append(mapped_stage)

            except (ValueError, UnicodeDecodeError):
                continue

    return pd.DataFrame({'sleep_stage': stages_list})


class ANPHYProcessor:
    """Process ANPHY-Sleep HD-EEG recordings."""

    # Preferred channels for sleep analysis (in order of preference)
    PREFERRED_CHANNELS = ['C3', 'C4', 'CZ', 'FZ', 'F3', 'F4', 'Fp1', 'Fp2']

    def __init__(self, edf_path: str, target_sfreq: float = 100.0):
        """
        Initialize processor for ANPHY EDF file.

        Args:
            edf_path: Path to EDF file
            target_sfreq: Target sampling frequency for downsampling (default 100 Hz)
        """
        self.edf_path = edf_path
        self.target_sfreq = target_sfreq

        # Load raw data (memory-mapped, not fully loaded yet)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        self.sfreq = self.raw.info['sfreq']
        self.ch_names = self.raw.ch_names

    def get_eeg_channels(self) -> List[str]:
        """Get list of EEG channels (excluding EOG, ECG, EMG)."""
        exclude_patterns = ['EOG', 'ECG', 'EMG', 'EKG', 'CHIN', 'LEG']
        eeg_channels = []
        for ch in self.ch_names:
            if not any(pattern in ch.upper() for pattern in exclude_patterns):
                eeg_channels.append(ch)
        return eeg_channels

    def select_channel(self) -> str:
        """Select best channel for sleep analysis."""
        eeg_channels = self.get_eeg_channels()

        # Try preferred channels in order
        for pref in self.PREFERRED_CHANNELS:
            if pref in eeg_channels:
                return pref

        # Fall back to first EEG channel
        if eeg_channels:
            return eeg_channels[0]

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

        # Load and pick channel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)

        raw.pick_channels([channel])

        # Downsample if needed (ANPHY is 1000 Hz, quite high)
        if raw.info['sfreq'] > self.target_sfreq:
            raw.resample(self.target_sfreq)

        # Create fixed-length epochs
        epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration,
                                               preload=True, verbose=False)

        # Compute power spectral density
        psds = epochs.compute_psd(method='welch', fmin=0.5, fmax=30,
                                  n_fft=int(self.target_sfreq * 4),
                                  verbose=False)
        freqs = psds.freqs

        band_powers = []
        for idx in range(len(psds)):
            psd_data = psds.get_data()[idx, 0, :]

            # Calculate band powers
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
        """Calculate power in frequency band using trapezoidal integration."""
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if np.sum(idx) < 2:
            return 0.0

        psd_band = psd[idx]
        freqs_band = freqs[idx]

        return np.sum((psd_band[1:] + psd_band[:-1]) / 2.0 * np.diff(freqs_band))


def process_anphy_subject(subject_dir: Path, output_dir: Path) -> Optional[Path]:
    """
    Process a single ANPHY subject.

    Args:
        subject_dir: Path to subject directory (e.g., data/raw/anphy/EPCTL02/)
        output_dir: Path to output directory

    Returns:
        Path to output CSV file, or None if processing failed
    """
    subject_dir = Path(subject_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find EDF and annotation files
    edf_files = list(subject_dir.glob('*.edf'))
    txt_files = list(subject_dir.glob('*.txt'))

    if not edf_files:
        print(f"  No EDF file found in {subject_dir}")
        return None

    if not txt_files:
        print(f"  No annotation TXT file found in {subject_dir}")
        return None

    edf_path = edf_files[0]
    txt_path = txt_files[0]
    subject_id = subject_dir.name

    print(f"Processing {subject_id}...")
    print(f"  EDF: {edf_path.name}")
    print(f"  Annotations: {txt_path.name}")

    try:
        # Extract band powers
        processor = ANPHYProcessor(str(edf_path))
        band_powers = processor.extract_band_powers()

        # Load sleep stages
        stages = parse_anphy_annotations(str(txt_path))

        # Combine (align lengths)
        min_len = min(len(band_powers), len(stages))
        df = band_powers.iloc[:min_len].copy()
        df['sleep_stage'] = stages['sleep_stage'].iloc[:min_len].values

        # Save
        output_file = output_dir / f'{subject_id}_processed.csv'
        df.to_csv(output_file, index=False)

        print(f"  Saved: {output_file.name} ({len(df)} epochs)")

        # Print stage distribution
        stage_counts = df['sleep_stage'].value_counts()
        print(f"  Stage distribution: {dict(stage_counts)}")

        return output_file

    except Exception as e:
        print(f"  Error processing {subject_id}: {e}")
        return None


def process_all_anphy(anphy_dir: Path, output_dir: Path):
    """
    Process all ANPHY subjects.

    Args:
        anphy_dir: Path to ANPHY data directory (containing subject folders)
        output_dir: Path to output directory
    """
    anphy_dir = Path(anphy_dir)
    output_dir = Path(output_dir)

    # Find all subject directories (EPCTL*)
    subject_dirs = sorted([d for d in anphy_dir.iterdir()
                          if d.is_dir() and d.name.startswith('EPCTL')])

    if not subject_dirs:
        print(f"No subject directories found in {anphy_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Processing {len(subject_dirs)} ANPHY subjects")
    print(f"{'='*70}\n")

    processed = 0
    failed = 0

    for subject_dir in subject_dirs:
        result = process_anphy_subject(subject_dir, output_dir)
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
        anphy_dir = Path(sys.argv[1])
    else:
        anphy_dir = Path('data/raw/anphy')

    output_dir = Path('data/processed')

    process_all_anphy(anphy_dir, output_dir)
