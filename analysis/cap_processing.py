"""
CAP Sleep Database Processing Adapter

Processes the CAP Sleep Database healthy subjects for sleep analysis validation.
Reference: Terzano et al., Ospedale Maggiore di Parma, Italy

Key characteristics:
- 16 healthy subjects (n1-n16, excluding n11 which doesn't exist)
- 21 channels at 512 Hz (EEG: C4-A1, F3-C3, C3-P3, etc.)
- Sleep annotations in TXT format (30-second epochs)
- R&K scoring: W=Wake, S1=N1, S2=N2, S3/S4=N3, R=REM
"""

import mne
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')


def parse_cap_hypnogram(txt_path: str) -> pd.DataFrame:
    """
    Parse CAP Sleep Database TXT annotation file.

    CAP uses R&K scoring in first column with 30-second epochs:
    - W = Wake
    - S1 = Stage 1 (N1)
    - S2 = Stage 2 (N2)
    - S3 = Stage 3 (N3)
    - S4 = Stage 4 (N3 in AASM)
    - R = REM

    Two format variants:
    - 6-column: Stage, Position, Time, Event, Duration, Location
    - 5-column: Stage, Time, Event, Duration, Location

    Args:
        txt_path: Path to annotation TXT file

    Returns:
        DataFrame with 'sleep_stage' column (30-second epochs)
    """
    stage_mapping = {
        'W': 'Wake',
        'S1': 'N1',
        'S2': 'N2',
        'S3': 'N3',
        'S4': 'N3',  # S3 and S4 combined as N3 in AASM
        'R': 'REM',
        'MT': 'Wake',  # Movement time treated as wake
    }

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    stages = []
    in_data = False
    event_col_idx = None  # Will be determined from header

    for line in lines:
        line = line.strip()

        # Detect header and determine format
        if line.startswith('Sleep Stage'):
            in_data = True
            header_parts = line.split('\t')
            # Find which column contains "Event"
            for i, col in enumerate(header_parts):
                if col.strip() == 'Event':
                    event_col_idx = i
                    break
            continue

        if not in_data or not line or event_col_idx is None:
            continue

        # Parse data line
        parts = line.split('\t')
        if len(parts) > event_col_idx:
            stage = parts[0].strip()
            event = parts[event_col_idx].strip()

            # Only process sleep stage events (SLEEP-S0, SLEEP-S1, etc.)
            if event.startswith('SLEEP-'):
                mapped_stage = stage_mapping.get(stage, None)
                if mapped_stage:
                    stages.append(mapped_stage)

    return pd.DataFrame({'sleep_stage': stages})


class CAPProcessor:
    """Process CAP Sleep Database PSG recordings."""

    # EEG channels in CAP dataset (bipolar montages)
    EEG_CHANNELS = ['C4-A1', 'C3-P3', 'F3-C3', 'C4-P4', 'F4-C4', 'P3-O1', 'P4-O2']
    PREFERRED_CHANNEL = 'C4-A1'  # Central channel referenced to mastoid

    def __init__(self, edf_path: str, target_sfreq: float = 100.0):
        """
        Initialize processor for CAP EDF file.

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
            ch_upper = ch.upper()
            if any(x in ch_upper for x in ['C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2']):
                # Exclude EOG and EMG channels
                if 'EOG' not in ch_upper and 'EMG' not in ch_upper and 'ECG' not in ch_upper:
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

        # Resample if needed (CAP is at 512 Hz)
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


def process_cap_subject(edf_path: Path, hypno_path: Path,
                        output_dir: Path) -> Optional[Path]:
    """
    Process a single CAP subject.

    Args:
        edf_path: Path to EDF file
        hypno_path: Path to hypnogram TXT file
        output_dir: Output directory

    Returns:
        Path to output CSV file, or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = edf_path.stem  # e.g., "n1"

    print(f"Processing {subject_id}...")
    print(f"  EDF: {edf_path.name}")
    print(f"  Hypnogram: {hypno_path.name}")

    try:
        processor = CAPProcessor(str(edf_path))
        band_powers = processor.extract_band_powers()

        stages = parse_cap_hypnogram(str(hypno_path))

        print(f"  Band power epochs: {len(band_powers)}, Annotation epochs: {len(stages)}")

        # Align lengths
        min_len = min(len(band_powers), len(stages))
        if min_len == 0:
            print(f"  Error: No epochs to process")
            return None

        df = band_powers.iloc[:min_len].copy()
        df['sleep_stage'] = stages['sleep_stage'].iloc[:min_len].values

        output_file = output_dir / f'CAP_{subject_id}_processed.csv'
        df.to_csv(output_file, index=False)

        print(f"  Saved: {output_file.name} ({len(df)} epochs)")
        print(f"  Stage distribution: {dict(df['sleep_stage'].value_counts())}")

        return output_file

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_all_cap(cap_dir: Path, output_dir: Path, max_subjects: int = None):
    """
    Process all CAP healthy subjects.

    Args:
        cap_dir: Path to CAP data directory
        output_dir: Output directory
        max_subjects: Maximum number of subjects to process (None = all)
    """
    cap_dir = Path(cap_dir)
    output_dir = Path(output_dir)

    # Find all healthy subject EDF files (n1-n16, excluding n11)
    edf_files = sorted(cap_dir.glob('n*.edf'))

    if max_subjects:
        edf_files = edf_files[:max_subjects]

    print(f"\n{'='*70}")
    print(f"Processing {len(edf_files)} CAP healthy subjects")
    print(f"{'='*70}\n")

    processed = 0
    failed = 0

    for edf_path in edf_files:
        subject_id = edf_path.stem
        hypno_path = cap_dir / f'{subject_id}.txt'

        if not hypno_path.exists():
            print(f"  Hypnogram not found for {edf_path.name}")
            failed += 1
            continue

        result = process_cap_subject(edf_path, hypno_path, output_dir)
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
        cap_dir = Path(sys.argv[1])
    else:
        cap_dir = Path('data/raw/cap')

    output_dir = Path('data/processed')

    process_all_cap(cap_dir, output_dir)
