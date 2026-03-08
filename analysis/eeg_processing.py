"""
EEG Processing Module for Sleep Entrainment Study.

Provides EDF file parsing, band power extraction, and sleep stage loading
for polysomnography recordings from multiple databases (Sleep-EDF, CAP,
DREAMS, HMC, SLPDB).

Band powers are normalized to relative values (sum to 1.0) to ensure
consistent sleep depth ratios across recording equipment and gain settings.
"""

import mne
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')


class EDFProcessor:
    def __init__(self, edf_path: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        self.sfreq = self.raw.info['sfreq']
        
    def extract_band_powers(self, channel='EEG Fpz-Cz', epoch_duration=30, normalize=True):
        """
        Extract EEG band powers for each epoch.

        Args:
            channel: EEG channel name to use
            epoch_duration: Duration of each epoch in seconds
            normalize: If True, return relative band powers (sum to 1.0)
                      This ensures consistent sleep_depth ratios across different
                      recording equipment and gain settings.
        """
        if channel not in self.raw.ch_names:
            available_channels = [ch for ch in self.raw.ch_names if 'EEG' in ch]
            if available_channels:
                channel = available_channels[0]
            else:
                raise ValueError(f"EEG channel not found. Available: {self.raw.ch_names}")

        raw_eeg = self.raw.copy().pick_channels([channel])

        epochs = mne.make_fixed_length_epochs(raw_eeg, duration=epoch_duration)

        psds = epochs.compute_psd(method='welch', fmin=0.5, fmax=30, n_fft=2048)
        freqs = psds.freqs

        band_powers = []
        for idx in range(len(psds)):
            psd_data = psds.get_data()[idx, 0, :]

            # Calculate raw band powers
            delta = self._band_power(psd_data, freqs, 0.5, 4.0)
            theta = self._band_power(psd_data, freqs, 4.0, 8.0)
            alpha = self._band_power(psd_data, freqs, 8.0, 13.0)
            beta = self._band_power(psd_data, freqs, 13.0, 30.0)

            if normalize:
                # Normalize to relative band powers (sum to 1.0)
                # This makes sleep_depth ratios comparable across different
                # recording equipment, gain settings, and data sources
                total = delta + theta + alpha + beta
                if total > 0:
                    delta = delta / total
                    theta = theta / total
                    alpha = alpha / total
                    beta = beta / total

            powers = {
                'delta_power': delta,
                'theta_power': theta,
                'alpha_power': alpha,
                'beta_power': beta,
            }
            band_powers.append(powers)

        return pd.DataFrame(band_powers)
    
    @staticmethod
    def _band_power(psd, freqs, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if len(psd[idx]) == 0:
            return 0.0
        
        psd_band = psd[idx]
        freqs_band = freqs[idx]
        
        if len(psd_band) < 2:
            return 0.0
        
        return np.sum((psd_band[1:] + psd_band[:-1]) / 2.0 * np.diff(freqs_band))


def load_sleep_stages(hypno_file: str, epoch_duration: float = 30.0) -> pd.DataFrame:
    """
    Load sleep stages from Sleep-EDF hypnogram annotation files.

    The Sleep-EDF dataset uses EDF+ annotation format with descriptions like:
    - 'Sleep stage W' = Wake
    - 'Sleep stage 1' = N1
    - 'Sleep stage 2' = N2
    - 'Sleep stage 3' or 'Sleep stage 4' = N3
    - 'Sleep stage R' = REM
    - 'Sleep stage ?' = Unknown/Movement
    """
    try:
        # Read annotations from EDF+ file
        annots = mne.read_annotations(hypno_file)

        # Map Sleep-EDF descriptions to standard stage names
        stage_mapping = {
            'Sleep stage W': 'Wake',
            'Sleep stage 1': 'N1',
            'Sleep stage 2': 'N2',
            'Sleep stage 3': 'N3',
            'Sleep stage 4': 'N3',  # Old N4 = N3 in AASM
            'Sleep stage R': 'REM',
            'Sleep stage ?': 'Wake',  # Movement/unknown as wake
        }

        # Find total recording duration
        total_duration = max(annots.onset + annots.duration)
        n_epochs = int(total_duration / epoch_duration)

        # Initialize all epochs as Unknown
        stages = ['Unknown'] * n_epochs

        # Fill in stages from annotations
        for onset, duration, desc in zip(annots.onset, annots.duration, annots.description):
            stage = stage_mapping.get(str(desc), 'Unknown')
            start_epoch = int(onset / epoch_duration)
            end_epoch = int((onset + duration) / epoch_duration)

            for epoch in range(start_epoch, min(end_epoch, n_epochs)):
                stages[epoch] = stage

        return pd.DataFrame({'sleep_stage': stages})
    except Exception as e:
        print(f"Warning: Could not load hypnogram from {hypno_file}: {e}")
        return pd.DataFrame()


def process_all_recordings(data_dir: Path, output_dir: Path):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    edf_files = list(data_dir.glob('*PSG.edf')) + list(data_dir.glob('*psg.edf'))
    
    if not edf_files:
        print(f"No PSG EDF files found in {data_dir}")
        return
    
    total_files = len(edf_files)
    print(f"\n{'='*70}")
    print(f"Processing {total_files} EEG recordings")
    print(f"{'='*70}\n")
    
    import time
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    for idx, edf_file in enumerate(edf_files, 1):
        try:
            progress_pct = (idx / total_files) * 100
            elapsed = time.time() - start_time
            if idx > 1:
                avg_time = elapsed / (idx - 1)
                remaining = avg_time * (total_files - idx)
                eta_str = f" | ETA: {remaining/60:.1f} min"
            else:
                eta_str = ""
            
            print(f"[{idx}/{total_files}] ({progress_pct:.1f}%) Processing {edf_file.name}{eta_str}")
            processor = EDFProcessor(str(edf_file))
            band_powers = processor.extract_band_powers()
            
            # Find corresponding hypnogram file
            # Sleep-EDF naming: SC4001E0-PSG.edf -> SC4001E*-Hypnogram.edf
            # The suffix letter varies (C, H, J, P, etc.)
            psg_stem = edf_file.stem.replace('-PSG', '')
            hypno_pattern = edf_file.parent / f"{psg_stem[:-1]}*-Hypnogram.edf"
            hypno_candidates = list(edf_file.parent.glob(f"{psg_stem[:-1]}*-Hypnogram.edf"))

            if hypno_candidates:
                hypno_file = str(hypno_candidates[0])
                stages = load_sleep_stages(hypno_file)
            else:
                stages = pd.DataFrame()
            
            if len(stages) > 0 and len(stages) == len(band_powers):
                df = pd.concat([band_powers, stages], axis=1)
            else:
                df = band_powers.copy()
                if len(stages) > 0:
                    min_len = min(len(df), len(stages))
                    df['sleep_stage'] = stages['sleep_stage'].iloc[:min_len].values
            
            output_file = output_dir / f'{edf_file.stem}_processed.csv'
            df.to_csv(output_file, index=False)
            processed_count += 1
            print(f"  Saved ({len(df)} epochs)")
        except Exception as e:
            error_count += 1
            print(f"  Error: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"  Processed: {processed_count}/{total_files}")
    print(f"  Errors: {error_count}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path('data/raw/sleep-edfx/1.0.0/sleep-cassette')
    
    output_dir = Path('data/processed')
    process_all_recordings(data_dir, output_dir)
