"""
MIT-BIH Polysomnographic Database (slpdb) Processing Adapter

Converts slpdb WFDB records into per-epoch band powers + sleep stage labels
compatible with the in-silico pipeline.

Reference: https://physionet.org/content/slpdb/1.0.0/
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import signal


STAGE_MAP = {
    "W": "Wake",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",  # legacy N4 -> N3
    "R": "REM",
}

EEG_HINTS = ("EEG", "C3", "C4", "O1", "O2", "CZ", "FP")


def _require_wfdb():
    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "wfdb is required for slpdb processing. Install with `pip install wfdb`."
        ) from exc
    return wfdb


def list_slpdb_records(raw_dir: Path) -> List[str]:
    raw_dir = Path(raw_dir)
    records = sorted({p.stem for p in raw_dir.glob("*.hea")})
    # Filter out non-record headers if any
    records = [r for r in records if r and not r.startswith(".")]
    return records


def select_eeg_channel(sig_names: Iterable[str]) -> int:
    names = list(sig_names)
    for idx, name in enumerate(names):
        if "EEG" in name.upper():
            return idx
    for idx, name in enumerate(names):
        upper = name.upper()
        if any(hint in upper for hint in EEG_HINTS):
            return idx
    return 0


def _band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if np.sum(idx) < 2:
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))


def extract_band_powers(
    signal_data: np.ndarray,
    sfreq: float,
    epoch_duration: float = 30.0,
    normalize: bool = True,
    n_fft: int = 2048,
) -> pd.DataFrame:
    samples_per_epoch = int(sfreq * epoch_duration)
    n_epochs = len(signal_data) // samples_per_epoch
    if n_epochs <= 0:
        return pd.DataFrame()

    band_rows = []
    for epoch_idx in range(n_epochs):
        start = epoch_idx * samples_per_epoch
        end = start + samples_per_epoch
        epoch = signal_data[start:end]

        nperseg = min(n_fft, len(epoch))
        noverlap = nperseg // 2
        freqs, psd = signal.welch(
            epoch,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=max(n_fft, nperseg),
            detrend="constant",
        )

        delta = _band_power(psd, freqs, 0.5, 4.0)
        theta = _band_power(psd, freqs, 4.0, 8.0)
        alpha = _band_power(psd, freqs, 8.0, 13.0)
        beta = _band_power(psd, freqs, 13.0, 30.0)

        if normalize:
            total = delta + theta + alpha + beta
            if total > 0:
                delta /= total
                theta /= total
                alpha /= total
                beta /= total

        band_rows.append(
            {
                "delta_power": delta,
                "theta_power": theta,
                "alpha_power": alpha,
                "beta_power": beta,
            }
        )

    return pd.DataFrame(band_rows)


def parse_slpdb_stages(
    record_name: str,
    sfreq: float,
    total_samples: int,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    wfdb = _require_wfdb()
    samples_per_epoch = int(sfreq * 30.0)
    n_epochs = total_samples // samples_per_epoch
    stages = ["Unknown"] * n_epochs

    ann = wfdb.rdann(str(Path(raw_dir) / record_name), "st")
    for sample, note in zip(ann.sample, ann.aux_note):
        if note is None:
            continue
        text = str(note).replace("\x00", "").strip()
        if not text:
            continue
        code = text.split()[0]
        stage = STAGE_MAP.get(code)
        if stage is None:
            continue
        epoch_idx = int(sample // samples_per_epoch)
        if 0 <= epoch_idx < n_epochs:
            stages[epoch_idx] = stage

    # Forward-fill any gaps if stage codes are sparse
    last_stage = None
    for idx, value in enumerate(stages):
        if value == "Unknown" and last_stage is not None:
            stages[idx] = last_stage
        elif value != "Unknown":
            last_stage = value

    return pd.DataFrame({"sleep_stage": stages})


def process_slpdb_record(
    record_name: str,
    raw_dir: Path,
    output_dir: Path,
) -> Optional[Path]:
    wfdb = _require_wfdb()
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record_path = raw_dir / record_name
    record = wfdb.rdrecord(str(record_path))

    channel_idx = select_eeg_channel(record.sig_name)
    if len(record.sig_name) > 1:
        record = wfdb.rdrecord(str(record_path), channels=[channel_idx])

    signal_data = record.p_signal[:, 0]
    sfreq = float(record.fs)

    band_powers = extract_band_powers(signal_data, sfreq)
    stages = parse_slpdb_stages(record_name, sfreq, len(signal_data), raw_dir=raw_dir)

    if len(band_powers) == 0:
        return None

    min_len = min(len(band_powers), len(stages))
    df = band_powers.iloc[:min_len].copy()
    if len(stages) > 0:
        df["sleep_stage"] = stages["sleep_stage"].iloc[:min_len].values

    output_file = output_dir / f"SLPDB_{record_name}_processed.csv"
    df.to_csv(output_file, index=False)
    return output_file


def process_slpdb_dataset(
    raw_dir: Path,
    output_dir: Path,
    record_pattern: Optional[str] = None,
    max_records: Optional[int] = None,
) -> List[Path]:
    import re

    records = list_slpdb_records(raw_dir)
    if record_pattern:
        regex = re.compile(record_pattern)
        records = [r for r in records if regex.search(r)]
    if max_records is not None:
        records = records[:max_records]

    outputs: List[Path] = []
    for idx, record in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] Processing {record}")
        output_file = process_slpdb_record(record, raw_dir, output_dir)
        if output_file is not None:
            outputs.append(output_file)

    return outputs
