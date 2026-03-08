"""
EEG-to-HRV Translation Module.

Provides both a discrete stage-based lookup (backward compatible) and a
continuous EEG-features-to-HRV mapping that interpolates between stage
anchor points based on the continuous sleep depth ratio.

The continuous mapping avoids the discontinuity of the original discrete
lookup and introduces physiologically motivated noise scaling.
"""

import pandas as pd
import numpy as np
from scipy import stats
try:
    import neurokit2 as nk  # Optional dependency
except Exception:
    nk = None
from typing import Dict, Tuple, Optional
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sleep depth thresholds used for anchor point placement on the
# continuous sleep-depth axis.  These match the staging thresholds
# in the simulation module.
_STAGE_DEPTH_ANCHORS = {
    'Wake': 0.4,   # typical wake ratio
    'N1':   1.1,   # midpoint of Wake-N2 transition
    'N2':   2.0,   # midpoint of N1-N3 transition
    'N3':   3.5,   # deep sleep
}

# Ordered stages for interpolation (excluding REM, handled separately)
_NREM_STAGES_ORDERED = ['Wake', 'N1', 'N2', 'N3']


class EEGtoHRVTranslator:
    def __init__(self, shhs_data_path: Optional[str] = None):
        if shhs_data_path and Path(shhs_data_path).exists():
            self.shhs_df = pd.read_csv(shhs_data_path)
            self._build_translation_model()
        else:
            self.shhs_df = None
            self._build_default_model()

    def _build_default_model(self):
        self.translation_model = {
            'Wake': {
                'rmssd_mean': 37.5,
                'rmssd_std': 12.5,
                'hr_mean': 70.0,
                'hr_std': 5.0,
                'lf_hf_ratio': 1.2,
                'sdnn': 45.0
            },
            'N1': {
                'rmssd_mean': 42.5,
                'rmssd_std': 12.5,
                'hr_mean': 67.0,
                'hr_std': 5.0,
                'lf_hf_ratio': 1.0,
                'sdnn': 50.0
            },
            'N2': {
                'rmssd_mean': 50.0,
                'rmssd_std': 15.0,
                'hr_mean': 63.0,
                'hr_std': 5.0,
                'lf_hf_ratio': 0.8,
                'sdnn': 55.0
            },
            'N3': {
                'rmssd_mean': 70.0,
                'rmssd_std': 20.0,
                'hr_mean': 57.0,
                'hr_std': 5.0,
                'lf_hf_ratio': 0.6,
                'sdnn': 70.0
            },
            'REM': {
                'rmssd_mean': 32.5,
                'rmssd_std': 12.5,
                'hr_mean': 71.5,
                'hr_std': 6.5,
                'lf_hf_ratio': 1.1,
                'sdnn': 40.0
            }
        }

        self.validated_ranges = {
            'Wake': {'rmssd': (20, 55), 'hr': (60, 80)},
            'N1':   {'rmssd': (25, 60), 'hr': (58, 76)},
            'N2':   {'rmssd': (30, 70), 'hr': (54, 72)},
            'N3':   {'rmssd': (40, 95), 'hr': (48, 66)},
            'REM':  {'rmssd': (18, 50), 'hr': (60, 82)}
        }

    def _build_translation_model(self):
        mappings = {}

        for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
            stage_data = self.shhs_df[self.shhs_df['sleep_stage'] == stage]

            if len(stage_data) > 0:
                mappings[stage] = {
                    'rmssd_mean': stage_data['rmssd'].mean(),
                    'rmssd_std': stage_data['rmssd'].std(),
                    'hr_mean': stage_data['heart_rate'].mean(),
                    'hr_std': stage_data['heart_rate'].std(),
                    'lf_hf_ratio': stage_data['lf_hf_ratio'].mean() if 'lf_hf_ratio' in stage_data.columns else 1.0,
                    'sdnn': stage_data['sdnn'].mean() if 'sdnn' in stage_data.columns else 50.0
                }
            else:
                self._build_default_model()
                return

        self.translation_model = mappings

        self.validated_ranges = {
            'Wake': {'rmssd': (20, 55), 'hr': (60, 80)},
            'N1':   {'rmssd': (25, 60), 'hr': (58, 76)},
            'N2':   {'rmssd': (30, 70), 'hr': (54, 72)},
            'N3':   {'rmssd': (40, 95), 'hr': (48, 66)},
            'REM':  {'rmssd': (18, 50), 'hr': (60, 82)}
        }

    # ------------------------------------------------------------------
    # Continuous EEG-features-to-HRV mapping (NEW)
    # ------------------------------------------------------------------

    def eeg_features_to_hrv(
        self,
        band_powers: Dict[str, float],
        order_parameter: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Continuous mapping from EEG spectral features to HRV metrics.

        Instead of discretizing into a sleep stage and sampling from that
        stage's distribution, this method:
        1. Computes a continuous sleep_depth ratio.
        2. Linearly interpolates HRV parameters between the two nearest
           stage anchor points.
        3. Adds physiological noise that scales with EEG variability
           (proxied by 1 - order_parameter).

        Args:
            band_powers: dict with keys 'delta_power', 'theta_power',
                         'alpha_power', 'beta_power' (or without '_power').
            order_parameter: Kuramoto order parameter r in [0, 1].
                Higher = more synchronized EEG = less HRV noise.
            rng: numpy random Generator (for reproducibility).

        Returns:
            dict with 'rmssd', 'heart_rate', 'lf_hf_ratio',
            'stress_level', 'inferred_stage'.
        """
        if rng is None:
            rng = np.random.default_rng()

        delta = band_powers.get('delta_power', band_powers.get('delta', 0.0))
        theta = band_powers.get('theta_power', band_powers.get('theta', 0.0))
        alpha = band_powers.get('alpha_power', band_powers.get('alpha', 0.0))
        beta = band_powers.get('beta_power', band_powers.get('beta', 0.0))

        sleep_depth = (delta + theta) / (alpha + beta + 1e-9)

        # Determine inferred stage (for labeling / downstream compat)
        if sleep_depth > 2.5:
            inferred_stage = 'N3'
        elif sleep_depth > 1.5:
            inferred_stage = 'N2'
        elif sleep_depth > 0.8:
            inferred_stage = 'N1'
        else:
            inferred_stage = 'Wake'

        # Interpolate HRV parameters along the NREM depth axis
        rmssd_mean, hr_mean, lf_hf, rmssd_base_std, hr_base_std = \
            self._interpolate_hrv_params(sleep_depth)

        # Noise scaling: more variable EEG → more variable HRV
        noise_scale = 1.0 + 0.3 * (1.0 - np.clip(order_parameter, 0.0, 1.0))
        rmssd_std = rmssd_base_std * noise_scale
        hr_std = hr_base_std * noise_scale

        rmssd = rng.normal(rmssd_mean, rmssd_std)
        hr = rng.normal(hr_mean, hr_std)

        # Physiological clipping (wider ranges than before)
        rmssd = float(np.clip(rmssd, 15.0, 100.0))
        hr = float(np.clip(hr, 45.0, 85.0))

        return {
            'rmssd': rmssd,
            'heart_rate': hr,
            'lf_hf_ratio': float(lf_hf),
            'stress_level': self._calculate_stress(rmssd, hr),
            'inferred_stage': inferred_stage,
        }

    def _interpolate_hrv_params(
        self, sleep_depth: float
    ) -> Tuple[float, float, float, float, float]:
        """
        Linearly interpolate HRV parameters between adjacent NREM stage
        anchor points on the continuous sleep-depth axis.

        Returns (rmssd_mean, hr_mean, lf_hf_ratio, rmssd_std, hr_std).
        """
        stages = _NREM_STAGES_ORDERED
        depths = [_STAGE_DEPTH_ANCHORS[s] for s in stages]

        # Clamp sleep_depth to anchor range
        depth_clamped = np.clip(sleep_depth, depths[0], depths[-1])

        # Find the two bracketing anchors
        for i in range(len(depths) - 1):
            if depth_clamped <= depths[i + 1]:
                break

        lo_stage, hi_stage = stages[i], stages[i + 1]
        lo_depth, hi_depth = depths[i], depths[i + 1]

        # Interpolation weight (0 = fully lo_stage, 1 = fully hi_stage)
        span = hi_depth - lo_depth
        t = (depth_clamped - lo_depth) / span if span > 0 else 0.0

        lo = self.translation_model[lo_stage]
        hi = self.translation_model[hi_stage]

        rmssd_mean = lo['rmssd_mean'] + t * (hi['rmssd_mean'] - lo['rmssd_mean'])
        hr_mean = lo['hr_mean'] + t * (hi['hr_mean'] - lo['hr_mean'])
        lf_hf = lo['lf_hf_ratio'] + t * (hi['lf_hf_ratio'] - lo['lf_hf_ratio'])
        rmssd_std = lo['rmssd_std'] + t * (hi['rmssd_std'] - lo['rmssd_std'])
        hr_std = lo['hr_std'] + t * (hi['hr_std'] - lo['hr_std'])

        return rmssd_mean, hr_mean, lf_hf, rmssd_std, hr_std

    # ------------------------------------------------------------------
    # Legacy discrete interface (backward compatible)
    # ------------------------------------------------------------------

    def eeg_stage_to_hrv(self, eeg_stage: str) -> dict:
        """
        Backward-compatible discrete stage-to-HRV lookup.

        Internally delegates to eeg_features_to_hrv using the stage's
        typical sleep depth anchor so that noise scaling is consistent.
        """
        if eeg_stage not in self.translation_model:
            raise ValueError(f"Unknown stage: {eeg_stage}")

        # Map stage to a representative band-power dict
        depth = _STAGE_DEPTH_ANCHORS.get(eeg_stage, 1.0)
        # Construct synthetic band powers that produce the target depth
        # (delta+theta)/(alpha+beta) = depth  with sum = 1.0
        slow = depth / (1.0 + depth)
        fast = 1.0 - slow
        band_powers = {
            'delta_power': slow * 0.6,
            'theta_power': slow * 0.4,
            'alpha_power': fast * 0.55,
            'beta_power': fast * 0.45,
        }
        result = self.eeg_features_to_hrv(band_powers, order_parameter=0.5)
        # Drop inferred_stage from legacy interface for compat
        return {k: v for k, v in result.items() if k != 'inferred_stage'}

    def _calculate_stress(self, rmssd: float, hr: float) -> float:
        rmssd_stress = (1 - (rmssd - 20) / 80) * 100
        hr_stress = ((hr - 50) / 30) * 100
        stress = 0.6 * rmssd_stress + 0.4 * hr_stress
        return float(np.clip(stress, 0, 100))

    def validate_translation_accuracy(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import cross_val_score

        if not hasattr(self, 'shhs_df') or self.shhs_df is None:
            logger.warning(
                "No SHHS data available for validation. "
                "Cannot compute translation accuracy."
            )
            return float('nan'), None

        required_cols = ['rmssd', 'heart_rate', 'sleep_stage']
        if not all(col in self.shhs_df.columns for col in required_cols):
            logger.warning("Required columns missing from SHHS data.")
            return float('nan'), None

        X = self.shhs_df[['rmssd', 'heart_rate']].copy()
        if 'lf_hf_ratio' in self.shhs_df.columns:
            X['lf_hf_ratio'] = self.shhs_df['lf_hf_ratio']
        if 'sdnn' in self.shhs_df.columns:
            X['sdnn'] = self.shhs_df['sdnn']

        y = self.shhs_df['sleep_stage']
        X = X.fillna(X.mean())

        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        cv_scores = cross_val_score(clf, X, y, cv=5)
        accuracy = cv_scores.mean()

        # Confusion matrix on held-out fold predictions (not training data)
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(clf, X, y, cv=5)
        cm = confusion_matrix(y, y_pred)

        logger.info(f"HRV-only sleep stage classification accuracy: {accuracy:.2%}")
        return accuracy, cm

    def save_translation_model(self, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.translation_model, f, indent=2)

        logger.info(f"Translation model saved to {output_path}")
