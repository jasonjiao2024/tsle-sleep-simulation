"""
Sleep Transition Dynamics Index (STDI) Module

Computes transition-based sleep quality metrics:
- Transition probability matrix from sleep stage sequences
- Markov chain entropy rate (Cover & Thomas, Ch. 4) of the transition matrix
- N3 consolidation ratio and cycle regularity
- STDI composite score combining the above

Individual components (transition entropy, consolidation ratio) are standard
measures in sleep research. The STDI composite -- weighting entropy rate,
consolidation, and spectral features into a single index -- is a novel
contribution of this work.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard sleep stage ordering
SLEEP_STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_TO_IDX = {stage: idx for idx, stage in enumerate(SLEEP_STAGES)}


class SleepTransitionAnalyzer:
    """
    Analyzer for sleep stage transition dynamics.

    Computes novel metrics based on the patterns of transitions between
    sleep stages, rather than just time spent in each stage.
    """

    def __init__(self, epoch_duration_sec: float = 30.0):
        """
        Initialize the analyzer.

        Args:
            epoch_duration_sec: Duration of each epoch in seconds
        """
        self.epoch_duration_sec = epoch_duration_sec
        self.epoch_duration_min = epoch_duration_sec / 60.0

    def compute_transition_matrix(
        self,
        hypnogram: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute the transition probability matrix from a hypnogram.

        Args:
            hypnogram: List of sleep stage labels for each epoch
            normalize: If True, return probabilities; if False, return counts

        Returns:
            5x5 matrix where element [i,j] is P(transition from stage i to stage j)
        """
        n_stages = len(SLEEP_STAGES)
        transition_counts = np.zeros((n_stages, n_stages))

        for i in range(len(hypnogram) - 1):
            current = hypnogram[i]
            next_stage = hypnogram[i + 1]

            if current in STAGE_TO_IDX and next_stage in STAGE_TO_IDX:
                from_idx = STAGE_TO_IDX[current]
                to_idx = STAGE_TO_IDX[next_stage]
                transition_counts[from_idx, to_idx] += 1

        if normalize:
            row_sums = transition_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return transition_counts / row_sums
        else:
            return transition_counts

    def compute_transition_entropy(
        self,
        transition_matrix: np.ndarray
    ) -> float:
        """
        Compute Markov chain entropy rate of the transition probability matrix.

        The entropy rate captures the average uncertainty per transition,
        weighted by how often each source state is visited (stationary
        distribution). This is the correct information-theoretic measure
        for a Markov chain, replacing the previous (incorrect) approach
        of computing Shannon entropy over the flattened matrix.

        H = sum_i pi_i * H(P_i)
        where pi is the stationary distribution and
        H(P_i) = -sum_j P_ij * log2(P_ij) is the entropy of row i.

        Reference: Cover & Thomas, "Elements of Information Theory", Ch. 4.

        Interpretation:
        - Low entropy rate: Predictable, healthy sleep cycle patterns
        - High entropy rate: Fragmented, chaotic sleep with random transitions

        Args:
            transition_matrix: Normalized transition probability matrix
                (each row sums to 1.0)

        Returns:
            Normalized entropy rate in [0, 1]
        """
        n = transition_matrix.shape[0]

        # Check for degenerate matrix (all zeros)
        if np.allclose(transition_matrix, 0):
            return 0.0

        # Compute stationary distribution via eigendecomposition of P^T
        # pi * P = pi, so pi is left eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])

        # Normalize and ensure non-negative
        pi_sum = np.sum(np.abs(pi))
        if pi_sum == 0:
            return 0.0
        pi = np.abs(pi) / pi_sum

        # Compute entropy rate: H = sum_i pi_i * H(row_i)
        entropy_rate = 0.0
        for i in range(n):
            if pi[i] <= 1e-12:
                continue
            row = transition_matrix[i]
            nonzero = row[row > 0]
            if len(nonzero) > 0:
                row_entropy = -np.sum(nonzero * np.log2(nonzero))
                entropy_rate += pi[i] * row_entropy

        # Normalize by max entropy (log2 of number of states)
        max_entropy = np.log2(n)
        if max_entropy == 0:
            return 0.0

        return float(np.clip(entropy_rate / max_entropy, 0.0, 1.0))

    def compute_consolidation_ratio(
        self,
        hypnogram: List[str],
        target_stage: str = 'N3'
    ) -> float:
        """
        Compute the consolidation ratio for a target sleep stage.

        Stage consolidation ratio: a standard sleep continuity measure.

        Definition: CR = longest_bout_length / total_stage_time

        Interpretation:
        - CR = 1.0: All time in stage is in one continuous bout (maximally consolidated)
        - CR → 0: Stage time is fragmented into many short bouts

        Args:
            hypnogram: List of sleep stage labels
            target_stage: Which stage to analyze (default: N3 for deep sleep)

        Returns:
            Consolidation ratio in range [0, 1]
        """
        if target_stage not in STAGE_TO_IDX:
            raise ValueError(f"Unknown stage: {target_stage}")

        # Find all bouts of the target stage
        bouts = []
        current_bout_length = 0

        for stage in hypnogram:
            if stage == target_stage:
                current_bout_length += 1
            else:
                if current_bout_length > 0:
                    bouts.append(current_bout_length)
                    current_bout_length = 0

        # Don't forget the last bout if it ends at the end
        if current_bout_length > 0:
            bouts.append(current_bout_length)

        if len(bouts) == 0:
            return 0.0  # No time in target stage

        longest_bout = max(bouts)
        total_time = sum(bouts)

        return longest_bout / total_time

    def compute_cycle_regularity_index(
        self,
        hypnogram: List[str],
        expected_cycle_duration_min: float = 90.0
    ) -> float:
        """
        Compute the regularity of sleep cycles (ultradian rhythm).

        Cycle regularity quantifies ultradian rhythm consistency.

        A healthy sleep cycle is approximately 90 minutes and includes
        progression through N1 → N2 → N3 → REM. This metric measures
        how consistent these cycles are.

        Definition: CRI = 1 - CV(cycle_durations)
        where CV = coefficient of variation = std / mean

        Args:
            hypnogram: List of sleep stage labels
            expected_cycle_duration_min: Expected cycle duration in minutes

        Returns:
            Cycle regularity index in range [0, 1] (1 = perfectly regular)
        """
        # Detect cycle boundaries (transitions into REM or Wake after REM)
        cycle_durations = []
        current_cycle_start = 0
        in_rem = False

        for i, stage in enumerate(hypnogram):
            if stage == 'REM':
                in_rem = True
            elif in_rem and stage != 'REM':
                # End of REM period = end of cycle
                cycle_duration = (i - current_cycle_start) * self.epoch_duration_min
                if cycle_duration > 30:  # Minimum 30 min to count as a cycle
                    cycle_durations.append(cycle_duration)
                current_cycle_start = i
                in_rem = False

        if len(cycle_durations) < 2:
            # Not enough cycles to compute regularity
            return 0.5  # Neutral value

        mean_duration = np.mean(cycle_durations)
        std_duration = np.std(cycle_durations)

        if mean_duration == 0:
            return 0.0

        # Coefficient of variation
        cv = std_duration / mean_duration

        # Convert to regularity index (higher is better)
        # Clip CV at 1.0 for extreme cases
        cri = 1.0 - min(cv, 1.0)

        return cri

    def compute_fragmentation_index(
        self,
        hypnogram: List[str]
    ) -> float:
        """
        Compute sleep fragmentation index.

        Definition: FI = number_of_transitions / total_epochs

        Higher values indicate more fragmented sleep.

        Args:
            hypnogram: List of sleep stage labels

        Returns:
            Fragmentation index (transitions per epoch)
        """
        if len(hypnogram) < 2:
            return 0.0

        transitions = sum(
            1 for i in range(len(hypnogram) - 1)
            if hypnogram[i] != hypnogram[i + 1]
        )

        return transitions / len(hypnogram)

    def compute_sleep_efficiency(
        self,
        hypnogram: List[str]
    ) -> float:
        """
        Compute sleep efficiency.

        Definition: SE = (total_sleep_time / total_recording_time) * 100

        Args:
            hypnogram: List of sleep stage labels

        Returns:
            Sleep efficiency as percentage
        """
        if len(hypnogram) == 0:
            return 0.0

        sleep_epochs = sum(1 for stage in hypnogram if stage != 'Wake')
        return (sleep_epochs / len(hypnogram)) * 100

    def compute_deep_sleep_percentage(
        self,
        hypnogram: List[str]
    ) -> float:
        """
        Compute percentage of sleep time spent in deep sleep (N3).

        Args:
            hypnogram: List of sleep stage labels

        Returns:
            Deep sleep percentage
        """
        if len(hypnogram) == 0:
            return 0.0

        sleep_epochs = [s for s in hypnogram if s != 'Wake']
        if len(sleep_epochs) == 0:
            return 0.0

        n3_epochs = sum(1 for s in sleep_epochs if s == 'N3')
        return (n3_epochs / len(sleep_epochs)) * 100

    def compute_stdi_composite(
        self,
        hypnogram: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Compute the Sleep Transition Dynamics Index (STDI) composite score.

        PATENT CLAIM: This method combines transition-based metrics
        (individually standard in sleep research) into a novel composite
        index using corrected Markov chain entropy rate weighting.

        Args:
            hypnogram: List of sleep stage labels
            weights: Optional custom weights for each component

        Returns:
            Dictionary with all component metrics and composite STDI score
        """
        if weights is None:
            # Default weights emphasizing consolidation and regularity
            weights = {
                'transition_entropy': -0.25,  # Negative: lower is better
                'n3_consolidation': 0.30,     # Positive: higher is better
                'cycle_regularity': 0.25,     # Positive: higher is better
                'fragmentation': -0.20        # Negative: lower is better
            }

        # Compute all component metrics
        transition_matrix = self.compute_transition_matrix(hypnogram)
        transition_entropy = self.compute_transition_entropy(transition_matrix)
        n3_consolidation = self.compute_consolidation_ratio(hypnogram, 'N3')
        n2_consolidation = self.compute_consolidation_ratio(hypnogram, 'N2')
        cycle_regularity = self.compute_cycle_regularity_index(hypnogram)
        fragmentation = self.compute_fragmentation_index(hypnogram)
        sleep_efficiency = self.compute_sleep_efficiency(hypnogram)
        deep_sleep_pct = self.compute_deep_sleep_percentage(hypnogram)

        # Compute weighted composite score
        # Scale each component to [0, 1] range before weighting
        components = {
            'transition_entropy': transition_entropy,
            'n3_consolidation': n3_consolidation,
            'cycle_regularity': cycle_regularity,
            'fragmentation': min(fragmentation, 1.0)  # Cap at 1.0
        }

        stdi_score = 50  # Base score
        for metric, weight in weights.items():
            if metric in components:
                stdi_score += weight * components[metric] * 50

        # Clamp to [0, 100]
        stdi_score = max(0, min(100, stdi_score))

        return {
            'stdi_score': stdi_score,
            'transition_entropy': transition_entropy,
            'n3_consolidation': n3_consolidation,
            'n2_consolidation': n2_consolidation,
            'cycle_regularity': cycle_regularity,
            'fragmentation_index': fragmentation,
            'sleep_efficiency': sleep_efficiency,
            'deep_sleep_percentage': deep_sleep_pct,
            'transition_matrix': transition_matrix.tolist(),
            'n_epochs': len(hypnogram),
            'total_duration_hours': len(hypnogram) * self.epoch_duration_min / 60
        }

    def analyze_stage_distribution(
        self,
        hypnogram: List[str]
    ) -> Dict[str, float]:
        """
        Compute time distribution across sleep stages.

        Args:
            hypnogram: List of sleep stage labels

        Returns:
            Dictionary with percentage of time in each stage
        """
        if len(hypnogram) == 0:
            return {stage: 0.0 for stage in SLEEP_STAGES}

        counts = Counter(hypnogram)
        total = len(hypnogram)

        return {
            stage: (counts.get(stage, 0) / total) * 100
            for stage in SLEEP_STAGES
        }

    def detect_sleep_onset(
        self,
        hypnogram: List[str],
        min_consecutive: int = 3
    ) -> Optional[int]:
        """
        Detect sleep onset (first sustained sleep).

        Args:
            hypnogram: List of sleep stage labels
            min_consecutive: Minimum consecutive non-Wake epochs to count as onset

        Returns:
            Epoch index of sleep onset, or None if no sleep detected
        """
        consecutive_sleep = 0

        for i, stage in enumerate(hypnogram):
            if stage != 'Wake':
                consecutive_sleep += 1
                if consecutive_sleep >= min_consecutive:
                    return i - min_consecutive + 1
            else:
                consecutive_sleep = 0

        return None


def analyze_hypnogram(
    stages: List[str],
    epoch_duration_sec: float = 30.0
) -> Dict:
    """
    Convenience function to analyze a hypnogram and return all STDI metrics.

    Args:
        stages: List of sleep stage labels
        epoch_duration_sec: Duration of each epoch

    Returns:
        Complete STDI analysis results
    """
    analyzer = SleepTransitionAnalyzer(epoch_duration_sec)
    return analyzer.compute_stdi_composite(stages)


if __name__ == '__main__':
    # Demo with synthetic hypnogram
    print("=" * 60)
    print("SLEEP TRANSITION DYNAMICS INDEX (STDI) DEMO")
    print("=" * 60)

    # Create a realistic synthetic hypnogram (8 hours)
    np.random.seed(42)

    # Typical sleep progression: Wake -> N1 -> N2 -> N3 -> N2 -> REM -> repeat
    hypnogram = []

    # Sleep onset (30 min of Wake)
    hypnogram.extend(['Wake'] * 60)

    # Two sleep cycles
    for cycle in range(4):
        # N1 transition
        hypnogram.extend(['N1'] * np.random.randint(5, 15))
        # N2 period
        hypnogram.extend(['N2'] * np.random.randint(30, 50))
        # N3 deep sleep (longer in first cycles)
        n3_duration = max(10, 40 - cycle * 10) + np.random.randint(-5, 5)
        hypnogram.extend(['N3'] * n3_duration)
        # N2 ascending
        hypnogram.extend(['N2'] * np.random.randint(10, 20))
        # REM (longer in later cycles)
        rem_duration = 15 + cycle * 10 + np.random.randint(-5, 5)
        hypnogram.extend(['REM'] * rem_duration)
        # Brief arousal between cycles
        if cycle < 3:
            hypnogram.extend(['Wake'] * np.random.randint(1, 5))

    # Final wake period
    hypnogram.extend(['Wake'] * 20)

    # Analyze
    analyzer = SleepTransitionAnalyzer()
    results = analyzer.compute_stdi_composite(hypnogram)

    print(f"\nAnalyzed {results['n_epochs']} epochs "
          f"({results['total_duration_hours']:.1f} hours)")
    print("-" * 40)
    print(f"STDI Score:           {results['stdi_score']:.1f}/100")
    print(f"Transition Entropy:   {results['transition_entropy']:.3f}")
    print(f"N3 Consolidation:     {results['n3_consolidation']:.3f}")
    print(f"Cycle Regularity:     {results['cycle_regularity']:.3f}")
    print(f"Fragmentation Index:  {results['fragmentation_index']:.3f}")
    print(f"Sleep Efficiency:     {results['sleep_efficiency']:.1f}%")
    print(f"Deep Sleep %:         {results['deep_sleep_percentage']:.1f}%")

    # Stage distribution
    distribution = analyzer.analyze_stage_distribution(hypnogram)
    print("\nStage Distribution:")
    for stage, pct in distribution.items():
        print(f"  {stage}: {pct:.1f}%")
