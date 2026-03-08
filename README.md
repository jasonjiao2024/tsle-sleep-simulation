# Stimulus-Specific Adaptation Reset Enhances Simulated Auditory Sleep Entrainment

A computational study using a Thalamocortical Stuart-Landau Ensemble (TSLE) model to evaluate 14 auditory entrainment protocols across 208 polysomnography subjects from 5 international sleep databases.

## Key Findings

- **SSA-reset protocol** (periodic 1 Hz frequency wobbles every 5 min during fixed-delta stimulation) produces the largest sleep depth enhancement (Cohen's d = 0.94 vs. fixed delta alone)
- **Progressive frequency descent** (10 → 8.5 → 6 → 2 Hz) performs comparably to fixed delta (d = 0.13, n.s.) but engages qualitatively distinct thalamocortical mechanisms
- **Pulsed (SO phase-locked) delivery** reduces efficacy by 31–37% compared to continuous stimulation
- **Baseline beta power** predicts individual entrainment response (high beta = lower SDRE)

## Study Design

**14 conditions × 208 subjects × 60 min** within-subject repeated-measures design:

| Condition | Description | Session SDRE |
|-----------|-------------|:------------:|
| SSA-Reset Delta | Fixed delta + 1 Hz wobbles every 5 min | +9.53 |
| Prog. Extended | Progressive + extended delta tail | +8.81 |
| Progressive | 10 → 8.5 → 6 → 2 Hz descent | +8.80 |
| Adaptive | Beta-guided protocol selection | +8.77 |
| Fixed Delta | 2 Hz throughout | +8.72 |
| Better Sham | Phase-randomized pulsed | +8.56 |
| Pulsed Delta | SO phase-locked 2 Hz | +8.52 |
| Pulsed Prog. | SO phase-locked progressive | +8.52 |
| Prog. Hybrid | Continuous → pulsed transition | +8.48 |
| Fixed Theta | 6 Hz throughout | +6.13 |
| Sham | Random freq per epoch | +4.72 |
| Fixed Alpha | 8.5 Hz throughout | +2.86 |
| Reverse | 2 → 6 → 8.5 → 10 Hz | +2.46 |
| No Stimulation | F = 0 | +0.45 |

## Model: Thalamocortical Stuart-Landau Ensemble (TSLE)

The TSLE extends the Kuramoto phase-oscillator model with:

- **Stuart-Landau oscillators** (N = 64): amplitude dynamics + frequency-selective resonance
- **Thalamocortical feedback loop**: fast variable T (τ = 10 s) drives frequency shift; slow variable H (τ = 600 s) drives excitability boost
- **Stimulus-specific adaptation (SSA)**: sustained forcing reduces response (τ = 600 s, max 60% reduction); resets on frequency change

Cortical ensemble equation:

```
dz_i/dt = (λ_i(t) + iω_i(t))z_i − |z_i|²z_i + K(z̄ − z_i) + F_eff·e^{iΩt} + σdW_i
```

## Reproduction

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the 14-condition study

```bash
# Full pipeline (208 subjects × 14 conditions × 60 min; ~10 hours)
python scripts/run_redesigned_study.py --workers 6 --duration 60

# Quick test (20 subjects)
python scripts/run_redesigned_study.py --workers 6 --duration 60 --n-subjects 20
```

### Generate figures

```bash
python -m analysis.redesigned_figures
```

## Project Structure

```
├── analysis/
│   ├── thalamocortical_model.py    # Core TSLE model
│   ├── kuramoto_entrainment.py     # Comparison Kuramoto model
│   ├── protocol_comparison.py      # Protocol definitions + metrics
│   ├── redesigned_protocols.py     # 14-condition protocol definitions
│   ├── statistical_validation.py   # Friedman, Wilcoxon, FDR, effect sizes
│   ├── redesigned_figures.py       # 6 publication figures
│   ├── figures.py                  # Additional figures
│   ├── eeg_processing.py           # EDF parsing + band power extraction
│   └── *_processing.py             # Per-database data processors
├── scripts/
│   ├── run_redesigned_study.py     # Main 14-condition pipeline
│   ├── run_protocol_study.py       # 7-condition study
│   ├── run_frequency_scan.py       # Frequency-response mapping
│   └── run_tsle_sensitivity.py     # Parameter sensitivity analysis
├── results/
│   └── redesigned_study/           # 14-condition study results
│       ├── session_metrics.csv     # Session-level metrics (N = 2,912)
│       ├── statistics/             # Statistical reports (JSON)
│       ├── figures/                # Figures (PDF + PNG)
│       ├── verification.json       # Simulation integrity checks
│       ├── cross_validation.json   # Discovery/validation split
│       └── responder_subgroups.json
├── requirements.txt
└── .gitignore
```

## Data Sources

Subject EEG data are publicly available from the following repositories:

| Database | N | Source | URL |
|----------|---|--------|-----|
| Sleep-EDF | 153 | PhysioNet | https://physionet.org/content/sleep-edfx/1.0.0/ |
| CAP Sleep | 15 | PhysioNet | https://physionet.org/content/capslpdb/1.0.0/ |
| DREAMS | 5 | Univ. de Mons | https://zenodo.org/records/2650142 |
| HMC | 11 | PhysioNet | https://physionet.org/content/hmc-sleep-staging/1.1/ |
| SLPDB | 18 | PhysioNet | https://physionet.org/content/slpdb/1.0.0/ |
| Other | 6 | ANPHY, EPCTL | See manuscript |

To download and process the raw EEG data:

```bash
# Download Sleep-EDF (largest database)
bash scripts/download_sleep_edf.sh

# Process all recordings into band power CSVs
python analysis/eeg_processing.py data/raw/sleep-edfx/1.0.0/sleep-cassette
```

Processed data (band powers per 30 s epoch) can be regenerated from the raw EDF files using the processing pipeline above.

## License

MIT License

## Citation

If you use this code or model in your research, please cite this repository.
