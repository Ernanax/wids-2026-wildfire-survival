# WiDS Global Datathon 2026 - Wildfire Survival Analysis

Predicting the probability that a wildfire reaches an evacuation zone within **12, 24, 48 and 72 hours**, using only data from the first 5 hours after ignition.

Competition: [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26) on Kaggle.

## Problem

This is a right-censored survival problem: for some fires, the hit is observed (`event = 1`). For others, tracking ended before the fire reached the zone (`event = 0`). The dataset is small (221 train / 95 test fires).

Submissions are scored with a hybrid metric:

```
Score  = 0.3 × C-index + 0.7 × (1 − WBrier)
WBrier = 0.3 × Brier@24h + 0.4 × Brier@48h + 0.3 × Brier@72h
```

## Approach

- **Gradient Boosting Survival Analysis**: 10 configurations × 60 seeds, on raw features
- **LightGBM classifiers**, one per horizon (12/24/48h), with IPCW weights to account for censoring, × 70 seeds, on engineered features (distance transforms, ETA, threat score…)
- **Random Survival Forest** and **CoxNet** as additional views
- Per-horizon **blend weights optimized with Nelder-Mead** on out-of-fold predictions
- Post-processing: power calibration at 24h, monotonic probabilities (p12 ≤ p24 ≤ p48 ≤ p72)

All models are evaluated with 5-fold stratified cross-validation.

**Result: OOF hybrid score of 0.975** (C-index 0.947, weighted Brier 0.013).

## Usage

1. Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition page into a `data/` folder.
2. Install the dependencies and run:

```bash
pip install -r requirements.txt
python pipeline.py
```

The submission is written to `submissions/submission.csv`. A full run takes about 1.5 hours on a laptop CPU (most of it in the GBSA ensemble: 3,000 models).
