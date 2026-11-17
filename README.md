# Adaptive Network IDS

## Improvements in this version

- Numerical model pipelines impute missing feature values using training-set medians.
- Infinite feature values are converted to missing values before prediction.
- Known-class stream accuracy is calculated from the supervised classifier before the Isolation Forest override. This avoids mixing classifier accuracy with anomaly-override behavior.
- The full adaptive configuration attempts retraining on DDM drift alarms and also on a periodic safety schedule (`RETRAIN_EVERY`).
- Retraining logs distinguish attempts, accepted candidates, and rejected candidates with reasons.
- Retraining uses a chronological holdout from recent feedback rather than randomly mixing recent records.
- Training samples are selected in a class-balanced manner.
- The final incomplete stream window is evaluated rather than silently discarded.

## Setup

1. Put CIC-IDS2017 CSV files in `data/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python train.py`

## Settings

Edit `src/config.py`:
- `UNKNOWN_REAL`: attack class held out from supervised training.
- `CLASS_CAP`: maximum rows retained per class by the loader.
- `TRAIN_CAP`: maximum class-balanced initial training sample.
- `WINDOW_SIZE`: number of records in each stream window.
- `RETRAIN_DELAY`: windows to wait after a DDM alarm.
- `RETRAIN_EVERY`: periodic retraining interval, in stream windows.
- `MIN_DDM_SAMPLES`: minimum verified errors observed before DDM can signal drift.
- `MIN_RETRAIN_SAMPLES`: minimum recent known-class feedback rows required to train a candidate.

## Important interpretation and limitations

- Stream labels are used as simulated analyst-verified feedback. A live IDS must obtain verified labels from an analyst or another trusted feedback process before using them for retraining.
- Candidate models are evaluated on a recent temporal holdout and are accepted only if their macro-F1 is at least as high as the current model. Therefore an attempted retrain may be rejected, and the currently deployed model may remain unchanged.
- The reported known-class accuracy measures the supervised classifier on known ground-truth classes. The final IDS metrics separately include the Isolation Forest override.
- Isolation Forest detects unusual traffic, not confirmed zero-day attacks.
- A held-out attack category is a controlled unknown-class experiment, not proof of universal zero-day detection.
- CIC-IDS2017 CSV file ordering is not guaranteed to represent a continuous live timeline. Concept-drift conclusions should be treated as offline experimental results.
- Models remain in memory only; this project does not save trained models to disk.
