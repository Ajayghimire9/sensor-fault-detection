# SensorWatch

Cost-aware sensor fault classification.

SensorWatch models an imbalanced binary fault label and chooses an operating threshold based on the relative cost of missed faults and false alarms.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -r requirements-portfolio.txt
python -m sensorwatch.train
# For a real dataset:
python -m sensorwatch.train --data path/to/sensors.csv
```

## Design decisions

Input is a CSV with numeric sensor columns and a fault label of 0 or 1. Median imputation and missingness indicators are fitted on training data only.

A class-weighted random forest is trained on 60% of the data; the validation partition selects the threshold and the test partition reports ROC-AUC, PR-AUC and confusion counts.

The artifact contains the fitted preprocessing pipeline, ordered features and selected threshold. Regression tests check cost-sensitive threshold behavior.

## Technology

Python, pandas, scikit-learn, joblib, pytest, GitHub Actions.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

The default run generates synthetic data to exercise the pipeline. Its scores are not APS benchmark results or evidence of field performance. The IID split must be replaced with machine- or time-separated evaluation when observations are correlated.
