import numpy as np

from sensorwatch.train import choose_threshold, demo, train


def test_end_to_end(tmp_path):
    report = train(demo(), tmp_path)
    assert 0 <= report["pr_auc"] <= 1
    assert (tmp_path / "model.joblib").exists()


def test_missed_fault_cost_affects_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.3, 0.8])
    assert choose_threshold(y, p, 100, 1) <= choose_threshold(y, p, 1, 100)
