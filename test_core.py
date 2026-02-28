"""Basic smoke tests for the detection and health monitoring pipeline.
Run with: python -m pytest test_core.py -v"""

import numpy as np
from data_generation import SignalGenerator
from fault_injection import DegradationSimulator
from evaluation import Evaluator
from health_monitor import HealthMonitor


def test_signal_length():
    """Signal length should equal sampling_rate * duration."""
    gen = SignalGenerator(sampling_rate=100, duration=5.0, random_seed=42)
    signal = gen.generate_clean_signal()
    assert len(signal) == 500, f"Expected 500 samples, got {len(signal)}"


def test_degradation_preserves_length():
    """Degradation injection should not change signal length."""
    signal = np.ones(1000)
    sim = DegradationSimulator(random_seed=42)
    degraded, idx = sim.inject_bias_drift(signal, start_ratio=0.5)
    assert len(degraded) == len(signal), "Degradation changed signal length"
    assert idx == 500, f"Expected onset at 500, got {idx}"


def test_evaluator_perfect_detector():
    """A perfect detector should have F1=1.0 and FPR=0.0."""
    time = np.linspace(0, 10, 1000)
    evaluator = Evaluator(time, ground_truth_idx=500)
    perfect_preds = np.zeros(1000, dtype=int)
    perfect_preds[500:] = 1
    metrics, _ = evaluator.evaluate(perfect_preds)
    assert metrics['F1'] == 1.0, f"Expected F1=1.0, got {metrics['F1']}"
    assert metrics['FPR'] == 0.0, f"Expected FPR=0.0, got {metrics['FPR']}"


def test_health_monitor_normal_operation():
    """Health should stay near 100 when confidence is near 0."""
    monitor = HealthMonitor(smoothing_window=5, health_threshold=60.0, persistence_count=3)
    for _ in range(20):
        status = monitor.update(0.0)
    assert status['health'] == 100.0, "Health should be 100 with zero anomaly"
    assert status['maintenance_recommended'] is False, "No maintenance expected"


def test_health_monitor_triggers_maintenance():
    """Sustained high confidence should trigger maintenance alert."""
    monitor = HealthMonitor(smoothing_window=5, health_threshold=60.0, persistence_count=3)
    # Feed high anomaly confidence
    for _ in range(20):
        status = monitor.update(90.0)
    assert status['health'] < 20.0, "Health should be low with high anomaly"
    assert status['maintenance_recommended'] is True, "Maintenance should be triggered"
