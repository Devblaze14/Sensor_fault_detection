"""End-to-end demonstration of equipment health monitoring pipeline.
Generates synthetic data, injects degradation, runs all detectors,
and produces a multi-panel diagnostic dashboard."""

import numpy as np
from data_generation import SignalGenerator
from fault_injection import DegradationSimulator
from state_estimation import KalmanEstimator
from detectors import (ThresholdDetector, ZScoreDetector,
                       KalmanResidualDetector, AutoencoderDetector)
from hybrid_logic import HybridDetector
from health_monitor import HealthMonitor
from evaluation import Evaluator
from visualization import plot_detection_results


def main():
    print("--- Equipment Health Monitoring Framework ---")

    # Settings
    dt = 0.01
    duration = 15.0
    noise_std = 0.05
    random_seed = 42

    print("[1/6] Generating baseline signal...")
    gen = SignalGenerator(sampling_rate=int(1 / dt), duration=duration,
                         noise_std=noise_std, random_seed=random_seed)
    time = gen.get_time_array()
    clean = gen.generate_clean_signal()
    noisy = gen.generate_noisy_signal()

    # Separate healthy dataset for autoencoder training
    train_healthy = SignalGenerator(
        sampling_rate=int(1 / dt), duration=20,
        noise_std=noise_std, random_seed=random_seed + 1
    ).generate_noisy_signal()

    print("[2/6] Injecting degradation scenario...")
    sim = DegradationSimulator(random_seed=random_seed)
    degraded_signal, gt_idx = sim.inject_bias_drift(
        noisy, start_ratio=0.5, drift_rate=0.05)

    print("[3/6] Initializing components...")
    kf = KalmanEstimator(dt=dt, process_noise_std=noise_std,
                         measurement_noise_std=noise_std)
    kf.initialize_state(degraded_signal[0])

    thresh_d = ThresholdDetector(lower_bound=clean.min() - 3 * noise_std,
                                upper_bound=clean.max() + 3 * noise_std)
    zscore_d = ZScoreDetector(window_size=int(0.5 / dt), threshold=3.0)
    kf_res_d = KalmanResidualDetector(threshold=9.0)

    ae_d = AutoencoderDetector(window_size=int(0.2 / dt),
                               threshold=noise_std ** 2 * 10)
    print("      Training autoencoder on healthy baseline...")
    ae_d.train_model(train_healthy, epochs=15)

    hybrid_d = HybridDetector()
    monitor = HealthMonitor(smoothing_window=10, health_threshold=60.0,
                            persistence_count=5)

    print("[4/6] Running online detection loop...")

    # Collection arrays
    estimates, residuals, residual_vars = [], [], []
    t_preds, z_preds, k_preds, a_preds, h_preds = [], [], [], [], []
    a_scores, h_scores = [], []
    health_scores, maint_flags = [], []

    for z in degraded_signal:
        # State estimation
        est_state, res, res_var = kf.step(z)
        estimates.append(est_state)
        residuals.append(res)
        residual_vars.append(res_var)

        # Detection
        t_preds.append(thresh_d.detect(z))
        z_preds.append(zscore_d.detect(z))
        k_preds.append(kf_res_d.detect(res, res_var))

        ae_det, a_score = ae_d.detect(z)
        a_preds.append(ae_det)
        a_scores.append(a_score)

        h_det, h_score = hybrid_d.detect(res, res_var, a_score)
        h_preds.append(h_det)
        h_scores.append(h_score)

        # Health monitoring
        status = monitor.update(h_score)
        health_scores.append(status['health'])
        maint_flags.append(status['maintenance_recommended'])

    print("[5/6] Evaluating results...")
    evaluator = Evaluator(time, gt_idx)
    metrics, roc_data = evaluator.evaluate(np.array(h_preds),
                                           np.array(h_scores))

    print(f"  Hybrid Detector — Latency: {metrics['Latency (s)']:.2f}s, "
          f"F1: {metrics['F1']:.2f}")
    print(f"  Lead Time: {metrics.get('Lead Time (s)', 'N/A')}s, "
          f"False Maintenance Alerts: {metrics.get('False Maintenance Alerts', 0)}")

    # Maintenance alert summary
    maint_arr = np.array(maint_flags)
    alert_indices = np.where(maint_arr)[0]
    if len(alert_indices) > 0:
        print(f"  First maintenance alert at t={time[alert_indices[0]]:.2f}s "
              f"(degradation onset at t={time[gt_idx]:.2f}s)")
    else:
        print("  No maintenance alert triggered.")

    print("[6/6] Generating diagnostic dashboard...")
    plot_detection_results(
        time, clean, degraded_signal, estimates, residuals, residual_vars,
        gt_idx, h_scores, t_preds, z_preds, k_preds, a_preds, h_preds,
        roc_data, health_scores=health_scores, maintenance_flags=maint_flags)

    print("Done.")


if __name__ == "__main__":
    main()
