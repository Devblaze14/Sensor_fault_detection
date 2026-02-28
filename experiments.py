"""Batch experiment runner for degradation detection benchmarking.
Sweeps noise levels and degradation severities, collecting metrics for all detectors."""

import numpy as np
import pandas as pd
from data_generation import SignalGenerator
from fault_injection import DegradationSimulator
from state_estimation import KalmanEstimator
from detectors import (ThresholdDetector, ZScoreDetector,
                       KalmanResidualDetector, AutoencoderDetector)
from hybrid_logic import HybridDetector
from health_monitor import HealthMonitor
from evaluation import Evaluator


def run_single_experiment(noise_std, severity, dt, duration=10, random_seed=42):
    """Run one degradation scenario and evaluate all detectors."""

    # 1. Generate baseline signal
    gen = SignalGenerator(sampling_rate=int(1 / dt), duration=duration,
                         noise_std=noise_std, random_seed=random_seed)
    time = gen.get_time_array()
    clean = gen.generate_clean_signal()
    noisy = gen.generate_noisy_signal()

    # Separate healthy dataset for autoencoder training
    gen_train = SignalGenerator(sampling_rate=int(1 / dt), duration=20,
                               noise_std=noise_std, random_seed=random_seed + 1)
    train_healthy = gen_train.generate_noisy_signal()

    # 2. Inject degradation (bias drift)
    sim = DegradationSimulator(random_seed=random_seed)
    degraded_signal, gt_idx = sim.inject_bias_drift(
        noisy, start_ratio=0.5, drift_rate=severity)

    # 3. Set up estimators and detectors
    kf = KalmanEstimator(dt=dt, process_noise_std=noise_std,
                         measurement_noise_std=noise_std)
    kf.initialize_state(degraded_signal[0])

    thresh_d = ThresholdDetector(lower_bound=clean.min() - 3 * noise_std,
                                upper_bound=clean.max() + 3 * noise_std)
    zscore_d = ZScoreDetector(window_size=int(0.5 / dt), threshold=3.0)
    kf_res_d = KalmanResidualDetector(threshold=9.0)

    ae_d = AutoencoderDetector(window_size=int(0.2 / dt),
                               threshold=noise_std ** 2 * 5)
    ae_d.train_model(train_healthy, epochs=10)

    hybrid_d = HybridDetector()
    monitor = HealthMonitor(smoothing_window=10, health_threshold=60.0,
                            persistence_count=5)

    # 4. Online detection loop
    thresh_preds, z_preds, kf_preds, ae_preds = [], [], [], []
    hybrid_preds, hybrid_scores = [], []
    health_scores, maint_flags = [], []

    for z in degraded_signal:
        est_state, res, res_var = kf.step(z)

        t_det = thresh_d.detect(z)
        z_det = zscore_d.detect(z)
        k_det = kf_res_d.detect(res, res_var)
        ae_det, ae_score = ae_d.detect(z)
        h_det, h_score = hybrid_d.detect(res, res_var, ae_score)

        status = monitor.update(h_score)

        thresh_preds.append(t_det)
        z_preds.append(z_det)
        kf_preds.append(k_det)
        ae_preds.append(ae_det)
        hybrid_preds.append(h_det)
        hybrid_scores.append(h_score)
        health_scores.append(status['health'])
        maint_flags.append(status['maintenance_recommended'])

    # 5. Evaluate
    evaluator = Evaluator(time, gt_idx)
    results = {}

    metrics, _ = evaluator.evaluate(np.array(thresh_preds))
    results['Threshold'] = metrics

    metrics, _ = evaluator.evaluate(np.array(z_preds))
    results['Z-Score'] = metrics

    metrics, _ = evaluator.evaluate(np.array(kf_preds))
    results['Kalman'] = metrics

    metrics, _ = evaluator.evaluate(np.array(ae_preds))
    results['Autoencoder'] = metrics

    metrics, _ = evaluator.evaluate(np.array(hybrid_preds),
                                    np.array(hybrid_scores))
    results['Hybrid'] = metrics

    return results


def run_experiments():
    print("Running Controlled Degradation Experiments...")

    noise_levels = [0.01, 0.05, 0.1]
    severities = [0.01, 0.05, 0.1]  # Drift rates

    all_results = []

    for noise in noise_levels:
        for sev in severities:
            res = run_single_experiment(noise_std=noise, severity=sev, dt=0.01)

            for detector_name, metrics in res.items():
                row = {
                    'Noise': noise,
                    'Severity': sev,
                    'Detector': detector_name,
                    'Latency (s)': metrics['Latency (s)'],
                    'Lead Time (s)': metrics.get('Lead Time (s)', ''),
                    'False Maint. Alerts': metrics.get('False Maintenance Alerts', ''),
                    'F1 Score': metrics['F1'],
                    'FPR': metrics['FPR']
                }
                all_results.append(row)

    df = pd.DataFrame(all_results)
    print("\nExperiment Results Summary:")
    print(df.groupby(['Noise', 'Detector']).mean(numeric_only=True)[
        ['Latency (s)', 'F1 Score', 'FPR']])

    return df


if __name__ == "__main__":
    df = run_experiments()
    df.to_csv("experiment_results.csv", index=False)
    print("Saved results to experiment_results.csv")
