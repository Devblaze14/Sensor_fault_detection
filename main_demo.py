import numpy as np
import matplotlib.pyplot as plt
from data_generation import IMUSignalGenerator
from fault_injection import FaultInjector
from state_estimation import SensorKalmanFilter
from detectors import ThresholdDetector, ZScoreDetector, KalmanResidualDetector, AutoencoderDetector
from hybrid_logic import HybridDetector
from evaluation import Evaluator

def main():
    print("--- Hybrid Sensor Fault Diagnosis Framework ---")
    
    # Settings
    dt = 0.01
    duration = 15.0
    noise_std = 0.05
    random_seed = 42

    print("[1/5] Generating Data...")
    gen = IMUSignalGenerator(sampling_rate=int(1/dt), duration=duration, noise_std=noise_std, random_seed=random_seed)
    time = gen.get_time_array()
    clean = gen.generate_clean_signal()
    noisy = gen.generate_noisy_signal()
    
    # Healthy dataset for Autoencoder training
    train_healthy = IMUSignalGenerator(sampling_rate=int(1/dt), duration=20, noise_std=noise_std, random_seed=random_seed+1).generate_noisy_signal()

    print("[2/5] Injecting Fault...")
    injector = FaultInjector(random_seed=random_seed)
    faulty_signal, gt_idx = injector.inject_spike(noisy, spike_ratio=0.5, severity=5.0)
    # faulty_signal, gt_idx = injector.inject_bias_drift(noisy, start_ratio=0.5, drift_rate=0.05)
    
    print("[3/5] Initializing Components...")
    kf = SensorKalmanFilter(dt=dt, process_noise_std=noise_std, measurement_noise_std=noise_std)
    kf.initialize_state(faulty_signal[0])
    
    thresh_d = ThresholdDetector(lower_bound=clean.min() - 3*noise_std, upper_bound=clean.max() + 3*noise_std)
    zscore_d = ZScoreDetector(window_size=int(0.5/dt), threshold=3.0)
    kf_res_d = KalmanResidualDetector(threshold=9.0)
    
    ae_d = AutoencoderDetector(window_size=int(0.2/dt), threshold=noise_std**2 * 10)
    print("      Training Autoencoder...")
    ae_d.train_model(train_healthy, epochs=15)
    
    hybrid_d = HybridDetector()

    print("[4/5] Running Online Detection Loop...")
    
    # Logs
    estimates, residuals, residual_vars = [], [], []
    t_preds, z_preds, k_preds, a_preds, h_preds = [], [], [], [], []
    a_scores, h_scores = [], []
    
    for z in faulty_signal:
        # Estimation
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
        
    print("[5/5] Evaluating Results...")
    evaluator = Evaluator(time, gt_idx)
    metrics, roc_data = evaluator.evaluate(np.array(h_preds), np.array(h_scores))
    print(f"Hybrid Detector Latency: {metrics['Latency (s)']:.2f}s, F1: {metrics['F1']:.2f}")

    # Plotting
    plots(time, clean, faulty_signal, estimates, residuals, residual_vars, gt_idx,
          h_scores, t_preds, z_preds, k_preds, a_preds, h_preds, roc_data)


def plots(time, clean, faulty_signal, estimates, residuals, residual_vars, gt_idx,
          h_scores, t_preds, z_preds, k_preds, a_preds, h_preds, roc_data):
    
    # Convert bool lists to int
    t_preds = np.array(t_preds, dtype=int)
    z_preds = np.array(z_preds, dtype=int)
    k_preds = np.array(k_preds, dtype=int)
    a_preds = np.array(a_preds, dtype=int)
    h_preds = np.array(h_preds, dtype=int)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Raw vs Estimated Signal
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time, faulty_signal, label='Faulty Meas.', alpha=0.5, color='gray')
    ax1.plot(time, clean, label='True Signal', linewidth=2, color='green')
    ax1.plot(time, estimates, label='KF Estimate', linestyle='--', color='blue')
    ax1.axvline(time[gt_idx], color='red', linestyle='-', label='Fault Onset')
    ax1.set_title("1. State Estimation")
    ax1.legend()
    
    # 2. Residuals and Covariance Evolution
    ax2 = plt.subplot(3, 2, 3)
    # NIS
    nis = (np.array(residuals)**2) / np.array(residual_vars)
    ax2.plot(time, nis, label='NIS (Residual^2 / Var)', color='purple')
    ax2.axhline(9.0, color='r', linestyle='--', label='3-Sigma Threshold')
    ax2.axvline(time[gt_idx], color='red', linestyle='-')
    ax2.set_title("2. Innovation Analysis (NIS)")
    ax2.legend()
    
    # 3. Hybrid Confidence
    ax3 = plt.subplot(3, 2, 5)
    ax3.plot(time, h_scores, label='Hybrid Confidence (0-100)', color='orange')
    ax3.axhline(50, color='r', linestyle='--', label='Detection Threshold')
    ax3.axvline(time[gt_idx], color='red', linestyle='-')
    ax3.set_title("3. Decision Uncertainty")
    ax3.legend()

    # 4. Detection Timeline
    ax4 = plt.subplot(3, 2, 2)
    y_offsets = [1, 2, 3, 4, 5]
    labels = ['Threshold', 'Z-Score', 'KF Residual', 'Autoencoder', 'Hybrid']
    preds = [t_preds, z_preds, k_preds, a_preds, h_preds]
    
    for y, p, l in zip(y_offsets, preds, labels):
        ax4.plot(time, p * 0.8 + y, label=l, drawstyle='steps-pre')
    ax4.axvline(time[gt_idx], color='red', linestyle='-')
    ax4.set_yticks([1.4, 2.4, 3.4, 4.4, 5.4])
    ax4.set_yticklabels(labels)
    ax4.set_title("4. Detection Timeline comparison")

    # 5. ROC Curve for Hybrid
    ax5 = plt.subplot(3, 2, 4)
    if roc_data:
        fpr, tpr, roc_auc = roc_data
        ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax5.set_xlim([0.0, 1.0])
        ax5.set_ylim([0.0, 1.05])
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title("5. ROC Curve (Hybrid Detector)")
        ax5.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('demo_results.png')
    print("Saved plots to demo_results.png")
    # plt.show()

if __name__ == "__main__":
    main()
