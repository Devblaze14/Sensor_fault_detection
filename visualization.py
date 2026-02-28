"""Plotting utilities for detection results and health monitoring.
Generates diagnostic dashboards for single-run visual demos."""

import numpy as np
import matplotlib.pyplot as plt


def plot_detection_results(time, clean, degraded_signal, estimates, residuals,
                           residual_vars, degradation_onset, h_scores,
                           t_preds, z_preds, k_preds, a_preds, h_preds,
                           roc_data, health_scores=None, maintenance_flags=None):
    """Generate a multi-panel diagnostic dashboard.

    Args:
        time: Time array.
        clean: Clean ground-truth signal.
        degraded_signal: Signal with injected degradation.
        estimates: Kalman filter state estimates.
        residuals: Kalman innovation residuals.
        residual_vars: Innovation covariance values.
        degradation_onset: Index where degradation begins.
        h_scores: Hybrid confidence scores (0-100).
        t_preds, z_preds, k_preds, a_preds, h_preds: Boolean detection arrays.
        roc_data: Tuple (fpr, tpr, auc) or None.
        health_scores: Optional health score array (0-100).
        maintenance_flags: Optional boolean array of maintenance alerts.
    """
    # Convert bool lists to int arrays
    t_preds = np.array(t_preds, dtype=int)
    z_preds = np.array(z_preds, dtype=int)
    k_preds = np.array(k_preds, dtype=int)
    a_preds = np.array(a_preds, dtype=int)
    h_preds = np.array(h_preds, dtype=int)

    has_health = health_scores is not None and maintenance_flags is not None
    num_rows = 4 if has_health else 3

    fig = plt.figure(figsize=(16, 4 * num_rows))

    # 1. State Estimation
    ax1 = plt.subplot(num_rows, 2, 1)
    ax1.plot(time, degraded_signal, label='Degraded Signal', alpha=0.5, color='gray')
    ax1.plot(time, clean, label='True Signal', linewidth=2, color='green')
    ax1.plot(time, estimates, label='KF Estimate', linestyle='--', color='blue')
    ax1.axvline(time[degradation_onset], color='red', linestyle='-', label='Degradation Onset')
    ax1.set_title("1. State Estimation")
    ax1.legend(fontsize=8)

    # 2. Innovation Analysis (NIS)
    ax2 = plt.subplot(num_rows, 2, 3)
    nis = (np.array(residuals)**2) / np.array(residual_vars)
    ax2.plot(time, nis, label='NIS (Residual² / Var)', color='purple')
    ax2.axhline(9.0, color='r', linestyle='--', label='3σ Threshold')
    ax2.axvline(time[degradation_onset], color='red', linestyle='-')
    ax2.set_title("2. Innovation Analysis (NIS)")
    ax2.legend(fontsize=8)

    # 3. Hybrid Confidence
    ax3 = plt.subplot(num_rows, 2, 5)
    ax3.plot(time, h_scores, label='Hybrid Confidence (0-100)', color='orange')
    ax3.axhline(50, color='r', linestyle='--', label='Detection Threshold')
    ax3.axvline(time[degradation_onset], color='red', linestyle='-')
    ax3.set_title("3. Degradation Confidence")
    ax3.legend(fontsize=8)

    # 4. Detection Timeline
    ax4 = plt.subplot(num_rows, 2, 2)
    y_offsets = [1, 2, 3, 4, 5]
    labels = ['Threshold', 'Z-Score', 'KF Residual', 'Autoencoder', 'Hybrid']
    preds = [t_preds, z_preds, k_preds, a_preds, h_preds]

    for y, p, l in zip(y_offsets, preds, labels):
        ax4.plot(time, p * 0.8 + y, label=l, drawstyle='steps-pre')
    ax4.axvline(time[degradation_onset], color='red', linestyle='-')
    ax4.set_yticks([1.4, 2.4, 3.4, 4.4, 5.4])
    ax4.set_yticklabels(labels)
    ax4.set_title("4. Detection Timeline Comparison")

    # 5. ROC Curve
    ax5 = plt.subplot(num_rows, 2, 4)
    if roc_data:
        fpr, tpr, roc_auc = roc_data
        ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax5.set_xlim([0.0, 1.0])
        ax5.set_ylim([0.0, 1.05])
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title("5. ROC Curve (Hybrid Detector)")
        ax5.legend(loc="lower right", fontsize=8)

    # 6. Health Score Timeline (if available)
    if has_health:
        ax6 = plt.subplot(num_rows, 2, 6)
        ax6.plot(time, health_scores, label='System Health (0-100)', color='teal', linewidth=2)
        ax6.axhline(60, color='orange', linestyle='--', label='Health Threshold')
        ax6.axvline(time[degradation_onset], color='red', linestyle='-', label='Degradation Onset')

        # Mark first maintenance alert
        maint_arr = np.array(maintenance_flags)
        alert_indices = np.where(maint_arr)[0]
        if len(alert_indices) > 0:
            first_alert = alert_indices[0]
            ax6.axvline(time[first_alert], color='darkred', linestyle='--',
                        linewidth=2, label='Maintenance Alert')
        ax6.set_title("6. System Health & Maintenance Alert")
        ax6.set_ylim([-5, 105])
        ax6.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150)
    print("Saved plots to demo_results.png")
