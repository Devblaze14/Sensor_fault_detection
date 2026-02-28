"""Parameterized degradation event simulator.
Supports drift, spike, noise increase, stuck-at, and dropout models."""

import numpy as np


class DegradationSimulator:
    """Injects parameterized degradation events into 1D time-series signals.

    Supported degradation types:
    - Linear bias drift
    - Sudden spike
    - Increased noise variance
    - Stuck-at constant value
    - Intermittent dropout
    """

    def __init__(self, random_seed: int = None):
        self.rng = np.random.default_rng(random_seed)

    def _get_target_indices(self, total_samples: int, start_ratio: float,
                            duration_ratio: float = 1.0) -> tuple:
        """Helper to get indices for the degradation window."""
        start_idx = int(total_samples * start_ratio)
        end_idx = int(total_samples * min(1.0, start_ratio + duration_ratio))
        return start_idx, end_idx

    def inject_bias_drift(self, signal: np.ndarray, start_ratio: float = 0.5,
                          drift_rate: float = 0.01) -> tuple:
        """Injects a linear bias drift starting at start_ratio."""
        degraded = signal.copy()
        start_idx, end_idx = self._get_target_indices(len(signal), start_ratio)

        drift = np.arange(end_idx - start_idx) * drift_rate
        degraded[start_idx:end_idx] += drift

        return degraded, start_idx

    def inject_noise_variance(self, signal: np.ndarray, start_ratio: float = 0.5,
                              multiplier: float = 3.0,
                              base_noise_std: float = 0.05) -> tuple:
        """Increases the underlying noise variance."""
        degraded = signal.copy()
        start_idx, end_idx = self._get_target_indices(len(signal), start_ratio)

        extra_noise = self.rng.normal(0, base_noise_std * multiplier, end_idx - start_idx)
        degraded[start_idx:end_idx] += extra_noise

        return degraded, start_idx

    def inject_spike(self, signal: np.ndarray, spike_ratio: float = 0.5,
                     severity: float = 5.0) -> tuple:
        """Injects a sudden single-sample spike."""
        degraded = signal.copy()
        start_idx = int(len(signal) * spike_ratio)
        degraded[start_idx] += severity

        return degraded, start_idx

    def inject_stuck_at(self, signal: np.ndarray, start_ratio: float = 0.5,
                        stuck_value: float = None) -> tuple:
        """Signal gets stuck at a constant reading."""
        degraded = signal.copy()
        start_idx, end_idx = self._get_target_indices(len(signal), start_ratio)

        if stuck_value is None:
            stuck_value = degraded[start_idx - 1] if start_idx > 0 else 0.0

        degraded[start_idx:end_idx] = stuck_value

        return degraded, start_idx

    def inject_intermittent_dropout(self, signal: np.ndarray, start_ratio: float = 0.5,
                                    dropout_probability: float = 0.1) -> tuple:
        """Intermittently drops signal to 0."""
        degraded = signal.copy()
        start_idx, end_idx = self._get_target_indices(len(signal), start_ratio)

        window_size = end_idx - start_idx
        dropouts = self.rng.random(window_size) < dropout_probability
        degraded[start_idx:end_idx][dropouts] = 0.0

        return degraded, start_idx


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_generation import SignalGenerator

    gen = SignalGenerator(duration=10, noise_std=0.05, random_seed=42)
    clean = gen.generate_clean_signal()
    time = gen.get_time_array()

    sim = DegradationSimulator(random_seed=42)
    degraded, start_idx = sim.inject_bias_drift(clean, start_ratio=0.5, drift_rate=0.02)

    plt.figure(figsize=(10, 4))
    plt.plot(time, clean, label="Clean Signal")
    plt.plot(time, degraded, label="Degraded Signal (Drift)", alpha=0.7)
    plt.axvline(time[start_idx], color='r', linestyle='--', label="Degradation Onset")
    plt.title("Degradation Simulation: Bias Drift")
    plt.legend()
    plt.show()
