"""Synthetic periodic signal generator with configurable noise.
Produces clean and noisy time-series for degradation simulation."""

import numpy as np


class SignalGenerator:
    """Generates synthetic 1D periodic signals with realistic dynamics
    and configurable baseline Gaussian noise.

    The signal is synthesized using a combination of low-frequency sine waves
    to simulate regular dynamic behavior in a monitored system.
    """

    def __init__(self, sampling_rate: int = 100, duration: float = 60.0,
                 noise_std: float = 0.05, dc_offset: float = 0.0,
                 random_seed: int = None):
        """
        Args:
            sampling_rate: Hz, number of samples per second.
            duration: Seconds, total length of the generated signal.
            noise_std: Standard deviation of the baseline Gaussian noise.
            dc_offset: Constant offset added to the signal.
            random_seed: Seed for reproducibility.
        """
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.noise_std = noise_std
        self.dc_offset = dc_offset
        self.rng = np.random.default_rng(random_seed)

        self.num_samples = int(self.sampling_rate * self.duration)
        self.time = np.linspace(0, self.duration, self.num_samples)

    def generate_clean_signal(self) -> np.ndarray:
        """Generates the true underlying signal without noise.
        Combines a DC offset with low-frequency dynamics (0.5, 2.0, 5.0 Hz).
        """
        dynamics = (
            0.5 * np.sin(2 * np.pi * 0.5 * self.time) +
            0.2 * np.sin(2 * np.pi * 2.0 * self.time) +
            0.1 * np.sin(2 * np.pi * 5.0 * self.time)
        )
        return self.dc_offset + dynamics

    def generate_noisy_signal(self) -> np.ndarray:
        """Generates the measured signal by adding Gaussian noise to the clean signal."""
        clean_signal = self.generate_clean_signal()
        noise = self.rng.normal(0, self.noise_std, self.num_samples)
        return clean_signal + noise

    def get_time_array(self) -> np.ndarray:
        """Returns the time vector."""
        return self.time


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generator = SignalGenerator(duration=10, noise_std=0.1, random_seed=42)
    time = generator.get_time_array()
    clean = generator.generate_clean_signal()
    noisy = generator.generate_noisy_signal()

    plt.figure(figsize=(10, 4))
    plt.plot(time, noisy, label="Noisy Signal", alpha=0.6)
    plt.plot(time, clean, label="Clean Ground Truth", linewidth=2)
    plt.title("Synthetic Signal Generation")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
