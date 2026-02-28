import numpy as np

class IMUSignalGenerator:
    """
    Generates synthetic 1D IMU acceleration data with realistic motion patterns
    and configurable baseline Gaussian noise.
    
    The motion pattern is synthesized using a combination of low-frequency sine waves
    to simulate structural vibration or regular dynamic movement.
    """
    def __init__(self, sampling_rate: int = 100, duration: float = 60.0,
                 noise_std: float = 0.05, base_value: float = 9.81, random_seed: int = None):
        """
        Args:
            sampling_rate: Hz, number of samples per second
            duration: seconds, total length of the generated signal
            noise_std: standard deviation of the baseline Gaussian noise
            base_value: The DC offset (e.g., 9.81 m/s^2 for gravity on Z-axis)
            random_seed: seed for reproducibility
        """
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.noise_std = noise_std
        self.base_value = base_value
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.num_samples = int(self.sampling_rate * self.duration)
        self.time = np.linspace(0, self.duration, self.num_samples)
        
    def generate_clean_signal(self) -> np.ndarray:
        """
        Generates the true underlying signal without any noise.
        Combines a DC offset with some low-frequency dynamics.
        """
        # Create a dynamic motion profile (e.g., simulating normal operation vibration)
        # Frequencies: 0.5 Hz, 2 Hz, 5 Hz
        dynamics = (
            0.5 * np.sin(2 * np.pi * 0.5 * self.time) +
            0.2 * np.sin(2 * np.pi * 2.0 * self.time) +
            0.1 * np.sin(2 * np.pi * 5.0 * self.time)
        )
        clean_signal = self.base_value + dynamics
        return clean_signal
        
    def generate_noisy_signal(self) -> np.ndarray:
        """
        Generates the measured signal by adding Gaussian noise to the clean signal.
        """
        clean_signal = self.generate_clean_signal()
        noise = np.random.normal(0, self.noise_std, self.num_samples)
        noisy_signal = clean_signal + noise
        return noisy_signal

    def get_time_array(self) -> np.ndarray:
        """Returns the time vector."""
        return self.time

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Simple test of the generator
    generator = IMUSignalGenerator(duration=10, noise_std=0.1, random_seed=42)
    time = generator.get_time_array()
    clean = generator.generate_clean_signal()
    noisy = generator.generate_noisy_signal()
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, noisy, label="Noisy Signal", alpha=0.6)
    plt.plot(time, clean, label="Clean Ground Truth", linewidth=2)
    plt.title("Synthetic IMU Signal Generation")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid()
    plt.show()
