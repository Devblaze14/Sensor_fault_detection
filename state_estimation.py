"""Linear Kalman filter for state estimation and innovation tracking.
Wraps filterpy to provide residual and covariance outputs for detection."""

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanEstimator:
    """State estimation using a standard linear Kalman Filter.

    Tracks a 1D state with velocity.
    State vector: [position, velocity]^T
    """

    def __init__(self, dt: float, process_noise_std: float = 0.1,
                 measurement_noise_std: float = 0.5):
        """
        Args:
            dt: Time step size.
            process_noise_std: Standard deviation of process noise.
            measurement_noise_std: Standard deviation of measurement noise.
        """
        self.dt = dt
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition: constant velocity model
        self.kf.F = np.array([[1., dt],
                              [0., 1.]])

        # Measurement function: observe position only
        self.kf.H = np.array([[1., 0.]])

        # Initial state and covariance
        self.kf.x = np.array([[0.], [0.]])
        self.kf.P *= 10.0

        # Measurement noise covariance
        self.kf.R = np.array([[measurement_noise_std**2]])

        # Process noise covariance (constant velocity + acceleration noise)
        q = process_noise_std**2
        self.kf.Q = np.array([[(dt**4)/4 * q, (dt**3)/2 * q],
                              [(dt**3)/2 * q, dt**2 * q]])

    def initialize_state(self, initial_measurement: float):
        """Set the initial state based on the first measurement."""
        self.kf.x = np.array([[initial_measurement], [0.]])

    def step(self, z: float) -> tuple:
        """Perform one predict-update cycle.

        Returns:
            estimated_state: Current state estimate.
            residual: Innovation (measurement - prediction).
            residual_variance: Innovation covariance.
        """
        self.kf.predict()
        self.kf.update(z)

        est_state = self.kf.x[0, 0]
        residual = self.kf.y[0, 0]
        residual_var = self.kf.S[0, 0]

        return est_state, residual, residual_var


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_generation import SignalGenerator

    gen = SignalGenerator(duration=5, noise_std=0.5, random_seed=42)
    clean = gen.generate_clean_signal()
    noisy = gen.generate_noisy_signal()
    time = gen.get_time_array()

    dt = 1.0 / gen.sampling_rate
    kf = KalmanEstimator(dt=dt, process_noise_std=0.5, measurement_noise_std=0.5)
    kf.initialize_state(noisy[0])

    estimates = []
    residuals = []

    for z in noisy:
        est, res, res_var = kf.step(z)
        estimates.append(est)
        residuals.append(res)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, noisy, label="Noisy Measurement", alpha=0.5)
    plt.plot(time, clean, label="True Signal", linewidth=2)
    plt.plot(time, estimates, label="KF Estimate", linestyle='--')
    plt.title("Kalman Filter State Estimation")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, residuals, color='r')
    plt.title("Innovation (Residuals)")
    plt.tight_layout()
    plt.show()
