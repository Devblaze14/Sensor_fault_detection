import numpy as np
from filterpy.kalman import KalmanFilter

class SensorKalmanFilter:
    """
    State estimation using a standard linear Kalman Filter.
    Tracks a 1D state with velocity.
    State vector: [position, velocity]^T
    """
    def __init__(self, dt: float, process_noise_std: float = 0.1, measurement_noise_std: float = 0.5):
        """
        Args:
            dt: Time step size
            process_noise_std: Standard deviation of process noise (acceleration)
            measurement_noise_std: Standard deviation of measurement noise
        """
        self.dt = dt
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State transition matrix
        # x_{k} = x_{k-1} + v_{k-1}*dt
        # v_{k} = v_{k-1}
        self.kf.F = np.array([[1., dt],
                              [0., 1.]])
                              
        # Measurement function
        # We only measure position (acceleration mapped loosely to a value)
        self.kf.H = np.array([[1., 0.]])
        
        # Initial state and covariance
        self.kf.x = np.array([[0.], [0.]])
        self.kf.P *= 10.0
        
        # Measurement noise covariance
        self.kf.R = np.array([[measurement_noise_std**2]])
        
        # Process noise covariance (assuming constant velocity model with acceleration noise)
        q = process_noise_std**2
        self.kf.Q = np.array([[(dt**4)/4 * q, (dt**3)/2 * q],
                              [(dt**3)/2 * q, dt**2 * q]])
                              
    def initialize_state(self, initial_measurement: float):
        """Set the initial state based on the first measurement."""
        self.kf.x = np.array([[initial_measurement], [0.]])
        
    def step(self, z: float) -> tuple:
        """
        Perform one prediction and update step.
        Returns:
            estimated_state: float
            residual: float (innovation)
            residual_variance: float (innovation covariance)
        """
        # Predict step
        self.kf.predict()
        
        # Update step saving intermediate terms for residual calculation
        # filterpy calculates y (residual) and S (system uncertainty) intrinsically
        self.kf.update(z)
        
        # Extract desired states
        est_state = self.kf.x[0, 0]
        residual = self.kf.y[0, 0] # Innovation
        residual_var = self.kf.S[0, 0] # Innovation covariance
        
        return est_state, residual, residual_var

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_generation import IMUSignalGenerator
    
    gen = IMUSignalGenerator(duration=5, noise_std=0.5, random_seed=42)
    clean = gen.generate_clean_signal()
    noisy = gen.generate_noisy_signal()
    time = gen.get_time_array()
    
    dt = 1.0 / gen.sampling_rate
    kf = SensorKalmanFilter(dt=dt, process_noise_std=0.5, measurement_noise_std=0.5)
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
