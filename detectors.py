"""Detection strategies for time-series degradation.
Includes threshold, statistical, model-based, and learned approaches."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import deque


class BaseDetector:
    """Abstract base class for all detection strategies."""

    def detect(self, signal_val: float, **kwargs) -> bool:
        raise NotImplementedError


class ThresholdDetector(BaseDetector):
    """Simple bounds checking on raw signal values."""

    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def detect(self, signal_val: float) -> bool:
        return signal_val < self.lower_bound or signal_val > self.upper_bound


class ZScoreDetector(BaseDetector):
    """Rolling window statistical z-score detector."""

    def __init__(self, window_size: int = 50, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)

    def detect(self, signal_val: float) -> bool:
        self.window.append(signal_val)
        if len(self.window) < self.window_size:
            return False

        mean = np.mean(self.window)
        std = np.std(self.window)
        if std == 0:
            return False

        z_score = abs((signal_val - mean) / std)
        return z_score > self.threshold


class KalmanResidualDetector(BaseDetector):
    """Uses the Normalized Innovation Squared (NIS) from the Kalman filter."""

    def __init__(self, threshold: float = 9.0):
        # 3-sigma corresponds to threshold ~9.0 for 1-DOF chi-square
        self.threshold = threshold

    def detect(self, residual: float, residual_var: float) -> bool:
        if residual_var <= 0:
            return False
        nis = (residual**2) / residual_var
        return nis > self.threshold


# --- Autoencoder Model ---

class StandardAutoencoder(nn.Module):
    """Shallow autoencoder for reconstruction-based anomaly detection.

    Architecture: input → 16 → latent_dim → 16 → input.
    Latent dim defaults to ~20% of input size, a standard compression
    ratio for shallow autoencoders on low-dimensional time-series windows.
    """

    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseDetector):
    """Autoencoder trained on healthy data to flag anomalies
    based on reconstruction error exceeding a threshold."""

    def __init__(self, window_size: int = 20, hidden_dim: int = 4,
                 threshold: float = 0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.model = StandardAutoencoder(input_dim=window_size, latent_dim=hidden_dim)
        self.window = deque(maxlen=window_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def train_model(self, healthy_data: np.ndarray, epochs: int = 50,
                    batch_size: int = 32, lr: float = 0.001):
        """Train the autoencoder on healthy data using sliding windows."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Create sliding windows from 1D data
        num_windows = len(healthy_data) - self.window_size + 1
        if num_windows <= 0:
            raise ValueError("Healthy data is shorter than window size.")

        windows = np.vstack([healthy_data[i:i + self.window_size]
                             for i in range(num_windows)])
        dataset = torch.tensor(windows, dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(dataset), batch_size=batch_size,
                            shuffle=True)

        for epoch in range(epochs):
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()

        self.model.eval()

    def get_anomaly_score(self, window_data: list) -> float:
        """Returns the MSE reconstruction error for a single window."""
        x = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            reconstruction = self.model(x)
            mse = torch.mean((x - reconstruction)**2).item()
        return mse

    def detect(self, signal_val: float) -> tuple:
        """Returns (is_anomaly, anomaly_score)."""
        self.window.append(signal_val)
        if len(self.window) < self.window_size:
            return False, 0.0

        score = self.get_anomaly_score(list(self.window))
        return score > self.threshold, score
