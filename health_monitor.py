"""System health scoring and maintenance alert logic.
Converts anomaly confidence into actionable maintenance recommendations."""

import numpy as np
from collections import deque


class HealthMonitor:
    """Tracks system health over time using a rolling anomaly window.

    Health score = 100 - smoothed anomaly confidence.
    Maintenance is recommended when health stays below a threshold
    for a sustained number of consecutive intervals.
    """

    def __init__(self, smoothing_window: int = 10,
                 health_threshold: float = 60.0,
                 persistence_count: int = 5):
        """
        Args:
            smoothing_window: Number of samples to average confidence over.
            health_threshold: Health below this triggers the persistence counter.
            persistence_count: Consecutive low-health intervals before alert.
        """
        self.smoothing_window = smoothing_window
        self.health_threshold = health_threshold
        self.persistence_count = persistence_count

        self._confidence_buffer = deque(maxlen=smoothing_window)
        self._low_health_streak = 0

    def update(self, anomaly_confidence: float) -> dict:
        """Process one timestep. Returns health status dict."""
        self._confidence_buffer.append(anomaly_confidence)

        # Smoothed confidence (moving average)
        smoothed = np.mean(self._confidence_buffer)
        health = 100.0 - smoothed

        # Persistence logic
        if health < self.health_threshold:
            self._low_health_streak += 1
        else:
            self._low_health_streak = 0

        maintenance_alert = self._low_health_streak >= self.persistence_count

        return {
            'health': round(health, 2),
            'smoothed_confidence': round(smoothed, 2),
            'low_health_streak': self._low_health_streak,
            'maintenance_recommended': maintenance_alert
        }

    def reset(self):
        """Reset state for a new scenario."""
        self._confidence_buffer.clear()
        self._low_health_streak = 0
