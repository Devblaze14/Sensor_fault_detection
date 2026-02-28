class HybridDetector:
    """
    Fuses multiple weak learners/detectors into a single robust decision.
    Specifically, combines:
    1. Kalman residual magnitude (normalized by covariance, i.e., NIS)
    2. ML Autoencoder anomaly score
    """
    def __init__(self, kf_threshold: float = 9.0, ml_threshold: float = 0.5):
        self.kf_threshold = kf_threshold
        self.ml_threshold = ml_threshold

    def calculate_confidence(self, nis: float, ml_score: float) -> float:
        """
        Maps the raw scores to a 0-100 confidence scale that a fault has occurred.
        Uses a heuristic bounded scaling approach.
        """
        # NIS contribution: 0 at nis=0, ~50 at threshold, max 100
        kf_confidence = min(100.0, (nis / self.kf_threshold) * 50.0)
        
        # ML contribution: 0 at score=0, ~50 at threshold, max 100
        ml_confidence = min(100.0, (ml_score / self.ml_threshold) * 50.0)
        
        # Weighted combination, giving slightly more weight to the Kalman filter 
        # because it models the physics/kinematics better than a raw blackbox ML model
        total_confidence = 0.6 * kf_confidence + 0.4 * ml_confidence
        
        return min(100.0, total_confidence)

    def detect(self, residual: float, residual_var: float, ml_score: float) -> tuple:
        """
        Returns a tuple: (is_fault_detected, confidence_score)
        """
        nis = 0.0
        if residual_var > 0:
            nis = (residual**2) / residual_var
            
        confidence = self.calculate_confidence(nis, ml_score)
        
        # Decision: If we are > 50% confident, we flag a fault
        is_detected = confidence > 50.0
        
        return is_detected, confidence
