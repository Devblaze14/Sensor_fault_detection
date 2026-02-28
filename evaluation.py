import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

class Evaluator:
    """
    Evaluates detector performance against given ground truth timestamps.
    """
    def __init__(self, time_array: np.ndarray, ground_truth_idx: int):
        self.time = time_array
        self.gt_idx = ground_truth_idx
        # Assumes single fault injection per scenario
        self.gt_labels = np.zeros(len(time_array), dtype=int)
        if self.gt_idx < len(self.time):
            self.gt_labels[self.gt_idx:] = 1

    def evaluate(self, detections: np.ndarray, scores: np.ndarray = None) -> dict:
        """
        Evaluates a boolean array of detections.
        
        Args:
            detections: Boolean or binary array of length N
            scores: Continuous confidence scores (optional, for ROC)
        """
        # Detection latency
        detected_indices = np.where(detections)[0]
        true_positives = detected_indices[detected_indices >= self.gt_idx]
        
        if len(true_positives) > 0:
            first_tp_idx = true_positives[0]
            latency = self.time[first_tp_idx] - self.time[self.gt_idx]
            latency_samples = first_tp_idx - self.gt_idx
        else:
            latency = float('inf')
            latency_samples = float('inf')

        # Standard classification metrics
        precision = precision_score(self.gt_labels, detections, zero_division=0)
        recall = recall_score(self.gt_labels, detections, zero_division=0)
        f1 = f1_score(self.gt_labels, detections, zero_division=0)
        
        # False positive rate (FPR) = FP / (FP + TN)
        fp = np.sum((detections == 1) & (self.gt_labels == 0))
        tn = np.sum((detections == 0) & (self.gt_labels == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics = {
            'Latency (s)': latency,
            'Latency (samples)': latency_samples,
            'FPR': fpr,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
        
        # ROC AUC if scores provided
        roc_data = None
        if scores is not None:
            fpr_curve, tpr_curve, _ = roc_curve(self.gt_labels, scores)
            roc_auc = auc(fpr_curve, tpr_curve)
            metrics['ROC_AUC'] = roc_auc
            roc_data = (fpr_curve, tpr_curve, roc_auc)
            
        return metrics, roc_data
