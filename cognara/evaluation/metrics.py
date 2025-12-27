"""
Metrics calculation for regression scoring
"""
import numpy as np

class RegressionMetrics:
    """Calculate various regression and quality metrics"""
    
    @staticmethod
    def calculate_regression_score(baseline_ssim: float, current_ssim: float, defect_count: int) -> float:
        """
        Calculate overall regression score
        
        Returns:
            Score from 0 (severe regression) to 1 (no regression)
        """
        # SSIM component (70% weight)
        ssim_component = current_ssim * 0.7
        
        # Defect penalty (30% weight)
        defect_penalty = max(0, 0.3 - (defect_count * 0.1))
        
        regression_score = ssim_component + defect_penalty
        
        return max(0, min(1, regression_score))
    
    @staticmethod
    def calculate_drift(baseline_metrics: List[float], current_metrics: List[float]) -> Dict:
        """
        Calculate metric drift over time
        
        Returns:
            Drift statistics (mean, std, trend)
        """
        baseline_array = np.array(baseline_metrics)
        current_array = np.array(current_metrics)
        
        mean_drift = np.mean(current_array - baseline_array)
        std_drift = np.std(current_array - baseline_array)
        
        # Simple trend detection
        if len(current_metrics) >= 3:
            recent_trend = np.mean(current_metrics[-3:]) - np.mean(baseline_metrics[-3:])
        else:
            recent_trend = mean_drift
        
        return {
            'mean_drift': float(mean_drift),
            'std_drift': float(std_drift),
            'recent_trend': float(recent_trend),
            'is_degrading': recent_trend < -0.05
        }
    
    @staticmethod
    def false_positive_analysis(predicted_defects: List, ground_truth_defects: List) -> Dict:
        """
        Calculate false positive/negative rates
        
        For evaluation pipeline validation
        """
        tp = len(set(predicted_defects) & set(ground_truth_defects))
        fp = len(set(predicted_defects) - set(ground_truth_defects))
        fn = len(set(ground_truth_defects) - set(predicted_defects))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }