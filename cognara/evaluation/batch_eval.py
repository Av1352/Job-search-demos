"""
Batch evaluation across multiple test runs
"""
import numpy as np
from typing import List, Dict

class BatchEvaluator:
    """Run and evaluate multiple test cases"""
    
    def __init__(self, visual_diff_engine, defect_detector):
        self.diff_engine = visual_diff_engine
        self.defect_detector = defect_detector
        
    def evaluate_batch(self, test_cases: List[Dict]) -> Dict:
        """
        Run batch evaluation
        
        Args:
            test_cases: List of dicts with 'baseline' and 'current' images
        
        Returns:
            Aggregated metrics and results
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            baseline = test_case['baseline']
            current = test_case['current']
            
            # Run visual diff
            diff_result = self.diff_engine.compute_diff(baseline, current)
            
            # Detect defects
            defects = self.defect_detector.detect_missing_elements(baseline, current)
            defects.extend(self.defect_detector.detect_layout_shifts(
                baseline, current, diff_result['changed_regions']
            ))
            
            results.append({
                'test_id': i,
                'name': test_case.get('name', f'Test_{i}'),
                'passed': diff_result['passed'] and len(defects) == 0,
                'ssim': diff_result['ssim'],
                'defect_count': len(defects),
                'defects': defects
            })
        
        # Calculate aggregate metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        avg_ssim = np.mean([r['ssim'] for r in results])
        total_defects = sum(r['defect_count'] for r in results)
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'avg_ssim': avg_ssim,
            'total_defects': total_defects,
            'results': results
        }