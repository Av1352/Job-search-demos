"""
Agentic UI interaction with state verification
"""
import time
from typing import Dict, List, Optional
import numpy as np

class UIAction:
    """Represents a single UI action"""
    def __init__(self, action_type: str, target: str, verify_state: Optional[str] = None):
        self.action_type = action_type  # 'click', 'type', 'wait', 'verify'
        self.target = target
        self.verify_state = verify_state
        self.status = 'pending'
        self.execution_time = 0
        self.error = None

class UITestAgent:
    """Agent that executes UI flows with perception-based verification"""
    
    def __init__(self, perception_engine):
        self.perception = perception_engine
        self.action_log = []
        
    def execute_flow(self, actions: List[UIAction], screenshots: Dict[str, np.ndarray]):
        """
        Execute a scripted UI flow with state verification
        
        Args:
            actions: List of UIAction objects
            screenshots: Dict mapping state names to screenshot arrays
        
        Returns:
            Execution results with success/failure for each step
        """
        results = []
        
        for i, action in enumerate(actions):
            step_result = {
                'step': i + 1,
                'action': f"{action.action_type.upper()}: {action.target}",
                'status': 'pending',
                'execution_time': 0,
                'error': None
            }
            
            start_time = time.time()
            
            try:
                # Simulate action execution
                if action.action_type == 'click':
                    # In production: actual device interaction
                    step_result['status'] = 'success'
                    time.sleep(0.1)  # Simulate click
                    
                elif action.action_type == 'type':
                    step_result['status'] = 'success'
                    time.sleep(0.05)  # Simulate typing
                    
                elif action.action_type == 'wait':
                    time.sleep(0.2)
                    step_result['status'] = 'success'
                    
                elif action.action_type == 'verify':
                    # Use perception to verify state
                    if action.verify_state in screenshots:
                        expected_state = screenshots[action.verify_state]
                        # In production: capture actual current screenshot
                        # For demo: simulate verification
                        verification_passed = True  # Placeholder
                        
                        if verification_passed:
                            step_result['status'] = 'success'
                        else:
                            step_result['status'] = 'failed'
                            step_result['error'] = 'State verification failed - UI mismatch detected'
                    else:
                        step_result['status'] = 'failed'
                        step_result['error'] = f'Unknown state: {action.verify_state}'
                
                # Record execution time
                step_result['execution_time'] = int((time.time() - start_time) * 1000)
                
            except Exception as e:
                step_result['status'] = 'error'
                step_result['error'] = str(e)
                step_result['execution_time'] = int((time.time() - start_time) * 1000)
            
            results.append(step_result)
            
            # Stop on failure for demo (in production: configurable)
            if step_result['status'] != 'success':
                break
        
        return results
    
    def categorize_failure(self, step_result: Dict) -> str:
        """
        Categorize failure type for debugging
        
        Returns:
            Failure category: 'perception', 'action', 'state_mismatch', 'timeout'
        """
        if 'verification failed' in step_result.get('error', '').lower():
            return 'perception_failure'
        elif 'not found' in step_result.get('error', '').lower():
            return 'action_failure'
        elif 'mismatch' in step_result.get('error', '').lower():
            return 'state_mismatch'
        elif step_result.get('execution_time', 0) > 5000:
            return 'timeout'
        else:
            return 'unknown_error'