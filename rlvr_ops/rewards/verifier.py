"""
Reward Verifier - Validates and computes verifiable rewards
"""
import re
from typing import Any, Callable, Dict, Optional
import numpy as np


class RewardVerifier:
    """
    Base class for reward verification.
    
    Ensures reward functions are properly formatted and validated.
    """
    
    def __init__(self, reward_fn: Callable):
        """
        Initialize verifier.
        
        Args:
            reward_fn: Reward function to wrap
        """
        self.reward_fn = reward_fn
        self.call_count = 0
        self.reward_history = []
    
    def verify(self, prediction: Any, ground_truth: Any) -> float:
        """
        Verify and compute reward.
        
        Args:
            prediction: Model prediction
            ground_truth: Expected answer
            
        Returns:
            Reward value in [0, 1]
        """
        # Compute reward
        reward = self.reward_fn(prediction, ground_truth)
        
        # Validate reward is in [0, 1]
        if not isinstance(reward, (int, float)):
            raise ValueError(f"Reward must be numeric, got {type(reward)}")
        
        if reward < 0 or reward > 1:
            raise ValueError(f"Reward must be in [0, 1], got {reward}")
        
        # Track history
        self.call_count += 1
        self.reward_history.append(reward)
        
        return float(reward)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get reward statistics."""
        if not self.reward_history:
            return {}
        
        return {
            'mean': np.mean(self.reward_history),
            'std': np.std(self.reward_history),
            'min': np.min(self.reward_history),
            'max': np.max(self.reward_history),
            'count': self.call_count
        }
    
    def reset(self):
        """Reset verifier state."""
        self.call_count = 0
        self.reward_history = []


class ExactMatchVerifier(RewardVerifier):
    """
    Verifier for exact string matching.
    """
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize exact match verifier.
        
        Args:
            case_sensitive: Whether to match case-sensitively
        """
        self.case_sensitive = case_sensitive
        super().__init__(self._exact_match)
    
    def _exact_match(self, prediction: str, ground_truth: str) -> float:
        """Exact match reward function."""
        pred = prediction.strip()
        gt = ground_truth.strip()
        
        if not self.case_sensitive:
            pred = pred.lower()
            gt = gt.lower()
        
        return 1.0 if pred == gt else 0.0


class NumericalVerifier(RewardVerifier):
    """
    Verifier for numerical answers (math problems).
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize numerical verifier.
        
        Args:
            tolerance: Tolerance for float comparison
        """
        self.tolerance = tolerance
        super().__init__(self._numerical_match)
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        # Look for numbers in text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])  # Take last number
            except ValueError:
                return None
        return None
    
    def _numerical_match(self, prediction: str, ground_truth: str) -> float:
        """Numerical matching with tolerance."""
        pred_num = self._extract_number(prediction)
        gt_num = self._extract_number(ground_truth)
        
        if pred_num is None or gt_num is None:
            return 0.0
        
        # Check if within tolerance
        if abs(pred_num - gt_num) <= self.tolerance:
            return 1.0
        
        return 0.0


class CodeExecutionVerifier(RewardVerifier):
    """
    Verifier for code execution (pass/fail test cases).
    """
    
    def __init__(self, test_cases: list, timeout: int = 5):
        """
        Initialize code execution verifier.
        
        Args:
            test_cases: List of (input, expected_output) tuples
            timeout: Execution timeout in seconds
        """
        self.test_cases = test_cases
        self.timeout = timeout
        super().__init__(self._code_execution)
    
    def _code_execution(
        self,
        code: str,
        ground_truth: str = None
    ) -> float:
        """
        Execute code against test cases.
        
        Returns pass rate (0.0 to 1.0).
        """
        if not self.test_cases:
            return 0.0
        
        passed = 0
        
        for test_input, expected_output in self.test_cases:
            try:
                # Create execution environment
                namespace = {}
                exec(code, namespace)
                
                # Run test
                if 'main' in namespace:
                    result = namespace['main'](test_input)
                else:
                    result = eval(code, namespace)
                
                # Check result
                if result == expected_output:
                    passed += 1
                    
            except Exception:
                # Test failed (syntax error, runtime error, etc.)
                continue
        
        return passed / len(self.test_cases)


class F1ScoreVerifier(RewardVerifier):
    """
    Verifier using F1 score for multi-label tasks.
    """
    
    def __init__(self):
        super().__init__(self._f1_score)
    
    def _f1_score(
        self,
        prediction: list,
        ground_truth: list
    ) -> float:
        """
        Compute F1 score.
        
        Args:
            prediction: Predicted labels
            ground_truth: True labels
            
        Returns:
            F1 score
        """
        pred_set = set(prediction)
        gt_set = set(ground_truth)
        
        if len(pred_set) == 0 and len(gt_set) == 0:
            return 1.0
        
        if len(pred_set) == 0 or len(gt_set) == 0:
            return 0.0
        
        # Compute precision and recall
        true_positives = len(pred_set & gt_set)
        precision = true_positives / len(pred_set)
        recall = true_positives / len(gt_set)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1