import torch
import numpy as np
from typing import Any, Callable, List
from functools import wraps

class VerifiableRewardLibrary:
    @staticmethod
    def classification_reward(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
        correct = (predictions == ground_truth).float()
        return correct.mean().item()
    
    @staticmethod
    def regression_reward(predictions: torch.Tensor, ground_truth: torch.Tensor, threshold: float = 0.1) -> float:
        mae = torch.abs(predictions - ground_truth).mean()
        reward = 1.0 / (1.0 + mae.item())
        return reward
    
    @staticmethod
    def exact_match_reward(prediction: str, ground_truth: str) -> float:
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
    
    @staticmethod
    def code_execution_reward(code: str, test_cases: List[Dict[str, Any]]) -> float:
        passed = 0
        for test in test_cases:
            try:
                local_vars = {}
                exec(code, {}, local_vars)
                if 'solution' in local_vars:
                    result = local_vars['solution'](*test['input'])
                    if result == test['expected']:
                        passed += 1
            except Exception:
                continue
        return passed / len(test_cases) if test_cases else 0.0
    
    @staticmethod
    def f1_score_reward(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
        tp = ((predictions == 1) & (ground_truth == 1)).sum().float()
        fp = ((predictions == 1) & (ground_truth == 0)).sum().float()
        fn = ((predictions == 0) & (ground_truth == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1.item()
    
    @staticmethod
    def custom_reward(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            assert 0.0 <= result <= 1.0, f"Reward must be in [0,1], got {result}"
            return result
        return wrapper
