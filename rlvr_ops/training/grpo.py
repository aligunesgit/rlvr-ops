import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Callable
import numpy as np

class GRPO:
    def __init__(self, policy, reward_fn: Callable, config: Dict[str, Any]):
        self.policy = policy
        self.reward_fn = reward_fn
        self.k_rollouts = config.get('k_rollouts', 4)
        self.temperature = config.get('temperature', 1.0)
        self.learning_rate = config.get('learning_rate', 1e-5)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate)
    
    def generate_rollouts(self, input_batch: torch.Tensor, num_rollouts: int) -> List[Dict[str, Any]]:
        rollouts = []
        with torch.no_grad():
            for _ in range(num_rollouts):
                outputs = self.policy.generate(
                    input_batch,
                    max_length=100,
                    do_sample=True,
                    temperature=self.temperature
                )
                rollouts.append({
                    'input': input_batch,
                    'output': outputs,
                    'log_probs': None
                })
        return rollouts
    
    def compute_rewards(self, rollouts: List[Dict[str, Any]], ground_truth: Any) -> torch.Tensor:
        rewards = []
        for rollout in rollouts:
            reward = self.reward_fn(rollout['output'], ground_truth)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
    
    def update_policy(self, rollouts: List[Dict[str, Any]], rewards: torch.Tensor):
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        loss = 0.0
        for rollout, advantage in zip(rollouts, advantages):
            if rollout['log_probs'] is not None:
                policy_loss = -(rollout['log_probs'] * advantage).mean()
                loss += policy_loss
        
        if loss != 0.0:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item() if isinstance(loss, torch.Tensor) else 0.0
    
    def train_step(self, input_batch: torch.Tensor, ground_truth: Any) -> Dict[str, float]:
        rollouts = self.generate_rollouts(input_batch, self.k_rollouts)
        rewards = self.compute_rewards(rollouts, ground_truth)
        loss = self.update_policy(rollouts, rewards)
        
        return {
            'loss': loss,
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'reward_max': rewards.max().item(),
            'reward_min': rewards.min().item()
        }
