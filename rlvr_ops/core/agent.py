"""
RLVR Agent - Coordinates interaction between policy and environment
"""
import torch
from typing import Dict, List, Any, Optional
import numpy as np


class RLVRAgent:
    """
    Agent that coordinates policy-environment interaction for RLVR training.
    
    The agent manages the interaction loop: receiving observations from the
    environment, generating actions via the policy, and collecting rewards.
    """
    
    def __init__(self, policy, environment, reward_fn):
        """
        Initialize RLVR Agent.
        
        Args:
            policy: Policy model for text generation
            environment: Task environment (e.g., GSM8k)
            reward_fn: Reward function for evaluating outputs
        """
        self.policy = policy
        self.environment = environment
        self.reward_fn = reward_fn
        self.episode_history = []
        
    def generate_rollout(
        self, 
        prompt: str, 
        ground_truth: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate a single rollout (prompt -> response -> reward).
        
        Args:
            prompt: Input prompt text
            ground_truth: Expected correct answer
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with response, reward, and metadata
        """
        # Generate response
        response = self.policy.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Compute reward
        reward = self.reward_fn(response, ground_truth)
        
        # Store rollout info
        rollout = {
            'prompt': prompt,
            'response': response,
            'ground_truth': ground_truth,
            'reward': reward,
            'temperature': temperature,
            'tokens': max_new_tokens
        }
        
        return rollout
    
    def generate_multi_rollout(
        self,
        prompt: str,
        ground_truth: str,
        num_rollouts: int = 4,
        temperatures: Optional[List[float]] = None,
        max_new_tokens: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple rollouts for a single input.
        
        Args:
            prompt: Input prompt
            ground_truth: Expected answer
            num_rollouts: Number of rollouts to generate
            temperatures: List of temperatures (one per rollout)
            max_new_tokens: Max tokens per rollout
            
        Returns:
            List of rollout dictionaries
        """
        if temperatures is None:
            temperatures = [0.7, 0.8, 0.9, 1.0][:num_rollouts]
        
        if len(temperatures) < num_rollouts:
            # Extend with default temperature
            temperatures.extend([1.0] * (num_rollouts - len(temperatures)))
        
        rollouts = []
        for i in range(num_rollouts):
            rollout = self.generate_rollout(
                prompt=prompt,
                ground_truth=ground_truth,
                max_new_tokens=max_new_tokens,
                temperature=temperatures[i]
            )
            rollout['rollout_id'] = i
            rollouts.append(rollout)
        
        return rollouts
    
    def compute_advantages(
        self, 
        rollouts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute advantages using GRPO baseline (mean reward).
        
        Args:
            rollouts: List of rollout dictionaries
            
        Returns:
            Rollouts with added 'advantage' field
        """
        rewards = [r['reward'] for r in rollouts]
        baseline = np.mean(rewards)
        
        for rollout in rollouts:
            rollout['baseline'] = baseline
            rollout['advantage'] = rollout['reward'] - baseline
        
        return rollouts
    
    def select_best_rollout(
        self,
        rollouts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Select rollout with highest reward.
        
        Args:
            rollouts: List of rollouts
            
        Returns:
            Best rollout dictionary
        """
        return max(rollouts, key=lambda x: x['reward'])
    
    def reset(self):
        """Reset agent state for new episode."""
        self.episode_history = []
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get agent statistics from episode history.
        
        Returns:
            Dictionary of statistics
        """
        if not self.episode_history:
            return {}
        
        rewards = [ep['reward'] for ep in self.episode_history]
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'num_episodes': len(self.episode_history)
        }


class BatchAgent(RLVRAgent):
    """
    Extended agent for batch processing multiple inputs.
    """
    
    def process_batch(
        self,
        prompts: List[str],
        ground_truths: List[str],
        num_rollouts: int = 4,
        temperatures: Optional[List[float]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Process a batch of inputs with multi-rollout generation.
        
        Args:
            prompts: List of input prompts
            ground_truths: List of ground truth answers
            num_rollouts: Rollouts per input
            temperatures: Temperature schedule
            
        Returns:
            List of rollout lists (one per input)
        """
        batch_rollouts = []
        
        for prompt, gt in zip(prompts, ground_truths):
            rollouts = self.generate_multi_rollout(
                prompt=prompt,
                ground_truth=gt,
                num_rollouts=num_rollouts,
                temperatures=temperatures
            )
            rollouts = self.compute_advantages(rollouts)
            batch_rollouts.append(rollouts)
        
        return batch_rollouts
    
    def compute_batch_statistics(
        self,
        batch_rollouts: List[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Compute statistics across entire batch.
        
        Args:
            batch_rollouts: Nested list of rollouts
            
        Returns:
            Batch-level statistics
        """
        all_rewards = []
        all_advantages = []
        best_rewards = []
        
        for rollouts in batch_rollouts:
            rewards = [r['reward'] for r in rollouts]
            advantages = [r['advantage'] for r in rollouts]
            
            all_rewards.extend(rewards)
            all_advantages.extend(advantages)
            best_rewards.append(max(rewards))
        
        return {
            'mean_reward': np.mean(all_rewards),
            'mean_advantage': np.mean(all_advantages),
            'mean_best_reward': np.mean(best_rewards),
            'std_reward': np.std(all_rewards),
            'accuracy': np.mean([1.0 if r > 0 else 0.0 for r in best_rewards])
        }