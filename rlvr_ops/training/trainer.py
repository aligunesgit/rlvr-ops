"""
RLVR Trainer - Manages training loop with GRPO algorithm
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Optional, Callable
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime


class GRPOTrainer:
    """
    Trainer implementing Group Relative Policy Optimization (GRPO).
    
    GRPO uses mean reward as baseline and optimizes policy using
    advantage-weighted policy gradients.
    """
    
    def __init__(
        self,
        policy,
        agent,
        learning_rate: float = 5e-6,
        num_rollouts: int = 4,
        temperatures: Optional[List[float]] = None,
        gradient_clip: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GRPO Trainer.
        
        Args:
            policy: Policy model to train
            agent: RLVR agent for rollout generation
            learning_rate: Learning rate for optimizer
            num_rollouts: Number of rollouts per input
            temperatures: Temperature schedule for sampling
            gradient_clip: Gradient clipping threshold
            device: Device for computation
        """
        self.policy = policy
        self.agent = agent
        self.num_rollouts = num_rollouts
        self.temperatures = temperatures or [0.7, 0.8, 0.9, 1.0]
        self.gradient_clip = gradient_clip
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            self.policy.model.parameters(),
            lr=learning_rate
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_reward': [],
            'val_reward': [],
            'val_accuracy': []
        }
    
    def compute_policy_loss(
        self,
        rollouts: List[Dict],
        policy_model
    ) -> torch.Tensor:
        """
        Compute GRPO policy gradient loss.
        
        Args:
            rollouts: List of rollout dictionaries with advantages
            policy_model: Policy model
            
        Returns:
            Loss tensor
        """
        losses = []
        
        for rollout in rollouts:
            prompt = rollout['prompt']
            response = rollout['response']
            advantage = rollout['advantage']
            
            # Tokenize
            inputs = self.policy.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)
            
            targets = self.policy.tokenizer(
                response,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            outputs = policy_model(**inputs, labels=targets.input_ids)
            log_prob = -outputs.loss  # Negative of CE loss
            
            # Advantage-weighted loss
            loss = -advantage * log_prob
            losses.append(loss)
        
        # Mean loss across rollouts
        return torch.stack(losses).mean()
    
    def train_epoch(
        self,
        train_loader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        self.policy.model.train()
        
        epoch_losses = []
        epoch_rewards = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            prompt = batch['prompts'][0]
            ground_truth = batch['final_answers'][0]
            
            # Generate rollouts
            rollouts = self.agent.generate_multi_rollout(
                prompt=prompt,
                ground_truth=ground_truth,
                num_rollouts=self.num_rollouts,
                temperatures=self.temperatures
            )
            
            # Compute advantages
            rollouts = self.agent.compute_advantages(rollouts)
            
            # Compute loss
            loss = self.compute_policy_loss(rollouts, self.policy.model)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(),
                self.gradient_clip
            )
            
            # Update
            self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            mean_reward = np.mean([r['reward'] for r in rollouts])
            epoch_rewards.append(mean_reward)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'reward': f"{mean_reward:.4f}"
            })
        
        return {
            'loss': np.mean(epoch_losses),
            'reward': np.mean(epoch_rewards)
        }
    
    def evaluate(
        self,
        val_loader,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation DataLoader
            max_samples: Max samples to evaluate (None = all)
            
        Returns:
            Validation metrics
        """
        self.policy.model.eval()
        
        correct = 0
        total = 0
        all_rewards = []
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                if max_samples and idx >= max_samples:
                    break
                
                prompt = batch['prompts'][0]
                ground_truth = batch['final_answers'][0]
                
                # Generate rollouts
                rollouts = self.agent.generate_multi_rollout(
                    prompt=prompt,
                    ground_truth=ground_truth,
                    num_rollouts=self.num_rollouts,
                    temperatures=self.temperatures
                )
                
                # Best reward
                best_reward = max(r['reward'] for r in rollouts)
                all_rewards.append(best_reward)
                
                if best_reward > 0:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'mean_reward': np.mean(all_rewards),
            'correct': correct,
            'total': total
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 3,
        save_dir: Optional[Path] = None
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_accuracy = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"\nTraining - Loss: {train_metrics['loss']:.4f}, "
                  f"Reward: {train_metrics['reward']:.4f}")
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            print(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"Reward: {val_metrics['mean_reward']:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_reward'].append(train_metrics['reward'])
            self.history['val_reward'].append(val_metrics['mean_reward'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Save checkpoint if best
            if save_dir and val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                checkpoint_path = save_dir / f"best_model_epoch{epoch}.pt"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"✅ Saved best model: {checkpoint_path}")
        
        # Save final history
        if save_dir:
            history_path = save_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"\n✅ Training complete! History saved: {history_path}")
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict
    ):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.policy.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.policy.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['metrics']