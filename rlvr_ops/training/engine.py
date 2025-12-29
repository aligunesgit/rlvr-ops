import torch
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from .grpo import GRPO

class RLVRTrainingEngine:
    def __init__(self, policy, reward_fn, config: Dict[str, Any]):
        self.policy = policy
        self.reward_fn = reward_fn
        self.config = config
        self.grpo = GRPO(policy, reward_fn, config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        
    def train(self, train_loader, num_epochs: int, checkpoint_dir: Optional[str] = None):
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            epoch_metrics = {'loss': 0.0, 'reward_mean': 0.0}
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, batch in enumerate(pbar):
                input_batch = batch['input'].to(self.device)
                ground_truth = batch['ground_truth']
                
                metrics = self.grpo.train_step(input_batch, ground_truth)
                
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'reward': f"{metrics['reward_mean']:.4f}"
                })
            
            for key in epoch_metrics:
                epoch_metrics[key] /= len(train_loader)
            
            print(f"\nEpoch {epoch+1} - Loss: {epoch_metrics['loss']:.4f}, Reward: {epoch_metrics['reward_mean']:.4f}")
            
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                save_path = checkpoint_path / f'checkpoint_epoch_{epoch+1}.pt'
                self.policy.save(str(save_path))
                print(f"Saved checkpoint to {save_path}")
