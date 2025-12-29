#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from rlvr_ops.core.gpt2_policy import GPT2Policy
from rlvr_ops.utils.data_loader import create_gsm8k_dataloader
from rlvr_ops.rewards.library import VerifiableRewardLibrary
import re
from tqdm import tqdm

def extract_final_answer(text: str) -> str:
    lines = text.strip().split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'-?\d+\.?\d*', line)
        if numbers:
            return numbers[-1]
    return ""

class SimpleGRPO:
    def __init__(self, policy, learning_rate=1e-6):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.model.parameters(), lr=learning_rate)
        
    def compute_loss(self, prompts, responses, rewards):
        """Simple GRPO loss: reward-weighted generation probability"""
        losses = []
        
        for prompt, response, reward in zip(prompts, responses, rewards):
            inputs = self.policy.tokenizer(prompt, return_tensors="pt")
            targets = self.policy.tokenizer(response, return_tensors="pt")
            
            inputs = {k: v.to(self.policy.device) for k, v in inputs.items()}
            targets = {k: v.to(self.policy.device) for k, v in targets.items()}
            
            outputs = self.policy.model(**inputs, labels=targets['input_ids'])
            
            loss = outputs.loss * (1.0 - reward)
            losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0)
    
    def update(self, prompts, responses, rewards):
        """Perform GRPO update"""
        loss = self.compute_loss(prompts, responses, rewards)
        
        if loss.item() > 0:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item()

def main():
    print("=" * 70)
    print("RLVR-Ops: Full GRPO Training on GSM8k")
    print("=" * 70)
    
    print("\n1. Initializing GPT-2 model...")
    policy = GPT2Policy(model_name="gpt2")
    grpo = SimpleGRPO(policy, learning_rate=5e-6)
    print(f"   Parameters: {policy.get_num_parameters():,}")
    print(f"   Device: {policy.device}")
    
    print("\n2. Loading GSM8k dataset...")
    train_loader = create_gsm8k_dataloader(
        split="train",
        batch_size=1,
        max_samples=20,
        shuffle=True
    )
    
    num_epochs = 2
    k_rollouts = 3
    temperature = 0.7
    
    print(f"\n3. Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Samples: {len(train_loader)}")
    print(f"   Rollouts per sample: {k_rollouts}")
    print(f"   Learning rate: 5e-6")
    
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            prompt = batch['prompts'][0]
            ground_truth = batch['final_answers'][0]
            
            rollouts = []
            for k in range(k_rollouts):
                generated = policy.generate(
                    prompt,
                    max_new_tokens=100,
                    temperature=temperature
                )
                
                predicted_answer = extract_final_answer(generated)
                reward = VerifiableRewardLibrary.exact_match_reward(
                    predicted_answer,
                    ground_truth
                )
                
                rollouts.append({
                    'text': generated,
                    'answer': predicted_answer,
                    'reward': reward
                })
            
            responses = [r['text'] for r in rollouts]
            rewards = torch.tensor([r['reward'] for r in rollouts])
            
            loss = grpo.update([prompt] * k_rollouts, responses, rewards)
            
            avg_reward = rewards.mean().item()
            epoch_rewards.append(avg_reward)
            epoch_losses.append(loss)
            
            pbar.set_postfix({
                'reward': f'{avg_reward:.3f}',
                'loss': f'{loss:.4f}'
            })
        
        epoch_avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Reward: {epoch_avg_reward:.4f}")
        print(f"  Average Loss: {epoch_avg_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    print("\nSaving model...")
    save_path = "checkpoints/gpt2_gsm8k_rlvr"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    policy.save(save_path)
    print(f"Model saved to {save_path}")
    
    print("\nTesting trained model...")
    test_prompt = "Question: If Tom has 15 apples and gives 5 away, how many does he have?\nAnswer: Let's solve this step by step.\n"
    result = policy.generate(test_prompt, max_new_tokens=50)
    print(f"Test: {test_prompt[:60]}...")
    print(f"Generated: {result[len(test_prompt):][:100]}...")

if __name__ == '__main__':
    main()
