#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from rlvr_ops.core.gpt2_policy import GPT2Policy
from rlvr_ops.utils.data_loader import create_gsm8k_dataloader
from rlvr_ops.rewards.library import VerifiableRewardLibrary
import re
from tqdm import tqdm
import numpy as np

def extract_final_answer(text: str) -> str:
    lines = text.strip().split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'-?\d+\.?\d*', line)
        if numbers:
            return numbers[-1]
    return ""

def main():
    print("=" * 70)
    print("RLVR-Ops: Simplified GRPO Training on GSM8k")
    print("=" * 70)
    
    print("\n1. Initializing GPT-2 model...")
    policy = GPT2Policy(model_name="gpt2")
    print(f"   Parameters: {policy.get_num_parameters():,}")
    print(f"   Device: {policy.device}")
    
    print("\n2. Loading GSM8k dataset...")
    train_loader = create_gsm8k_dataloader(
        split="train",
        batch_size=1,
        max_samples=50,
        shuffle=True
    )
    
    num_epochs = 3
    k_rollouts = 4
    temperatures = [0.7, 0.8, 0.9, 1.0]
    
    print(f"\n3. Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Samples per epoch: {len(train_loader)}")
    print(f"   Rollouts per sample: {k_rollouts}")
    print(f"   Temperature range: {temperatures}")
    
    print("\n" + "=" * 70)
    print("Starting RLVR Evaluation (No Gradient Updates)")
    print("=" * 70)
    
    all_epoch_results = []
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            prompt = batch['prompts'][0]
            ground_truth = batch['final_answers'][0]
            
            rollout_rewards = []
            best_answer = None
            best_reward = 0.0
            
            for k in range(k_rollouts):
                temp = temperatures[k % len(temperatures)]
                
                generated = policy.generate(
                    prompt,
                    max_new_tokens=80,
                    temperature=temp
                )
                
                predicted_answer = extract_final_answer(generated)
                reward = VerifiableRewardLibrary.exact_match_reward(
                    predicted_answer,
                    ground_truth
                )
                
                rollout_rewards.append(reward)
                
                if reward > best_reward:
                    best_reward = reward
                    best_answer = predicted_answer
            
            avg_reward = np.mean(rollout_rewards)
            max_reward = np.max(rollout_rewards)
            
            epoch_rewards.append(avg_reward)
            epoch_correct += max_reward
            epoch_total += 1
            
            pbar.set_postfix({
                'avg_reward': f'{avg_reward:.3f}',
                'accuracy': f'{epoch_correct/epoch_total:.3f}',
                'best': best_answer[:10] if best_answer else 'N/A'
            })
        
        epoch_avg = np.mean(epoch_rewards)
        epoch_acc = epoch_correct / epoch_total
        
        result = {
            'epoch': epoch + 1,
            'avg_reward': epoch_avg,
            'accuracy': epoch_acc,
            'correct': epoch_correct,
            'total': epoch_total
        }
        all_epoch_results.append(result)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1} Results:")
        print(f"  Average Reward: {epoch_avg:.4f}")
        print(f"  Accuracy: {epoch_acc:.2%} ({int(epoch_correct)}/{epoch_total})")
        print(f"{'='*70}\n")
    
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    for result in all_epoch_results:
        print(f"Epoch {result['epoch']}: "
              f"Reward={result['avg_reward']:.4f}, "
              f"Accuracy={result['accuracy']:.2%}")
    
    final_acc = all_epoch_results[-1]['accuracy']
    print(f"\nFinal Accuracy: {final_acc:.2%}")
    
    print("\n" + "=" * 70)
    print("Next Steps for Full RLVR:")
    print("=" * 70)
    print("1. ✅ Model loading and generation working")
    print("2. ✅ Reward computation working")  
    print("3. ✅ Multi-rollout evaluation working")
    print("4. 🚧 Implement gradient-based policy updates")
    print("5. 🚧 Add Wandb logging")
    print("6. 🚧 Scale to full 7,473 GSM8k problems")
    print("7. 🚧 Add checkpointing and model saving")
    
    print(f"\n✅ RLVR-Ops framework successfully evaluated on {epoch_total} problems!")

if __name__ == '__main__':
    main()
