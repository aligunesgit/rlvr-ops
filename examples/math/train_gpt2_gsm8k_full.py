#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from rlvr_ops.core.gpt2_policy import GPT2Policy
from rlvr_ops.utils.data_loader import create_gsm8k_dataloader
from rlvr_ops.rewards.library import VerifiableRewardLibrary
import re

def extract_final_answer(text: str) -> str:
    lines = text.strip().split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'-?\d+\.?\d*', line)
        if numbers:
            return numbers[-1]
    return ""

def main():
    print("=" * 70)
    print("RLVR-Ops: Training GPT-2 on GSM8k")
    print("=" * 70)
    
    print("\n1. Initializing GPT-2 model...")
    policy = GPT2Policy(model_name="gpt2")
    print(f"   Model parameters: {policy.get_num_parameters():,}")
    print(f"   Device: {policy.device}")
    
    print("\n2. Loading GSM8k dataset...")
    train_loader = create_gsm8k_dataloader(
        split="train",
        batch_size=1,
        max_samples=10,
        shuffle=False
    )
    print(f"   Training samples: {len(train_loader)}")
    
    print("\n3. Starting RLVR Training...")
    print("-" * 70)
    
    k_rollouts = 3
    temperature = 0.7
    
    for batch_idx, batch in enumerate(train_loader):
        prompt = batch['prompts'][0]
        ground_truth = batch['final_answers'][0]
        
        print(f"\nProblem {batch_idx + 1}:")
        print(f"Question: {batch['questions'][0][:80]}...")
        print(f"Ground Truth: {ground_truth}")
        
        rollouts = []
        print(f"\nGenerating {k_rollouts} rollouts...")
        
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
            
            print(f"  Rollout {k+1}: Answer={predicted_answer}, Reward={reward:.2f}")
        
        avg_reward = sum(r['reward'] for r in rollouts) / len(rollouts)
        print(f"\n  Average Reward: {avg_reward:.3f}")
        
        if batch_idx >= 4:
            break
    
    print("\n" + "=" * 70)
    print("Training Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Implement full GRPO gradient updates")
    print("  - Add Wandb logging")
    print("  - Train on full dataset")
    print("  - Save checkpoints")

if __name__ == '__main__':
    main()
