import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List, Dict, Optional
import re

class GSM8kDataset(Dataset):
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        print(f"Loading GSM8k dataset ({split} split)...")
        self.dataset = load_dataset("gsm8k", "main", split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['question']
        answer = item['answer']
        
        final_answer = self.extract_answer(answer)
        
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"
        
        return {
            'prompt': prompt,
            'question': question,
            'full_answer': answer,
            'final_answer': final_answer,
            'idx': idx
        }
    
    @staticmethod
    def extract_answer(answer_text: str) -> str:
        match = re.search(r'#### (.+)', answer_text)
        if match:
            return match.group(1).strip()
        
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            return numbers[-1]
        
        return ""

def create_gsm8k_dataloader(split: str = "train", batch_size: int = 4, 
                            max_samples: Optional[int] = None, shuffle: bool = True):
    dataset = GSM8kDataset(split=split, max_samples=max_samples)
    
    def collate_fn(batch):
        return {
            'prompts': [item['prompt'] for item in batch],
            'questions': [item['question'] for item in batch],
            'full_answers': [item['full_answer'] for item in batch],
            'final_answers': [item['final_answer'] for item in batch],
            'indices': [item['idx'] for item in batch]
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
