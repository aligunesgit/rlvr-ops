import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class GPT2Policy(nn.Module):
    def __init__(self, model_name: str = "gpt2", config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_name = model_name
        self.config = config or {}
        
        logger.info(f"Loading GPT-2 model: {model_name}")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50, 
                                temperature: float = 1.0, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )
        
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return {
            'text': generated_text,
            'ids': generated_ids,
            'logprobs': None
        }
    
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, **kwargs):
        result = self.generate_with_logprobs(prompt, max_new_tokens, temperature, **kwargs)
        return result['text']
    
    def compute_log_probs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs
    
    def save(self, path: str):
        logger.info(f"Saving model to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        logger.info(f"Loading model from {path}")
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.model.to(self.device)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def get_num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
