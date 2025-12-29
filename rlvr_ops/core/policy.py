import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class RLVRPolicy(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        self.num_layers = config.get('num_layers', 12)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        raise NotImplementedError("Subclass must implement forward method")
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, **kwargs):
        raise NotImplementedError("Subclass must implement generate method")
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
