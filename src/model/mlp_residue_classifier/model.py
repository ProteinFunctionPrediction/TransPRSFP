import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import TokenClassifierOutput
from model.model import Model

class MLPResidueClassifier(nn.Module, Model):
    def __init__(self, num_labels, input_size=1024, hidden_sizes=(512,), dropout=0.1, ignore_index=-100, encoder_model=None):
        nn.Module.__init__(self)
        Model.__init__(self)

        self.encoder = encoder_model
        self.num_labels = num_labels
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.ignore_index = ignore_index

        layers = [nn.LayerNorm(input_size)]
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, num_labels))

        self.mlp = nn.Sequential(*layers)
    
    def forward(self, last_hidden_state, labels=None):
        if last_hidden_state.ndim != 3:
            raise RuntimeError("last_hidden_state.ndim must be 3")
        
        if last_hidden_state.shape[-1] != self.input_size:
            raise RuntimeError(f"Unexpected embedding dimension {last_hidden_state.shape[-1]}")
        
        logits = self.mlp(last_hidden_state)
        loss = None

        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise RuntimeError(f"labels.shape != logits.shape[:2] ({labels.shape} != {logits.shape[:2]})")
            
            loss = F.cross_entropy(logits.reshape(-1, self.num_labels), labels.reshape(-1), ignore_index=self.ignore_index)

        return TokenClassifierOutput(loss=loss, logits=logits)
        