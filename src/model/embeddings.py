import torch
import torch.nn as nn
import math


class Token_Embedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int, padding_idx: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx)
        
    
    def forward(self, x):
        return self.embedding(x)

class Positional_Embedding(nn.Module):

    def __init__(self, seq_len: int, d_model:int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        return x
        
