import torch
import torch.nn as nn
import math


class Token_Embeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int, padding_idx: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx)
        
    
    def forward(self, x):
        return self.embedding(x)
    
    @property
    def weight(self):
        return self.embedding.weight

class Positional_Embeddings(nn.Module):

    def __init__(self, seq_len: int, d_model:int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device, past_len: int):
        
        pos = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long) # creating vector of 1-dimension
        pos = pos.unsqueeze(0) # add new dimension in the front (0th idx)

        return self.embedding(pos)
        
class GPTEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, seq_len: int, dropout: float, padding_idx: int):
        super().__init__()
        self.token = Token_Embeddings(vocab_size, d_model, padding_idx)
        self.pos = Positional_Embeddings(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx
    
    # input_ids --> tokenized labels
    def forward(self, input_ids, past_len):

        batch, seq_len = input_ids.shape
        device = input_ids.device

        tok = self.token(input_ids)

        # get the positional embeddings
        pos = self.pos(seq_len, device=device, past_len=past_len)

        x = tok + pos # adding token + positional embeddings
        x = self.dropout(x)

        attention_mask = (input_ids != self.padding_idx).to(x.dtype)
        mask = attention_mask.to(x.dtype).unsqueeze(-1)
        x = x * mask

        return x