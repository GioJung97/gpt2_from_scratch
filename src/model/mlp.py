'''
MLP:
    (1) input should be some tensor from decoder block.
    (2) 
'''

import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)

        self.act = nn.GeLU()
        # self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        return x