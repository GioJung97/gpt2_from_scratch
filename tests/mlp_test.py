import torch
from src.model.embeddings import Token_Embeddings, Positional_Embeddings, GPTEmbeddings
from src.model.mlp import MLP

d_model = 512
d_ff = 1024
vocab_size = 52056

seq_len = 1024
dropout = 0.1


output_from_decoder = torch.rand((1, d_model, d_model))


pos_emb = Positional_Embeddings(seq_len, d_model)

linear_mlp = MLP(d_model, d_model, dropout)
print(f"MLP: {linear_mlp(output_from_decoder)}")

