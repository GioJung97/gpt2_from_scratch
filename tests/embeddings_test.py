from src.model.embeddings import Token_Embeddings, Positional_Embeddings, GPTEmbeddings
import torch

# what we need
# data, d_model, vocab_size, seq_len, dropout, padding_idx
d_model = 512
vocab_size = 52056

seq_len = 1024
dropout = 0.1

# data = torch.randint(0, vocab_size, (seq_len,))
# batch_data = torch.randint(0, vocab_size, (seq_len,)).unsqueeze(0)

data = torch.arange(1, seq_len + 1)
batched_data = torch.arange(1, seq_len + 1).unsqueeze(0)

padding_idx = data[-1].item()

tok_emb = Token_Embeddings(d_model, vocab_size, padding_idx)
pos_emb = Positional_Embeddings(seq_len, d_model)



print(f"pos_emb: {pos_emb(seq_len)}")
print(f"data: {data}")

print(f"pos_emb: {pos_emb(seq_len).shape}")




# data --> tokenized embedding


#     tokenized data            positional embeddings
# (batch, seq_len, d_model) + (batch, seq_len, d_model)

gpt_emb = GPTEmbeddings(vocab_size, d_model, seq_len, dropout, padding_idx)

print(gpt_emb(data))