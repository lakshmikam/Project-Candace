import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        B, T, _ = x.shape

        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        mask = torch.tril(torch.ones(T, T)).to(scores.device)

        scores = scores.masked_fill(mask == 0, float("-inf"))
        

        weights = torch.softmax(scores, dim=-1)
        output = weights @ V

        output = output.transpose(1, 2)
        output = output.contiguous().view(B, T, self.d_model)
        output = self.out_proj(output)
        return output, weights
    
x = torch.randn(1, 5, 64)

mha = MultiHeadAttention(64, 8)

output, weights = mha(x)

print("Output shape:", output.shape)
print("Weights shape:", weights.shape)

print("\nAttention weights (Batch 0, Head 0):")
print(weights[0, 0])
        