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

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V

        output = output.transpose(1, 2)
        output = output.contiguous().view(B, T, self.d_model)
        output = self.out_proj(output)
        return output
    

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1=nn.Linear(d_model,4*d_model)
        self.gelu=nn.GELU()
        self.fc2=nn.Linear(4*d_model,d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.ln1=nn.LayerNorm(self.d_model)
        self.attn=MultiHeadAttention(d_model, n_heads)
        self.ln2=nn.LayerNorm(self.d_model)
        self.ff=FeedForward(d_model)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        y = self.attn(x)
        x = residual + y

        residual = x
        x = self.ln2(x)
        y = self.ff(x)
        x = residual + y

        return x

