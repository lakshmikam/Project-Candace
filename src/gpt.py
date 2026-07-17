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

        B, T, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        output = weights @ V

        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)

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
        
class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        block_size,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            d_model,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        pos = torch.arange(T, device=idx.device)

        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            logits_flat = logits.view(-1, logits.size(-1))

            targets_flat = targets.view(-1)

            loss = nn.functional.cross_entropy(
                logits_flat,
                targets_flat,
            )

        return logits, loss
    
model = GPT(
    vocab_size=vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=2,
    block_size=64,
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)


for step in range(200):

    xb, yb = get_batch("train")

    optimizer.zero_grad()

    logits, loss = model(xb, yb)

    loss.backward()

    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | Loss = {loss.item():.4f}")