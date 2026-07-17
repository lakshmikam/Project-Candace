import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        mask = torch.tril(
            torch.ones(T, T, device=x.device)
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)

        weights = self.dropout(weights)

        out = weights @ V

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, self.d_model)
        )

        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(
            d_model,
            dropout,
        )

    def forward(self, x):

        x = x + self.attn(self.ln1(x))

        x = x + self.ff(self.ln2(x))

        return x


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        block_size,
        dropout=0.2,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
        )

    def forward(
        self,
        idx,
        targets=None,
    ):

        B, T = idx.shape

        pos = torch.arange(
            T,
            device=idx.device,
        )

        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):

        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    top_k,
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            probs = F.softmax(
                logits,
                dim=-1,
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1,
            )

        return idx