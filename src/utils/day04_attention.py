import torch
import math

def self_attention(seq_len,d_model):


    X=torch.randn(seq_len,d_model)

    Wq=torch.randn(d_model,d_model)
    Wk=torch.randn(d_model,d_model)
    Wv=torch.randn(d_model,d_model)

    Q=X@Wq
    K=X@Wk
    V=X@Wv

    scores=(Q @ K.T)/math.sqrt(d_model)

    weights = torch.softmax(scores, dim=-1)

    output=weights @ V

    print("X shape:", X.shape)
    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    print("Scores shape:", scores.shape)
    print("Weights shape:", weights.shape)
    print("Output shape:", output.shape)

    print(weights.sum(dim=-1))

    assert output.shape == X.shape
    assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(seq_len),
    atol=1e-6
)

    return output

self_attention(100, 64)


