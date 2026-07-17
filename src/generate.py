import torch

from model import GPT

block_size = 128

d_model = 256
n_heads = 8
n_layers = 6

dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(
    "../datasets/tiny_shakespeare.txt",
    "r",
    encoding="utf-8",
) as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )


model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    block_size=block_size,
    dropout=dropout,
).to(device)

model.load_state_dict(
    torch.load(
        "../checkpoints/best_gpt_model.pth",
        map_location=device,
    )
)

model.eval()

prompt = input("Prompt: ")

context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device,
)

generated = model.generate(
    context,
    max_new_tokens=500,
    temperature=0.8,
    top_k=40,
)

print()

print(
    decode(
        generated[0].tolist()
    )
)