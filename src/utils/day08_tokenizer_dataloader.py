import torch

with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    token_ids = []
    for ch in text:
        token_ids.append(stoi[ch])
    return token_ids

def decode(token_ids):
    chars = []
    for token_id in token_ids:
        chars.append(itos[token_id])
    return "".join(chars)

token_ids = encode(text)
data = torch.tensor(token_ids, dtype=torch.long)

decoded_text = decode(token_ids)
print(decoded_text == text)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size, block_size):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y