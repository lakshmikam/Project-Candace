import os
import torch
import matplotlib.pyplot as plt

from model import GPT

# -----------------------------
# Hyperparameters
# -----------------------------

batch_size = 64
block_size = 128

max_iters = 5000
eval_interval = 200
eval_iters = 100

learning_rate = 3e-4

d_model = 256
n_heads = 8
n_layers = 6

dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

# -----------------------------
# Load Dataset
# -----------------------------

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
    return "".join([itos[i] for i in l])


data = torch.tensor(
    encode(text),
    dtype=torch.long,
)

# -----------------------------
# Train / Validation Split
# -----------------------------

n = int(0.9 * len(data))

train_data = data[:n]

val_data = data[n:]


# -----------------------------
# Batch Loader
# -----------------------------

def get_batch(split):

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,),
    )

    x = torch.stack(
        [
            data[i:i + block_size]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + block_size + 1]
            for i in ix
        ]
    )

    return (
        x.to(device),
        y.to(device),
    )


# -----------------------------
# Model
# -----------------------------

model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    block_size=block_size,
    dropout=dropout,
).to(device)

print(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
)


start_step = 0
best_val_loss = float("inf")

checkpoint_path = "../checkpoints/checkpoint.pth"

if os.path.exists(checkpoint_path):

    checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_step = checkpoint["step"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    print(f"Resuming from Step {start_step}")


# -----------------------------
# Validation
# -----------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    out = {}

    for split in [
        "train",
        "val",
    ]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            _, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


train_losses = []
val_losses = []

os.makedirs("../checkpoints", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

for step in range(start_step, max_iters):

    xb, yb = get_batch("train")

    optimizer.zero_grad()

    _, loss = model(xb, yb)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()

    if step % eval_interval == 0:

        losses = estimate_loss()

        train_loss = losses["train"].item()
        val_loss = losses["val"].item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Step {step:5d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
    model.state_dict(),
    "../checkpoints/best_gpt_model.pth",
)

        # Save resumable checkpoint EVERY evaluation
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            "../checkpoints/checkpoint.pth",
        )
torch.save(
    model.state_dict(),
    "../checkpoints/final_gpt_model.pth",
)
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Evaluation")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/loss_curve.png")
plt.show()

print("\nTraining Complete!")
print(f"Best Validation Loss: {best_val_loss:.4f}")