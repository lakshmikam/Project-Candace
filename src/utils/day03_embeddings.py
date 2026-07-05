import torch
import torch.nn as nn

class MyEmbedding(nn.Module):
    def __init__(self,vocab_size,embed_dim):
        super().__init__()
        param=torch.randn(vocab_size,embed_dim)
        

        self.embedding_table=nn.Parameter(param)

    def forward(self,token_ids):
        return self.embedding_table[token_ids]

MyEmbed=MyEmbedding(1000,128)
torch_embedding = nn.Embedding(1000, 128)

token_ids = torch.tensor([5, 18, 902, 71])

with torch.no_grad():
    MyEmbed.embedding_table.copy_(torch_embedding.weight)

my_output = MyEmbed(token_ids)
torch_output = torch_embedding(token_ids)

print(torch.allclose(my_output,torch_output))

file = open("datasets/input.txt")
text = file.read()
file.close()
chars=sorted(set(text))
enumerate(chars)

char_to_idx={}
idx_to_char={}
for index, char in enumerate(chars):
    char_to_idx[char]=index
    idx_to_char[index]=char

token_ids = [char_to_idx[char] for char in text]

print(text[:20])
print(token_ids[:20])

new_token=torch.tensor(token_ids)

embedding_output=MyEmbed(new_token)
print(new_token.shape)
print(embedding_output.shape)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Take only the first 200 embeddings
sample_embeddings = embedding_output[:200]

# Convert from PyTorch tensor to NumPy array
sample_embeddings = sample_embeddings.detach().numpy()

# Reduce 128D -> 2D
pca = PCA(n_components=2)
embedding_2d = pca.fit_transform(sample_embeddings)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])

# Label each point with its corresponding character
for i in range(200):
    plt.text(
        embedding_2d[i, 0],
        embedding_2d[i, 1],
        text[i],
        fontsize=8
    )

plt.title("Character Embeddings (Random Initialization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
