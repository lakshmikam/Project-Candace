import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
scalar=torch.tensor([1,2,3,-4,-5,0])
newscalar=torch.maximum(scalar,torch.tensor(0))
print(newscalar)


def relu(x):
    y=torch.tensor(0)
    x=torch.maximum(x,y)
    return x

def sigmoid(x):
    return (1/(1+torch.exp(-x)));

def softmax(x):

    y=torch.exp(x-torch.max(x))
    total=torch.sum(y)
    return y/total
x = torch.tensor([1, 2, 3, -4, -5, 0])
a=torch.tensor([-1000.0, 1000.0])

output1 = relu(x)
output2 = sigmoid(x)
output3=softmax(a)
print(output1)
print(output2)
print(output3)

class MyLinear:
    def __init__(self,in_features,out_features):
        self.in_features=in_features
        self.out_features=out_features
        self.weight = torch.randn(self.in_features, self.out_features)
        self.bias=torch.randn(self.out_features)
    
    def forward(self,tensor):
        output=tensor@self.weight+self.bias
        return output
class MLP:
    def __init__(self):
        self.layer1 = MyLinear(784, 128)
        self.layer2 = MyLinear(128, 64)
        self.layer3 = MyLinear(64, 10)

    def forward(self, x):

        x = x.reshape(784)

        x = self.layer1.forward(x)
        x = relu(x)

        x = self.layer2.forward(x)
        x = relu(x)

        x = self.layer3.forward(x)

        return x

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./datasets",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./datasets",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)

        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.layer3(x)

        return x

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5
losses = []

for epoch in range(epochs):

    model.train()

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(8,5))

plt.plot(losses)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.show()