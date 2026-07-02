"""
Project Candace

File: day01_tensor_fundamentals.py

Purpose:
Learn and experiment with PyTorch tensors.
This is the first coding file of Project Candace.
"""

import torch

scalar=torch.tensor(7)

print(scalar)
print("shape:",scalar.shape)
print("data type:",scalar.dtype)
print("dimensions:",scalar.ndim)

x=[1,2,3]
y=torch.tensor(x)
print(y)
print(y.shape)
print(y.ndim)

a=torch.tensor([
 [1,2,3],
 [4,5,6]
])
print(a.shape)
print(a.ndim)

print(torch.zeros(3,4))
print(torch.ones(2,3))

print(torch.randn(2,3))
r=torch.arange(12)
r1=torch.arange(12)
r[0]=100
r1[0]=100
print(r)
print(r.reshape(4,3))
print(r1.view(4,3))
print(r.reshape(4,3).T)
print(r.reshape(4,3))

print(r.unsqueeze(0))
print(r.squeeze)

A = torch.tensor([
    [1,2],
    [3,4]
])

B = torch.tensor([
    [5,6],
    [7,8]
])

print(A@B)

A1 = torch.tensor([
    [1,2,3],
    [4,5,6]
])

B2 = torch.tensor([10,20,30])

class MyLinear:
    def __init__(self,in_features,out_features):
        self.in_features=in_features
        self.out_features=out_features
        self.weight = torch.randn(self.in_features, self.out_features)
        self.bias=torch.randn(self.out_features)
    
    def forward(self,tensor):
        output=tensor@self.weight+self.bias
        return output

# -----------------------------
# Testing MyLinear
# -----------------------------

# Create a layer with 3 input features and 2 output features
layer = MyLinear(3, 2)

# Print initialized parameters
print("Weight Matrix:")
print(layer.weight)
print("Weight Shape:", layer.weight.shape)

print("\nBias:")
print(layer.bias)
print("Bias Shape:", layer.bias.shape)

# Create a random input tensor
x = torch.randn(1, 3)

print("\nInput Tensor:")
print(x)
print("Input Shape:", x.shape)

# Forward pass
output = layer.forward(x)

print("\nOutput:")
print(output)
print("Output Shape:", output.shape)