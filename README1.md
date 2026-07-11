# WIDS-5.0-midterm 
This repository contains my learning and practice code from the WIDS 5.0 program.
It contains PyTorch tensors and then implementing neural network with PyTorch 

# PyTorch Tensors
Here is Tensors basic 

```
import torch
```
# Creating tensors
```
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
```
Scalar → 0D tensor (single number)  
Vector → 1D tensor  
Matrix → 2D tensor  
```
print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:", matrix)
```

# Tensor properties
```
print("Shape of vector:", vector.shape)
print("Datatype:", vector.dtype)

```
shape → tells dimensions  
dtype → tells data type (int64, float32, etc.)

# Tensor Attributes
```
import torch

x = torch.randn(3, 4)

print("Tensor:", x)
print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Data type:", x.dtype)
print("Device:", x.device)
```
random number  
shape → size in each dimension  
ndim → number of dimensions  
dtype → float32 / float64  
device → CPU or GPU  

# Tensor Initialization
```
import torch

# Zeros
a = torch.zeros(3, 3)

# Ones
b = torch.ones(2, 4)

# Random values (normal distribution)
c = torch.randn(3, 3)

# Random values (uniform distribution)
d = torch.rand(3, 3)

print(a)
print(b)
print(c)
print(d)
```
zeros → bias initialization  
ones → testing / debugging  
randn → normal distribution (used in weights)  
rand → uniform distribution  
# Tensor Operations
```
import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([4, 5, 6], dtype=torch.float32)

# Addition
print(x + y)

# Subtraction
print(x - y)

# Element-wise multiplication
print(x * y)

# Element-wise division
print(x / y)
```
These are element-wise operations  
Shape of x and y must match  
Used heavily in loss computation & optimization  
# Tensor Multiplication
```
import torch

A = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float32)

B = torch.tensor([[5, 6],
                  [7, 8]], dtype=torch.float32)
```
```
print(A * B)
```
Multiplies corresponding elements  
NOT matrix multiplication
```
tensor([[ 5., 12.],
        [21., 32.]])
```
```
print(torch.matmul(A, B))
```
OR
```
print(A @ B)
```
```
tensor([[19., 22.],
        [43., 50.]])
```
Uses linear algebra rule
Used in neural networks:
    ```
    output = input @ weights + bias
    ```
# Dot Product 
```
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])

dot = torch.dot(x, y)
print(dot)
```
Dot product = weighted sum
# Reshaping & View
```
x = torch.arange(1, 7)

x_reshaped = x.view(2, 3)
print(x_reshaped)
```
Output
```
tensor([[1, 2, 3],
        [4, 5, 6]])
```

view() reshapes tensor  
No data copied, just reinterpreted  
Used to flatten images before fully connected layers  
# NumPy ↔ PyTorch Bridge
```
import torch
import numpy as np

# NumPy array
np_array = np.array([1, 2, 3])

# Convert NumPy → Tensor
torch_tensor = torch.from_numpy(np_array)

print(torch_tensor)
```
output 
```
tensor([1, 2, 3])
```
Tensor → NumPy
```
tensor = torch.tensor([4, 5, 6])

np_from_tensor = tensor.numpy()
print(np_from_tensor)
```
output
```
array([4, 5, 6])
```
They share memory  
Changing one changes the other
```
np_array[0] = 100
print(torch_tensor)
```
output
```
tensor([100,   2,   3])
```

# Neural Network (nn)
Below given   
# What is a Neuron?
```
import torch

# Input features
x = torch.tensor([1.0, 2.0])

# Weights
w = torch.tensor([0.5, -1.0])

# Bias
b = torch.tensor(0.2)

# Linear combination
z = torch.dot(x, w) + b

print("z (before activation):", z)
```
Output
```
z (before activation): tensor(-1.3)
```
z = (1.0 × 0.5) + (2.0 × -1.0) + 0.2  
  = 0.5 - 2.0 + 0.2  
  = -1.3  
This is exactly one neuron.  
# Activation Function(ReLU)
```
import torch

z = torch.tensor(-1.3)

output = torch.relu(z)

print("After ReLU:", output)
```
Output
```
After ReLU: tensor(0.)
```
ReLU(z) = max(0, z)  
Negative → 0  
Positive → unchanged  
Used to introduce non-linearity.  
# Single Neuron = Linear + ReLU
```
import torch

x = torch.tensor([1.0, 2.0])
w = torch.tensor([0.5, -1.0])
b = torch.tensor(0.2)

z = torch.dot(x, w) + b
output = torch.relu(z)

print("Final output:", output)
```
# Neural Network using PyTorch
```
import torch
import torch.nn as nn

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # FC layer

    def forward(self, x):
        return torch.relu(self.fc(x))

# Create model
model = SimpleNN()

# Input
x = torch.tensor([[1.0, 2.0]])

# Forward pass
output = model(x)

print("Model output:", output)
```
Output
```
Model output: tensor([[0.3174]], grad_fn=<ReluBackward0>)
```
NOTE: The answer is random as it depend on weights and bias which are randomly initialized and vary as many times as the code is run   
nn.Linear(2,1) → Fully Connected layer  
PyTorch handles weights & bias  
grad_fn → gradients are being tracked  
# Loss Function (How wrong is the model?)
```
import torch
import torch.nn as nn

y_true = torch.tensor([[1.0]])
y_pred = torch.tensor([[0.3]])

loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)

print("Loss:", loss)
```
Output
```
Loss: tensor(0.4900)
```
(y_true - y_pred)²  
= (1.0 - 0.3)²  
= 0.49  
Loss tells how bad the prediction is.
# Backpropagation
```
import torch

# Input with gradient tracking
x = torch.tensor(2.0, requires_grad=True)

# Simple operation
y = x ** 2

# Backward pass
y.backward()

print("Gradient dy/dx:", x.grad)
```
Output
```
Gradient dy/dx: tensor(4.)
```
y = x²
dy/dx = 2x
dy/dx at x=2 → 4  
PyTorch computed this automatically.
# Backpropagation in Neural Network
```
import torch
import torch.nn as nn

# Model
model = nn.Linear(1, 1)

# Data
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# Loss
criterion = nn.MSELoss()

# Forward pass
y_pred = model(x)
loss = criterion(y_pred, y)

print("Loss before backward:", loss)

# Backward pass
loss.backward()

print("Gradient of weight:", model.weight.grad)
print("Gradient of bias:", model.bias.grad)
```
Output
```
Loss before backward: tensor(8.5342, grad_fn=<MseLossBackward0>)
Gradient of weight: tensor([[-18.2134]])
Gradient of bias: tensor([-8.4211])
```
Gradients show how to update weights  
Used by optimizers (SGD, Adam)
# Training Loop
```
import torch
import torch.nn as nn

model = nn.Linear(1, 1)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
Output
```
Epoch 1, Loss: 8.5342
Epoch 2, Loss: 6.1128
Epoch 3, Loss: 4.3841
...
Epoch 10, Loss: 0.9123
```
Each epoch:  
Forward pass  
Loss calculation  
Backpropagation  
Weight update  
Loss ↓ means learning is happening  

# Kaggle contest practice
Here is the code I practiced for the kaggle contest in which we are asked to-  
In that competition, we had to build a digit classifier for MNIST using a simple idea:  

Images that look similar should have similar labels.  
The task is to implement a k-Nearest Neighbors (KNN) classifier for handwritten digits.  

There is no training phase. There are no learned parameters. Just data, distance, and decisions.  
Info was given:  

A training set of MNIST digits (60,000 × 784)  
A test set of MNIST digits (10,000 × 784)  
For each test image:  

Compute its distance to all training images  
Find the nearest neighbor (or top-k neighbors)  
Predict the digit based on those neighbors  
```
import numpy as np
import pandas as pd

train = pd.read_csv("/kaggle/input/wi-ds-dl-competition-1/train.csv")
test  = pd.read_csv("/kaggle/input/wi-ds-dl-competition-1/test.csv")

X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

X_test = test.values

X_train = X_train / 255.0
X_test = X_test / 255.0

batch_size = 500
predictions = []

for start in range(0, X_test.shape[0], batch_size):
    end = start + batch_size
    X_batch = X_test[start:end]

    dists = (
        np.sum(X_batch**2, axis=1).reshape(-1, 1)
        + np.sum(X_train**2, axis=1)
        - 2 * np.dot(X_batch, X_train.T)
    )

    nearest = np.argmin(dists, axis=1)
    predictions.extend(y_train[nearest])

    print(f"Processed {len(predictions)} samples")

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predictions) + 1),
    "Label": predictions
})

submission.to_csv("submission.csv", index=False)

print(X_test.shape)
submission.shape
print(len(predictions))
print("Using predictions length:", len(predictions))
```

Developed by: Anshikaydv
