# WIDS-5.0-Endterm
# Convolutional Neural Networks (CNNs) in PyTorch  
This repository provides a clear, structured, and step‑by‑step implementation of a CNN using PyTorch, with major emphasis on CNN hyperparameters and how they affect model performance.
# What is a Convolutional Neural Network (CNN)?
A Convolutional Neural Network (CNN) is a deep learning model designed specifically for image like data.  
Unlike fully connected networks, CNNs:  
• Exploit spatial structure in images  
• Learn local patterns (edges, textures, shapes)  
• Use shared weights, reducing parameters  
CNNs are widely used in image classification, medical imaging, object detection, and pattern recognition.

# Environment Setup  
Required Libraries  
Install dependencies using:
```
pip install torch torchvision torchmetrics numpy matplotlib tqdm
```
Importing Libraries
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchmetrics import Accuracy
```
Why these libraries?  
• torch – Core deep learning framework  
• torchvision – Datasets & image utilities  
• DataLoader – Efficient batching  
• torchmetrics – Clean evaluation metrics  

# Dataset Loading & Preprocessing    
We use the MNIST dataset (handwritten digits 0–9, size 28×28).  
Key Hyperparameter: batch_size  
Batch size controls how many samples are processed before updating weights.  
```
batch_size = 64

train_dataset = torchvision.datasets.MNIST(
    root="dataset",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = torchvision.datasets.MNIST(
    root="dataset",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```
Why normalization (ToTensor)?  
It converts pixel values from [0,255] → [0,1], improving training stability.  

# Building the CNN Architecture  
This is the most important section, where most hyperparameters appear.
```
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```
# CNN Hyperparameters (Layer‑wise Explanation)  
🔹 Convolution Layer Hyperparameters  
    <img width="596" height="223" alt="Screenshot 2026-02-01 at 10 47 09 PM" src="https://github.com/user-attachments/assets/ad233478-a562-4c2d-bed3-7427992056fa" />   
Example:  
Increasing out_channels from 8 → 32 improves feature learning but increases computation.

🔹 Pooling Hyperparameters
 ```
nn.MaxPool2d(kernel_size=2, stride=2)
```
• Reduces feature map size• Adds translation invariance• Prevents overfitting

🔹 Fully Connected Layer
```
nn.Linear(16 * 7 * 7, 10)
```
• Converts extracted features into class scores   
• Output size = number of classes 

# Device Selection (CPU vs GPU)
```
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)
```
Using GPU significantly speeds up training for large models.  

# Loss Function & Optimizer
Loss Function
```
criterion = nn.CrossEntropyLoss()
```
Used for multi‑class classification.   

Optimizer Hyperparameters
```
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
<img width="448" height="106" alt="Screenshot 2026-02-01 at 10 48 41 PM" src="https://github.com/user-attachments/assets/d96538a4-7bba-45ab-a155-c0d7256f5540" />       
Lower lr → stable but slow learning    
Higher lr → fast but unstable  

# Training Loop
```
epochs = 10

for epoch in range(epochs):
    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
<img width="357" height="110" alt="Screenshot 2026-02-01 at 10 50 45 PM" src="https://github.com/user-attachments/assets/7243cef9-740b-454b-bd87-a872d497629f" />   

# Model Evaluation
```
accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        accuracy.update(preds, labels)

print("Test Accuracy:", accuracy.compute().item())
```
# Saving & Loading the Model
```
torch.save(model.state_dict(), "cnn_model.pth")

model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()
```
Hyperparameter Tuning Summary     
<img width="460" height="194" alt="Screenshot 2026-02-01 at 10 53 47 PM" src="https://github.com/user-attachments/assets/9ec18c59-d4ed-4bb3-af18-6814faedf5f5" />

