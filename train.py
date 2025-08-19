# %% Import necessary libraries
import os
import sys
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using Device: {device}")

FILEPATH_TRAIN = sys.argv[1]
alpha = float(sys.argv[2])
gamma = float(sys.argv[3])

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace=True)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Custom dataset class
class CIFAR100Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # Convert the image to a PIL Image if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()  # Convert to numpy array
            image = np.transpose(image, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
            image = Image.fromarray((image * 255).astype(np.uint8))  # Convert to PIL Image

        if self.transform:
            image = self.transform(image)

        return image, label

# Load datasets
train_dataset = CIFAR100Dataset(FILEPATH_TRAIN, transform=transform_train)
# test_dataset = CIFAR100Dataset('test.pkl', transform=transform_test)

# Split the dataset into training and validation
train_data  = train_dataset

# Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# WideResNet implementation without pretrained weights
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out += self.shortcut(x)
        else:
            out += x
        return F.relu(out)

class WideResNet(nn.Module):
    def __init__(self, depth, width, num_classes):
        super(WideResNet, self).__init__()
        self.in_channels = 16
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16 * width, n)
        self.layer2 = self._make_layer(BasicBlock, 32 * width, n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64 * width, n, stride=2)
        self.fc = nn.Linear(64 * width, num_classes)

    def _make_layer(self, block, out_channels, n, stride=1):
        shortcut = None
        if stride != 1 or self.in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, shortcut))
        self.in_channels = out_channels
        for _ in range(1, n):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Instantiate the model, criterion, optimizer, and scheduler
num_classes = 100  # CIFAR-100 has 100 classes
model = WideResNet(depth=34, width=16, num_classes=num_classes).to(device)  # 28 layers, width factor of 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # Decay LR by 0.9 each epoch

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Step the scheduler
        scheduler.step()

    torch.save(model.state_dict(), "model.pth")
    return train_losses

# Train the model
num_epochs = 35
train_losses = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)