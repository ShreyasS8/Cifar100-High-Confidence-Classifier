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
from tqdm import tqdm

# print("started")

FILEPATH_MODEL = sys.argv[1]
FILEPATH_TEST = sys.argv[2]
alpha = float(sys.argv[3])
gamma = float(sys.argv[4])

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using Device: {device}")

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

class CIFAR100Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if isinstance(image, torch.Tensor):
            image = image.numpy()
            image = np.transpose(image, (1, 2, 0))
            image = Image.fromarray((image * 255).astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label

test_dataset = CIFAR100Dataset(FILEPATH_TEST, transform=transform_test)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

from collections import defaultdict


def test_model_save_predictions(model, test_loader, device, alpha=0.99, temperature=1.0,
                                output_file='test_predictions_with_confidence.csv'):
    model.eval()
    predictions = []
    low_confidence_count = 0
    confident_predictions = defaultdict(list)

    with torch.no_grad():
        for idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)

            outputs = model(inputs)

            scaled_logits = outputs / temperature

            probabilities = F.softmax(scaled_logits, dim=1)

            confidences, predicted = torch.max(probabilities, 1)

            for i in range(inputs.size(0)):
                confidence = confidences[i].item()
                predicted_class = predicted[i].item() if confidence >= alpha else -1

                if predicted_class == -1:
                    low_confidence_count += 1
                else:
                    confident_predictions[predicted_class].append(confidence)

                predictions.append({
                    'ID': idx * test_loader.batch_size + i,
                    'Predicted_label': predicted_class
                })

    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)

    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)

    # print(f"Total number of low confidence predictions (-1): {low_confidence_count}")
    # print(f"Total number of high confidence predictions (-1): {20000-low_confidence_count}")

num_classes = 100
model = WideResNet(depth=34, width=16, num_classes=num_classes).to(device)


filename = FILEPATH_MODEL
model.load_state_dict(torch.load(filename))    

temperatures = [2.95]
for T in temperatures:
    # print(f"Testing with temperature: {T}")
    test_model_save_predictions(model, test_loader, device, alpha=0.99, temperature=T, output_file=f'submission.csv')
