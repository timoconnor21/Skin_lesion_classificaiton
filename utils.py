# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:27:16 2024

@author: TimOConnor
"""
"""
Utils file for accessory functions
"""

import numpy as np
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset
from albumentations.core.transforms_interface import ImageOnlyTransform


# Define custom augmentations

#Additive gauassian noise augmentation to add noise to images
def addGausNoise(image, mean_val=0, var_val=0.0001):
    row, col, ch = np.shape(image)
    sigma = np.sqrt(var_val)
    gauss = np.random.normal(mean_val, sigma, (row, col, ch))
    image = image + gauss
    return image

class AddGaussNoise(ImageOnlyTransform):   
    def __init__(self, mean_val=0, var_val=0.01, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.mean_val = mean_val
        self.var_val = var_val   
            
    def apply(self, image, **params):
        if np.random.rand() < self.p:  # Apply with probability self.p
            return addGausNoise(image, self.mean_val, self.var_val)
        return image
    
    def get_params_dependent_on_targets(self, params):
        return {"mean_val": self.mean_val, "var_val": self.var_val}
    
    @property
    def targets_as_params(self):
        return ["image"]
    
    def get_transform_init_args_names(self):
        return ("mean_val", "var_val")
    
    
# Function to randomly select and display images
def show_example_images(image_ids, data_dir, category, n=10, plt_title='Example Images'):
    selected_images = random.sample(image_ids, min(len(image_ids), n))
    plt.figure(figsize=(20, 10))
    plt.suptitle(plt_title, fontsize=32)

    for i, img_id in enumerate(selected_images):
        img_path = os.path.join(data_dir, 'test', category, img_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f'{category}: {img_id}')
        plt.axis('off')
    plt.show()

    
# Define a simple CNN model for comparison to ResNet
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def get_simple_CNN(num_classes=2):
    return SimpleCNN(num_classes=num_classes)


# Custom dataset class for handling TTA (test-time augmentation)
class TTADataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, num_augmentations=5):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.num_augmentations = num_augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = image.astype(np.float32) / 255.0  # Set all data between 0 and 1
        label = self.labels[idx]

        if self.transform:
            augmented_imgs = []
            for i in range(self.num_augmentations):
                augmented_img = self.transform(image=image)["image"]
                augmented_imgs.append(augmented_img)

            return augmented_imgs, label, img_path
        else:
            return image, label, img_path


# Function to perform TTA and aggregate predictions
def tta_predictions(model, tta_loader, device):
    all_labels = []
    all_preds = []
    all_probs = []
    fp_paths = []
    fn_paths = []

    model.eval()
    with torch.no_grad():
        for inputs, labels, paths in tta_loader:
            inputs = [inp.to(device) for inp in inputs]
            labels = [lbl.to(device) for lbl in labels] 
                
            outputs = torch.stack([model(inp) for inp in inputs])
            outputs_avg = torch.mean(outputs, dim=0)
            preds = torch.argmax(outputs_avg, dim=1)
            probs = nn.functional.softmax(outputs_avg, dim=1)[:, 1]

            all_labels.extend([lbl.cpu().numpy() for lbl in labels])
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] == 1 and labels[i] == 0:
                    fp_paths.append(paths[i])
                elif preds[i] == 0 and labels[i] == 1:
                    fn_paths.append(paths[i])


    return all_labels, all_preds, all_probs, fp_paths, fn_paths