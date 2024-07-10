# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:03:31 2024

@author: TimOConnor
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import recall_score, matthews_corrcoef, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from utils import AddGaussianNoise, PixelDropout, UnevenIllumination, create_subset, get_simple_CNN

# Define custom dataloader class
class SkinLesionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, subdir in enumerate(['Benign', 'Malignant']):
            subdir_path = os.path.join(root_dir, subdir)
            for img_name in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = image.astype(np.float32) / 255.0  # Set all data between 0 and 1
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


# Define augmentation and preprocessing pipelines
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        AddGaussianNoise(mean=0, std=0.1),
        PixelDropout(drop_prob=0.2),
        UnevenIllumination(alpha_range=(0.5, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Train the model with early stopping and plot training curves
def train_model(model, device, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25, patience=10):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    best_acc = 0.0
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.cpu().item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation loss has decreased
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                epochs_no_improve = 0
            elif phase == 'val':
                epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel(' Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Classification Accuracy')

    plt.show()

    return model, best_loss, best_acc

def train_ResNet():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define datasets and transforms
    full_train_dataset = SkinLesionDataset(root_dir='raw_data/train', transform=None)
    test_dataset = SkinLesionDataset(root_dir='raw_data/test', transform=data_transforms['test'])

    # Split full training dataset into training and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Apply transforms after splitting
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['test']

    # Subsample the training dataset
    train_dataset = create_subset(train_dataset.dataset, fraction=1)

    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    model = get_simple_CNN()

    # # Freeze layers except the final fully connected layer to speed up training
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze the final two layers to increase trainable capacity
    # for name, param in model.named_parameters():
    #     if "layer4" in name:  # or "layer3" in name:
    #         param.requires_grad = True

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model, best_loss, best_acc = train_model(model, device, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=50, patience=15)

    # Save the model
    torch.save(model.state_dict(), 'resnet18_skin_lesions.pth')
    print(f"Training completed with best validation loss: {best_loss:.4f} and best validation accuracy: {best_acc:.4f}")

    # Evaluate the model on the test set
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    fp_paths = []
    fn_paths = []

    with torch.no_grad():
        for inputs, labels, paths in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] == 1 and labels[i] == 0:
                    fp_paths.append(paths[i])
                elif preds[i] == 0 and labels[i] == 1:
                    fn_paths.append(paths[i])

    confMat = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confMat, display_labels=["Benign", "Malignant"])
    disp.plot(cmap=plt.cm.Blues, colorbar=False)
    plt.title("Classification confusion matrix (Test data)")

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC: {auc:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    print("False Positive image paths:")
    for path in fp_paths:
        print(path)

    print("False Negative image paths:")
    for path in fn_paths:
        print(path)
        
    results = {}
    results['Acc': accuracy, 'Sens': sensitivity, 'TP':tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Spec': specificity, 'MCC': mcc, 'AUC', auc, 'FPs': list(fp_paths), 'FNs': list(fn_paths)]
    return results

if __name__ == '__main__':
    results = train_ResNet()





























