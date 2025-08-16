import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.optim as optim
from collections import Counter


train_dir = "pose_classification/train"
val_dir = "pose_classification/val"




def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load pretrained model
    model = models.resnet18(pretrained=True)

    # Replace final layer for 4 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_list = []
    for epoch in range(100):  # number of epochs
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
        
        loss_list.append(running_loss / len(train_loader))

        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()
    # Save model
    torch.save(model.state_dict(), "pose_classification/classifier.pth")

def validate_dataset_via_torch(train_dir, val_dir):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Class mapping
    print("\n📂 Class-to-Index Mapping:")
    for cls_name, cls_idx in train_dataset.class_to_idx.items():
        print(f"  {cls_name}: {cls_idx}")

    # Check that train and val have the same classes
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print("\n⚠️ WARNING: Train and Val class mappings differ!")
        print(f"Train classes: {train_dataset.classes}")
        print(f"Val classes:   {val_dataset.classes}")

    # Count images per class
    train_counts = Counter([label for _, label in train_dataset.samples])
    val_counts = Counter([label for _, label in val_dataset.samples])

    print("\n📊 Image Counts per Class:")
    for cls_name, cls_idx in train_dataset.class_to_idx.items():
        print(f"  {cls_name}: Train={train_counts[cls_idx]}, Val={val_counts[cls_idx]}")

    print("\n✅ Dataset loaded successfully via torchvision!")