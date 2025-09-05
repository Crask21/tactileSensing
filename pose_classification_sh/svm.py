import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# === CONFIG ===
TRAIN_DIR = "pose_classification_sh/train"
VAL_DIR = "pose_classification_sh/val"
DATASET_DIR = "DataCollection/sh_gs_processed"
BATCH_SIZE = 32
NUM_CLASSES = 4  # Change if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Define transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

# --- 2. Load datasets ---
# train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
# val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

# Load the full dataset from DATASET_DIR
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

# Split indices for train/val
indices = np.arange(len(full_dataset))
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, stratify=[full_dataset.samples[i][1] for i in indices], random_state=42
)

# Subset the dataset
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# For compatibility with .classes attribute
train_dataset.classes = full_dataset.classes
val_dataset.classes = full_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Load pretrained ResNet18 for feature extraction ---
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # Remove final classification layer
model = model.to(DEVICE)
model.eval()

def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(DEVICE)
            feats = model(imgs)  # Shape: (batch, 512)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# --- 4. Extract features ---
print("Extracting train features...")
X_train, y_train = extract_features(train_loader)

print("Extracting val features...")
X_val, y_val = extract_features(val_loader)

# --- 5. Train SVM ---
print("Training SVM classifier...")
svm_clf = SVC(kernel='linear', C=1.0)  # You can change kernel='rbf' etc.
svm_clf.fit(X_train, y_train)

# --- 6. Evaluate ---
y_pred = svm_clf.predict(X_val)
print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred, target_names=val_dataset.classes))

print("Performance of SVM on data where shadowhand grasps the gelsights")
print(f"Train image count: {X_train.shape[0]}")
print(f"Val image count: {X_val.shape[0]}")
print("\nConfusion Matrix for following classes:")
print(val_dataset.classes)
print(confusion_matrix(y_val, y_pred))

print("\nAccuracy:", np.mean(y_pred == y_val))
