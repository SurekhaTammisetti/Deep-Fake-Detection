import os
import torch
import timm
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        ai_dir = os.path.join(root_dir, "AI-Generated Images")
        real_dir = os.path.join(root_dir, "Real Images")

        for img in os.listdir(ai_dir):
            self.samples.append((os.path.join(ai_dir, img), 1))  # AI = 1

        for img in os.listdir(real_dir):
            self.samples.append((os.path.join(real_dir, img), 0))  # Real = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
dataset_path = "/content/dataset/Human Faces Dataset"

dataset = FaceDataset(dataset_path, transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=1
).to(device)

# Ensure gradients are enabled
for p in model.parameters():
    p.requires_grad = True
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        imgs = imgs.to(device)
        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # -------- Accuracy --------
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Accuracy: {epoch_acc * 100:.2f}%"
    )

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total
val_acc = evaluate_accuracy(model, val_loader)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
SAVE_PATH = "/content/ai_generated_face_detector_vit_best.pth"

torch.save(model.state_dict(), SAVE_PATH)

print(f"Model saved successfully at: {SAVE_PATH}")
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
def predict_image(image_path, model, threshold=0.5):
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(img_tensor)
        prob = torch.sigmoid(logit).item()

    label = "AI Generated" if prob > threshold else "Real"

    return label, prob
image_path = "/content/real_face_photo.webp"
label, confidence = predict_image(image_path, model)

print("Prediction:", label)
print(f"Confidence: {confidence:.4f}")
