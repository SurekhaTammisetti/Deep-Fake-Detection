import zipfile

zip_path = "/content/faces.zip"
extract_path = "/content"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Unzipped successfully!")
# ================================
# STEP 0: IMPORTS & SETUP
# ================================
import os
import torch
import timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 15
DATA_ROOT = "/content/content/faces"

print("Using device:", DEVICE)


# ================================
# STEP 1: DATASET CLASS
# ================================
class FaceDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for label, cls in enumerate(["real", "fake"]):
            cls_path = os.path.join(root, cls)
            if not os.path.exists(cls_path):
                raise ValueError(f"Missing folder: {cls_path}")

            for vid in os.listdir(cls_path):
                vid_path = os.path.join(cls_path, vid)
                if not os.path.isdir(vid_path):
                    continue

                for img in os.listdir(vid_path):
                    if img.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.samples.append(
                            (os.path.join(vid_path, img), label)
                        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        print(f"Total samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

# ================================
# STEP 2: LOAD + SPLIT DATA
# ================================
dataset = FaceDataset(DATA_ROOT)

train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples:   {len(val_ds)}")
# ================================
# STEP 3: MODEL (SWIN TRANSFORMER)
# ================================
class SwinDeepFake(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        self.head = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

model = SwinDeepFake().to(DEVICE)

# Freeze backbone (recommended first)
for p in model.backbone.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.head.parameters(), lr=2e-4)



# ================================
# STEP 4: TRAIN & VALIDATE
# ================================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    train_correct = train_total = 0

    for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_correct += (out.argmax(1) == y).sum().item()
        train_total += y.size(0)

    train_acc = train_correct / train_total

    # ---- VALIDATE ----
    model.eval()
    val_correct = val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val   Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/content/swin_deepfake_best.pth")
        print("✅ Best model saved")

# ================================
# STEP 5: FINAL SAVE
# ================================
torch.save(model.state_dict(), "/content/swin_deepfake_final.pth")
print("🎉 Training complete")



