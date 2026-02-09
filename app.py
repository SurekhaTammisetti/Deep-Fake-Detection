import os
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from retinaface import RetinaFace

# =========================
# Flask setup
# =========================
app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Transforms
# =========================
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

swin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# Load ViT (AI-generated detector)
# (NO face extraction – full image)
# =========================
vit_model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=1
)
vit_model.load_state_dict(
    torch.load(
        "models/ai_generated_face_detector_vit_best.pth",
        map_location=DEVICE
    )
)
vit_model.to(DEVICE).eval()
print("✅ ViT AI-generated model loaded")

# =========================
# Swin model class (MUST MATCH TRAINING)
# =========================
class SwinDeepFake(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0
        )
        self.head = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

# =========================
# Load Swin (Face-swap detector)
# =========================
swin_model = SwinDeepFake()
swin_model.load_state_dict(
    torch.load(
        "models/swin_deepfake_final.pth",
        map_location=DEVICE
    )
)
swin_model.to(DEVICE).eval()
print("✅ Swin face-swap model loaded")

# =========================
# Face extraction (ONLY for Swin)
# =========================
def extract_face(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = RetinaFace.detect_faces(img)

    if not faces:
        return None

    # pick largest face
    face = max(
        faces.values(),
        key=lambda f: (f["facial_area"][2] - f["facial_area"][0]) *
                      (f["facial_area"][3] - f["facial_area"][1])
    )

    x1, y1, x2, y2 = face["facial_area"]
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)

# =========================
# Prediction functions
# =========================
@torch.no_grad()
def predict_vit(img):
    x = vit_transform(img).unsqueeze(0).to(DEVICE)
    logit = vit_model(x)
    return torch.sigmoid(logit).item()

@torch.no_grad()
def predict_swin(face_img):
    x = swin_transform(face_img).unsqueeze(0).to(DEVICE)
    logits = swin_model(x)
    return F.softmax(logits, dim=1)[0, 1].item()

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict-image", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files["image"]).convert("RGB")

    # ---- ViT: full image ----
    ai_prob = predict_vit(image)

    # ---- Swin: face only ----
    face = extract_face(image)
    if face is None:
        return jsonify({
            "error": "❌ No face detected. Please upload a clear face image."
        }), 400

    face_prob = predict_swin(face)

    # ---- Final decision ----
    if ai_prob > 0.5:
        label = "AI GENERATED"
    elif face_prob > 0.5:
        label = "FACE SWAPPED"
    else:
        label = "REAL"

    return jsonify({
        "label": label,
        "ai_prob": f"{ai_prob:.4f}",
        "face_prob": f"{face_prob:.4f}"
    })

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
