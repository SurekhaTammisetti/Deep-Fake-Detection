# =========================================================
# STEP 0: INSTALL
# =========================================================
!pip install -q timm retina-face opencv-python albumentations tqdm scikit-learn

# =========================================================
# STEP 1: DOWNLOAD FACEFORENSICS++
# =========================================================
!wget -q https://kaldir.vc.in.tum.de/faceforensics_download_v4.py
!sed -i "s/_ = input('')/# _ = input('')/g" faceforensics_download_v4.py

# REAL
!python faceforensics_download_v4.py /kaggle/working/ffpp \
 -d original \
 -c c23 \
 -t videos \
 -n 200 \
 --server EU2

# FAKE
!python faceforensics_download_v4.py /kaggle/working/ffpp \
 -d Deepfakes \
 -c c23 \
 -t videos \
 -n 200 \
 --server EU2

# =========================================================
# STEP 2: IMPORTS
# =========================================================
import os, cv2, torch, timm
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from retinaface import RetinaFace
from torchvision import transforms

# =========================================================
# STEP 3: CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3

REAL_VIDEOS = "/kaggle/working/ffpp/original_sequences/youtube/c23/videos"
FAKE_VIDEOS = "/kaggle/working/ffpp/manipulated_sequences/Deepfakes/c23/videos"

FACE_DATA = "/kaggle/working/faces"
os.makedirs(FACE_DATA, exist_ok=True)

# =========================================================
# STEP 4: FACE ALIGNMENT
# =========================================================
def align_and_crop(img, facial_area, landmarks):
    x1,y1,x2,y2 = facial_area
    le = landmarks["left_eye"]
    re = landmarks["right_eye"]

    dx, dy = re[0]-le[0], re[1]-le[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((x1+x2)//2, (y1+y2)//2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, img.shape[1::-1])
    face = aligned[y1:y2, x1:x2]

    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))

# =========================================================
# STEP 5: FACE EXTRACTION
# =========================================================
def extract_faces(video_path, out_dir, every_n=10):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % every_n == 0:
            faces = RetinaFace.detect_faces(frame)
            if faces:
                face = max(
                    faces.values(),
                    key=lambda f:(f["facial_area"][2]-f["facial_area"][0]) *
                                 (f["facial_area"][3]-f["facial_area"][1])
                )
                crop = align_and_crop(frame, face["facial_area"], face["landmarks"])
                cv2.imwrite(f"{out_dir}/{saved:05d}.jpg", crop)
                saved += 1
        idx += 1

    cap.release()

# =========================================================
# STEP 6: PROCESS VIDEOS
# =========================================================
for cls, src_dir in [("real", REAL_VIDEOS), ("fake", FAKE_VIDEOS)]:
    dst_root = f"{FACE_DATA}/{cls}"
    os.makedirs(dst_root, exist_ok=True)

    for vid in tqdm(os.listdir(src_dir), desc=f"Extracting {cls} faces"):
        extract_faces(
            f"{src_dir}/{vid}",
            f"{dst_root}/{vid.replace('.mp4','')}"
        )

# =========================================================
# STEP 7: DATASET
# =========================================================
class FaceDataset(Dataset):
    def __init__(self, root):
        self.data = []
        for label, cls in enumerate(["real","fake"]):
            for vid in os.listdir(f"{root}/{cls}"):
                for img in os.listdir(f"{root}/{cls}/{vid}"):
                    self.data.append((f"{root}/{cls}/{vid}/{img}", label))

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        path,label = self.data[i]
        return self.tf(Image.open(path).convert("RGB")), label


