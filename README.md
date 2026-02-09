### Deep-Fake-Detection

### Software and Tools Requirements

1. [GithubAccount](https://github.com)
2. [HerokuAccount](https://heroku.com)
3. [VSCodeIDE](https://code.visualstudio.com/)
4. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

Create a new Environment

```
conda create -p venv python==3.7 -y
```

# Deep Fake Detection System

A Deep Learning–powered web app that detects whether an image is:

✔ **AI-generated**  
✔ **Face-swap / deepfake**  
✔ **Real human photo**

Built using:
- Vision Transformer (ViT) for AI-generated face detection
- Swin Transformer for face-swap detection
- RetinaFace for face cropping
- Flask for web server

---

## 🔍 Features

**Detects**
- GAN-generated images
- Face-swap deepfakes
- Edited/forensic face images

**How it works**
1. Upload an image
2. ViT analyzes the full image for AI generation
3. RetinaFace extracts facial region
4. Swin model predicts face swap
5. Combined prediction shown to user

---

## 🚀 Setup

### 1. Clone the repo

```bash
git clone https://github.com/SurekhaTammisetti/Deep-Fake-Detection.git
cd Deep-Fake-Detection
