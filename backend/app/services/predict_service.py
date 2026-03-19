# services/predict_service.py
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import pydicom
import os

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# MODEL ARCHITECTURE — khớp đúng với notebook training
# =========================================================
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        backbone = models.resnet50(weights=None)

        # Backbone: tất cả trừ avgpool & fc
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # GAP + GMP – concat -> 2048*2 = 4096
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),          # head.0
            nn.Linear(4096, 512),         # head.1
            nn.BatchNorm1d(512),          # head.2
            nn.ReLU(inplace=True),        # head.3
            nn.Dropout(dropout / 2),      # head.4
            nn.Linear(512, num_classes),  # head.5
        )

    def forward(self, x):
        feat = self.features(x)
        gap  = self.gap(feat).flatten(1)
        gmp  = self.gmp(feat).flatten(1)
        return self.head(torch.cat([gap, gmp], dim=1))


model = ResNet50Classifier(num_classes=2, dropout=0.5)

# =========================================================
# LOAD CHECKPOINT
# =========================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/best_resnet50.pth")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# =========================================================
# TRANSFORM
# =========================================================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]

# =========================================================
# CLASS NAMES — đúng thứ tự từ notebook: index 0=normal, 1=lung_opacity
# =========================================================
CLASSES = ["Bình Thường", "Mờ Phổi"] 

# =========================================================
# READ IMAGE (DICOM + JPG/PNG)
# =========================================================
def read_image(file) -> Image.Image:
    if isinstance(file, str):
        return Image.open(file).convert("RGB")

    file.seek(0)
    try:
        dicom = pydicom.dcmread(file, force=True)
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = (image * 255).astype(np.uint8)
        if getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1":
            image = np.max(image) - image
        return Image.fromarray(image).convert("RGB")
    except Exception:
        file.seek(0)
        return Image.open(file).convert("RGB")

# =========================================================
# PREDICT
# =========================================================
def predict_image(image_file, use_tta: bool = True) -> dict:
    image = read_image(image_file)

    with torch.no_grad():
        if use_tta:
            all_probs = []
            for tfm in tta_transforms:
                tensor = tfm(image).unsqueeze(0).to(device)
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)
            avg_probs = torch.stack(all_probs).mean(dim=0)  # (1, 2)
        else:
            tensor = infer_transform(image).unsqueeze(0).to(device)
            outputs = model(tensor)
            avg_probs = torch.softmax(outputs, dim=1)

    p_normal       = avg_probs[0][0].item()
    p_lung_opacity = avg_probs[0][1].item()

    if p_lung_opacity >= p_normal:
        label = "Mờ Phổi"
        conf  = p_lung_opacity
    else:
        label = "Bình Thường"
        conf  = p_normal

    return {
        "prediction_service": label,
        "confidence_service": round(conf, 4),
        "probabilities": {
            "Bình Thường":       round(p_normal, 4),
            "Mờ Phổi": round(p_lung_opacity, 4),
        },
    }

# =========================================================
# QUICK TEST
# =========================================================
if __name__ == "__main__":
    result = predict_image("test.jpg", use_tta=True)
    print(result)