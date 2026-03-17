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
# MODEL ARCHITECTURE  — khớp với checkpoint
# =========================================================
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features  # 2048

model.fc = nn.Sequential(
    nn.Dropout(0.5),          # fc.0
    nn.Linear(num_ftrs, 512), # fc.1
    nn.BatchNorm1d(512),      # fc.2
    nn.ReLU(),                # fc.3
    nn.Dropout(0.4),          # fc.4
    nn.Linear(512, 128),      # fc.5  ← checkpoint shape [128, 512]
    nn.BatchNorm1d(128),      # fc.6  ← có trong checkpoint
    nn.ReLU(),                # fc.7
    nn.Dropout(0.3),          # fc.8
    nn.Linear(128, 3),        # fc.9  ← có trong checkpoint
)

# =========================================================
# LOAD CHECKPOINT
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/best_resnet50_rsna.pth")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# =========================================================
# TRANSFORM
# =========================================================
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# =========================================================
# CLASS NAMES
# =========================================================
class_names = ["LUNG_OPACITY", "NORMAL", "NOT_NORMAL"]


# =========================================================
# READ IMAGE  (DICOM + JPG/PNG)
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
def predict_image(image_file) -> dict:
    image = read_image(image_file)
    input_tensor = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "prediction_service": class_names[pred.item()],
        "confidence_service": float(conf.item()),
    }


# =========================================================
# QUICK TEST
# =========================================================
if __name__ == "__main__":
    result = predict_image("test.jpg")
    print(result)