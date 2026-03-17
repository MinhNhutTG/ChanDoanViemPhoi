
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
# MODEL ARCHITECTURE — khớp với checkpoint hiện có
# Checkpoint keys: conv1/bn1/layer1-4 (backbone gốc) + fc.1/fc.2/fc.5/fc.6/fc.9
#
# fc layout được suy ra từ keys:
#   fc.0  = Dropout(0.5)
#   fc.1  = Linear(2048, 512)
#   fc.2  = BatchNorm1d(512)
#   fc.3  = ReLU
#   fc.4  = Dropout(0.4)
#   fc.5  = Linear(512, 128)
#   fc.6  = BatchNorm1d(128)
#   fc.7  = ReLU
#   fc.8  = Dropout(0.3)
#   fc.9  = Linear(128, 3)   <- 3 class goc
# =========================================================
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features  # 2048
model.fc = nn.Sequential(
    nn.Dropout(0.5),            # fc.0
    nn.Linear(num_ftrs, 512),   # fc.1
    nn.BatchNorm1d(512),        # fc.2
    nn.ReLU(),                  # fc.3
    nn.Dropout(0.4),            # fc.4
    nn.Linear(512, 128),        # fc.5
    nn.BatchNorm1d(128),        # fc.6
    nn.ReLU(),                  # fc.7
    nn.Dropout(0.3),            # fc.8
    nn.Linear(128, 3),          # fc.9 -> 3 class
)

# =========================================================
# LOAD CHECKPOINT
# =========================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/best_resnet50_rsna.pth")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

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

# TTA - 4 transform, average xac suat
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
# CLASS NAMES (3 class goc cua checkpoint)
# index: LUNG_OPACITY=0, NORMAL=1, NOT_NORMAL=2
# =========================================================
RAW_CLASSES = ["LUNG_OPACITY", "NORMAL", "NOT_NORMAL"]

# =========================================================
# Mapping ve 2 class chan doan:
#   NORMAL          -> NORMAL        (chi khi raw predict = NORMAL)
#   LUNG_OPACITY    -> LUNG_OPACITY  (gop LUNG_OPACITY + NOT_NORMAL)
#   NOT_NORMAL      -> LUNG_OPACITY
# =========================================================
def map_to_2class(raw_probs: torch.Tensor):
    """
    raw_probs: tensor shape (1, 3) - [p_LUNG_OPACITY, p_NORMAL, p_NOT_NORMAL]
    Returns: (label, confidence, prob_dict)
    """
    p_lung_opacity = raw_probs[0][0].item() + raw_probs[0][2].item()  # LUNG_OPACITY + NOT_NORMAL
    p_normal       = raw_probs[0][1].item()

    # Re-normalize
    total          = p_normal + p_lung_opacity
    p_normal       /= total
    p_lung_opacity /= total

    if p_lung_opacity >= p_normal:
        label = "LUNG_OPACITY"
        conf  = p_lung_opacity
    else:
        label = "NORMAL"
        conf  = p_normal

    return label, conf, {"NORMAL": p_normal, "LUNG_OPACITY": p_lung_opacity}


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
            avg_probs = torch.stack(all_probs).mean(dim=0)  # (1, 3)
        else:
            tensor = infer_transform(image).unsqueeze(0).to(device)
            outputs = model(tensor)
            avg_probs = torch.softmax(outputs, dim=1)       # (1, 3)

    label, conf, prob_2class = map_to_2class(avg_probs)

    return {
        "prediction_service": label,
        "confidence_service": round(conf, 4),
        "probabilities": {k: round(v, 4) for k, v in prob_2class.items()},
    }


# =========================================================
# QUICK TEST
# =========================================================
if __name__ == "__main__":
    result = predict_image("test.jpg", use_tta=True)
    print(result)