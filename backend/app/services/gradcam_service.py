import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pydicom
import io
import base64

# =========================================================
# GradCAM CLASS
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.features).sum(dim=1)

        cam = F.relu(cam).squeeze()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), class_idx


# =========================================================
# READ IMAGE (DICOM / JPG)
# =========================================================
def read_image(file_path):

    try:
        dcm = pydicom.dcmread(file_path)

        img = dcm.pixel_array.astype(np.float32)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = (img * 255).astype(np.uint8)

        if hasattr(dcm, "PhotometricInterpretation"):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                img = np.max(img) - img

        return Image.fromarray(img).convert("RGB")

    except:
        return Image.open(file_path).convert("RGB")


# =========================================================
# MAIN GRADCAM
# =========================================================
def run_gradcam(img_path, checkpoint_path, device):

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features

    # ==============================
    # MODEL ARCHITECTURE (GIỐNG NOTEBOOK)
    # ==============================
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),   # ← 128
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 3)      # ← 3 classes
    )

    model = model.to(device)

    # ==============================
    # LOAD WEIGHTS
    # ==============================
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # ==============================
    # IMAGE PREPROCESS
    # ==============================
    img_pil = read_image(img_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ==============================
    # GRADCAM
    # ==============================
    gradcam = GradCAM(model, model.layer4[-1])

    cam, pred_idx = gradcam.generate(input_tensor)

    # resize CAM
    cam_resized = cv2.resize(cam, (224, 224))

    # đảo cam để vùng activation cao -> đỏ
    cam_inverted = 1.0 - cam_resized

    # làm mượt heatmap
    cam_inverted = cv2.GaussianBlur(cam_inverted, (11, 11), 0)

    # convert sang 0-255
    heatmap_raw = np.uint8(255 * cam_inverted)

    # áp dụng colormap
    heatmap = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

    # ảnh gốc
    img_np = np.array(img_pil.resize((224, 224)))

    # overlay
    overlay = (heatmap * 0.4 + img_np * 0.6).astype(np.uint8)

    # ==============================
    # BASE64
    # ==============================
    buffer = io.BytesIO()

    Image.fromarray(overlay).save(buffer, format="PNG")

    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    # ==============================
    # CLASS NAMES (ĐÚNG THỨ TỰ NOTEBOOK)
    # ==============================
    class_names = ["LUNG_OPACITY", "NORMAL", "NOT_NORMAL"]

    return class_names[pred_idx], img_base64