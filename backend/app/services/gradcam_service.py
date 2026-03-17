"""
gradcam_pneumonia.py
──────────────────────────────────────────────────────────────────────────────
GradCAM cho mô hình ResNet50 + GAP/GMP chẩn đoán viêm phổi (2 class).

Thay đổi so với code cũ (3-class):
  1. self.features   – tên khớp checkpoint (cũ dùng "backbone")
  2. head structure  – Linear→ReLU→BN→Dropout→ReLU→Linear  (index 0–5)
     Phân tích từ error log:
       Unexpected keys: head.2.* (BN), head.5.weight/bias (Linear cuối)
       → index 0=Linear, 1=ReLU, 2=BN, 3=Dropout, 4=ReLU, 5=Linear
  3. NUM_CLASSES: 3 → 2   |   class_names: ['normal','lung_opacity']
  4. Checkpoint key: 'model_state_dict' → 'state_dict'
  5. target_layer: model.layer4[-1] → model.features[-1]
──────────────────────────────────────────────────────────────────────────────
"""
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
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

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
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam).squeeze()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), class_idx


# =========================================================
# READ IMAGE (DICOM / JPG / PNG)
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
    except Exception:
        return Image.open(file_path).convert("RGB")


# =========================================================
# ANALYZE HEATMAP
# =========================================================
def analyze_heatmap(cam):
    h, w = cam.shape
    mask = cam > 0.6

    if np.sum(mask) == 0:
        return "no significant region", "low"

    ys, xs = np.where(mask)
    center_x = (xs.min() + xs.max()) / 2
    center_y = (ys.min() + ys.max()) / 2

    side   = "left lung"  if center_x < w / 2 else "right lung"
    zone   = "upper"      if center_y < h / 2 else "lower"
    region = f"{zone} {side}"

    mean_i = cam.mean()
    intensity = "high" if mean_i > 0.6 else ("medium" if mean_i > 0.3 else "low")

    return region, intensity


# =========================================================
# MODEL  –  khớp CHÍNH XÁC với checkpoint đã lưu
# =========================================================
class ResNet50GAPGMPHead(nn.Module):
    """
    Tên attribute và cấu trúc head suy ra từ error log:

    Unexpected keys (checkpoint có):
      head.2.weight / head.2.bias / head.2.running_mean  → index 2 = BatchNorm1d
      head.5.weight / head.5.bias                        → index 5 = Linear(→2)

    Cấu trúc đầy đủ:
      0 = Linear(4096 → 512)
      1 = ReLU                 (no param)
      2 = BatchNorm1d(512)
      3 = Dropout              (no param)
      4 = ReLU                 (no param)
      5 = Linear(512 → num_classes)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        base = models.resnet50(weights=None)

        # Tên "features" khớp với checkpoint (không phải "backbone")
        self.features = nn.Sequential(*list(base.children())[:-2])

        num_ftrs = base.fc.in_features  # 2048

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),                    # index 0  (no param)
            nn.Linear(num_ftrs * 2, 512),   # index 1  head.1.weight / head.1.bias
            nn.BatchNorm1d(512),             # index 2  head.2.weight / head.2.bias / running_*
            nn.ReLU(inplace=True),           # index 3  (no param)
            nn.Dropout(dropout),             # index 4  (no param)
            nn.Linear(512, num_classes),     # index 5  head.5.weight / head.5.bias
        )

    def forward(self, x):
        feat    = self.features(x)                    # (B, 2048, 7, 7)
        gap_out = self.gap(feat).flatten(1)            # (B, 2048)
        gmp_out = self.gmp(feat).flatten(1)            # (B, 2048)
        x = torch.cat([gap_out, gmp_out], dim=1)       # (B, 4096) – đã flat, Flatten() trong head là no-op
        return self.head(x)                            # (B, num_classes)


# =========================================================
# MAIN
# =========================================================
def run_gradcam(img_path: str, checkpoint_path: str, device: torch.device):
    """
    Returns
    -------
    pred_label : 'normal' hoặc 'lung_opacity'
    img_base64 : overlay PNG dạng base64
    """

    # 1. Model
    model = ResNet50GAPGMPHead(num_classes=2, dropout=0.5).to(device)

    # 2. Load weights – hỗ trợ cả key 'state_dict' lẫn 'model_state_dict'
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    # 3. Tiền xử lý
    img_pil = read_image(img_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 4. GradCAM – target = layer4 (phần tử cuối của self.features)
    gradcam = GradCAM(model, model.features[-1])
    cam, pred_idx = gradcam.generate(input_tensor)

    # 5. Resize
    cam_resized = cv2.resize(cam, (224, 224))

    # 6. Phân tích
    affected_region, heatmap_intensity = analyze_heatmap(cam_resized)
    print(f"Affected region  : {affected_region}")
    print(f"Heatmap intensity: {heatmap_intensity}")

    # 7. Overlay
    cam_inv     = cv2.GaussianBlur(1.0 - cam_resized, (11, 11), 0)
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_inv), cv2.COLORMAP_JET)
    img_np      = np.array(img_pil.resize((224, 224)))
    overlay     = (heatmap * 0.4 + img_np * 0.6).astype(np.uint8)

    # 8. Base64
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    # 9. Kết quả
    class_names = ["normal", "lung_opacity"]   # đúng thứ tự CLASSES trong notebook
    pred_label  = class_names[pred_idx]
    print(f"Prediction       : {pred_label}")

    return pred_label, img_base64 ,affected_region


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python gradcam_pneumonia.py <img_path> <checkpoint_path>")
        sys.exit(1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label, b64 = run_gradcam(sys.argv[1], sys.argv[2], dev)

    out = "gradcam_output.png"
    with open(out, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"\n[Result] label={label}  |  overlay → {out}")