import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO  # <--- THÊM DÒNG NÀY
import base64           # <--- THÊM DÒNG NÀY NẾU CHƯA CÓ


def load_model(checkpoint_path, num_classes=2, device="cpu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)   # pretrained=False khi load checkpoint
    model.fc = torch.nn.Linear(512, num_classes)          # normal / pneumonia

    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model 

def preprocess_image(img_path, device):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225]
        )
    ])

    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    return img_pil,input_tensor

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output
            output.retain_grad()

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

        cam = F.relu(cam)
        cam = cam.squeeze()

        cam -= cam.min()
        cam /= cam.max()

        return cam.detach().cpu().numpy(), class_idx, torch.softmax(output, dim=1)[0, class_idx].item()

def create_heatmap(cam, img_pil, alpha=0.4):
    cam = cv2.resize(cam, (224, 224))
    img_np = np.array(img_pil.resize((224, 224)))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = heatmap * alpha + img_np * (1 - alpha)
    overlay = overlay.astype(np.uint8)

    return heatmap, overlay, img_np

def run_gradcam(img_path,checkpoint_path,save_path,device):
    model = load_model(checkpoint_path, device=device)

    target_layer = model.layer4[-1]  # ✅ lấy từ chính model

    img_pil, input_tensor = preprocess_image(img_path, device)

    gradcam = GradCAM(model, target_layer)
    cam, pred_class, confidence = gradcam.generate(input_tensor)

    heatmap, overlay, img_np = create_heatmap(cam, img_pil)

    plt.figure(figsize=(12, 4))

    # plt.subplot(1, 3, 1)
    # plt.title("Original")
    # plt.imshow(img_np)
    # plt.axis("off")

    # plt.subplot(1, 3, 2)
    # plt.title("Grad-CAM")
    # plt.imshow(heatmap)
    # plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.savefig(save_path, bbox_inches="tight") # Lưu ảnh trực tiếp vào file
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches="tight") # Lưu ảnh vào bộ nhớ đệm
    plt.close()

    buffer.seek(0) # Đặt con trỏ về đầu bộ nhớ đệm
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8') # Mã hóa ảnh thành base64

    return pred_class, confidence, img_base64



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pred_class, confidence = run_gradcam(
#     img_path="models/test/PNEUMONIA/person115_virus_218.jpeg",
#     checkpoint_path="models/best_resnet_pneumonia.pth",
#     save_path="models/output/grad_cam.png",
#     device=device
# )


# print("Predicted class:", pred_class)
# print("Confidence:", confidence)



