import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load kiến trúc
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load trọng số
model.load_state_dict(torch.load("models/best_resnet_pneumonia.pth", map_location=device))
model = model.to(device)
model.eval()

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ["NORMAL", "PNEUMONIA"]

def predict_image(image_path, model=model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]
    confidence = conf.item()

    return {
        "Ket_qua": predicted_class,
        "Do_tin_cay": confidence
    }
    # return predicted_class, confidence

# test_image_path = "models/15879362.jpg"

# result  = predict_image(test_image_path, model)

# print(result)
