import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Define the deepfake detection model
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device), strict=False)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define prediction function
def detect_deepfake(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)
    return "Fake" if prediction.item() == 1 else "Real"

# Gradio interface
iface = gr.Interface(fn=detect_deepfake, inputs=gr.Video(), outputs="text")
iface.launch()
