import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image

# Define the deepfake detection model
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")  # Use Xception if available
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)

# Load .pth file (weights only)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device), strict=False)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess image
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)

    # Assign label
    label = "Real" if prediction.item() == 0 else "Fake"
    color = (0, 255, 0) if label == "Real" else (0, 0, 255)

    # Display result
    cv2.putText(frame, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Deepfake Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


