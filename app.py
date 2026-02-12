from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import base64
from io import BytesIO
import os

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convolution_block1 = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.BatchNorm2d(8))
        self.pooling11 = nn.MaxPool2d(2, 2)
        self.convolution_block2 = nn.Sequential(nn.Conv2d(8, 20, 3), nn.ReLU(), nn.BatchNorm2d(20))
        self.pooling22 = nn.MaxPool2d(2, 2)
        self.convolution_block3 = nn.Sequential(nn.Conv2d(20, 10, 1), nn.ReLU(), nn.BatchNorm2d(10))
        self.pooling33 = nn.MaxPool2d(2, 2)
        self.convolution_block4 = nn.Sequential(nn.Conv2d(10, 20, 3), nn.ReLU(), nn.BatchNorm2d(20))
        self.convolution_block5 = nn.Sequential(nn.Conv2d(20, 32, 1), nn.ReLU(), nn.BatchNorm2d(32))
        self.convolution_block6 = nn.Sequential(nn.Conv2d(32, 10, 3), nn.ReLU(), nn.BatchNorm2d(10))
        self.convolution_block7 = nn.Sequential(nn.Conv2d(10, 10, 1), nn.ReLU(), nn.BatchNorm2d(10))
        self.convolution_block8 = nn.Sequential(nn.Conv2d(10, 14, 3), nn.ReLU(), nn.BatchNorm2d(14))
        self.convolution_block9 = nn.Sequential(nn.Conv2d(14, 16, 3), nn.ReLU(), nn.BatchNorm2d(16))
        self.gap = nn.AvgPool2d(4)
        self.convolution_block_out = nn.Conv2d(16, 2, 4)

    def forward(self, x):
        x = self.convolution_block1(x)
        x = self.pooling11(x)
        x = self.convolution_block2(x)
        x = self.pooling22(x)
        x = self.convolution_block3(x)
        x = self.pooling33(x)
        x = self.convolution_block4(x)
        x = self.convolution_block5(x)
        x = self.convolution_block6(x)
        x = self.convolution_block7(x)
        x = self.convolution_block8(x)
        x = self.convolution_block9(x)
        x = self.gap(x)
        x = self.convolution_block_out(x)
        x = x.view(-1, 2)
        return torch.sigmoid(x)

# Load model (relative path)
MODEL_PATH = os.path.join(os.getcwd(), "xray_model_full.pth")

try:
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    print("✓ Model loaded!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = Net()

model = model.to(device)
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'PNEUMONIA']

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    
    try:
        img = Image.open(file).convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            probabilities = probs[0].cpu().numpy()
            pred_class = torch.argmax(probs, dim=1).item()
            prediction = class_names[pred_class]

        return render_template('index.html', prediction=prediction, probabilities=probabilities, img_data=img_data)
    except Exception as e:
        return render_template('index.html', error=str(e))

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
