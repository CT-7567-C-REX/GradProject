from pathlib import Path
import os
import secrets
from flaskWebSite import app
import numpy as np
from torch import nn
import torch
from PIL import Image
import torchvision.transforms as transforms

# Save picture with pathlib
def save_picture(form_picture, wheretosave):
    randomhex = secrets.token_hex(8)
    picture_fn = randomhex + '.png'  # Assuming the image is being saved as PNG
    save_dir = Path(app.root_path) / wheretosave
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    path = save_dir / picture_fn
    
    # Save the image file
    form_picture.save(str(path))
    
    return picture_fn


# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load FashionMNISTModelV2
loaded_model_2 = FashionMNISTModelV2(input_shape=3, 
                                    hidden_units=10,
                                    output_shape=20)

# Load the saved state_dict using pathlib
model_path_2 = Path("project_env/flaskWebSite/models/03_pytorch_computer_vision_model_2.pth")
loaded_model_2.load_state_dict(torch.load(str(model_path_2), weights_only=True))
loaded_model_2 = loaded_model_2.to(device)

# Transform setup
transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

classes = ['Not sure', 'T-Shirt', 'Shoes', 'Shorts', 'Shirt', 'Pants', 'Skirt',
           'Other', 'Top', 'Outwear', 'Dress', 'Body', 'Longsleeve', 
           'Undershirt', 'Hat', 'Polo', 'Blouse', 'Hoodie', 'Skip', 'Blazer']

# Prediction function
def predict_single_image(image_path, model=loaded_model_2, device=device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        pred_logits = model(image)
        pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
    
    pred_class = torch.argmax(pred_prob).item()
    return classes[pred_class]


# VGG19 UNet model
from flaskWebSite.vgg19 import VGGUNET19

modelHV = VGGUNET19()

# Load checkpoint using pathlib
modelHV_path = Path("/Users/salmantas/Desktop/Py_Enviroments/vgg19_env/Heritage-Vision/models/VGGUnet19_Segmentation_best.pth")
checkpoint = torch.load(str(modelHV_path), map_location=torch.device('cpu'), weights_only=True)
modelHV.load_state_dict(checkpoint)

# Visualize output
def visualize_output(predictions, from_tensor=True):
    color_mapping = {
        0: (0, 0, 0),
        1: (255, 80, 80),
        2: (80, 80, 255),
        3: (255, 255, 255),
    }
    
    if from_tensor:
        predictions = torch.round(predictions).type(torch.LongTensor)
        predictions = predictions.squeeze(0).squeeze(0).numpy()
    
    height, width = predictions.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for label, color in color_mapping.items():
        colored_mask[predictions == label] = color

    return Image.fromarray(colored_mask)


# Generate function
def generate(image_path, model=modelHV, device=device):
    image = Image.open(image_path)
    model = model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)

    output_image_pil = visualize_output(output)
    saved_image_path = save_picture(output_image_pil, 'static/outputimgs')
    
    return saved_image_path
