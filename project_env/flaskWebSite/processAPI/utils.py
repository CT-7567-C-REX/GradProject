from PIL import Image
from io import BytesIO
import base64
import secrets
from pathlib import Path
import numpy as np
from torch import nn
import torch
import torchvision.transforms as transforms

def convert_json_to_pil(json_data, image_key='image'):
    try:
        # Get the base64-encoded image string
        image_base64 = json_data.get(image_key)
        if not image_base64:
            raise ValueError(f"Key '{image_key}' not found in JSON data or value is empty.")

        # Decode the base64 string into bytes
        image_data = base64.b64decode(image_base64)

        # Convert bytes into a PIL Image object
        image = Image.open(BytesIO(image_data))

        return image
    except Exception as e:
        raise ValueError(f"Failed to convert JSON to PIL Image: {str(e)}")
    
def convert_pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can choose PNG or JPEG depending on your needs
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
    

# Save picture with pathlib
def save_picture(form_picture, wheretosave):
    """
    Save the given image to the specified directory with a random filename.
    """
    randomhex = secrets.token_hex(8)
    picture_fn = randomhex + '.png'  # Assuming the image is being saved as PNG
    save_dir = Path(wheretosave) 
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    path = save_dir / picture_fn
    
    # Save the image file
    form_picture.save(str(path))
    
    return picture_fn



def model_loader (model, model_path):

    device = torch.device('cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)

    return model

# Visualize output
def visualize_output(predictions, from_tensor=True):
    color_mapping = {
        0: (0, 0, 0),
        1: (255, 80, 80),
        2: (80, 80, 255),
        3: (255, 255, 0),
        4: (255, 255, 255),  
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
def generate(image, model):
    device = torch.device('cpu')

    model = model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)

    output_image_pil = visualize_output(output)
    
    return output_image_pil

def predict_single_image(image, model):
    device = torch.device('cpu')
    model = model.to(device)

    transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    classes = ['Not sure', 'T-Shirt', 'Shoes', 'Shorts', 'Shirt', 'Pants', 'Skirt',
           'Other', 'Top', 'Outwear', 'Dress', 'Body', 'Longsleeve', 
           'Undershirt', 'Hat', 'Polo', 'Blouse', 'Hoodie', 'Skip', 'Blazer']
    
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        pred_logits = model(image)
        pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
    
    pred_class = torch.argmax(pred_prob).item()
    return classes[pred_class]