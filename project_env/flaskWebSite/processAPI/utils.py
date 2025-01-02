from PIL import Image
from io import BytesIO
import base64
import numpy as np
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
    pil_image.save(buffered, format="PNG") 
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


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