from PIL import Image
from io import BytesIO
import base64
import numpy as np
import torch
import os
from pathlib import Path
import torchvision.transforms as transforms
from flaskWebSite.processAPI.utils_eval import eval_datasets, eval_fn
from torch.utils.data import DataLoader
import json 

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
        0: (0, 0, 0),       # Walls
        1: (80, 80, 255),   # Room
        2: (255, 80, 80),   # Iwan
        3: (255, 255, 0),   # Stairs
        4: (255, 255, 255), # Background  
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
def generate(image, model, device):

    model = model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)

    output_image_pil = visualize_output(output)
    
    return output_image_pil

def save_model(model, root_folder, iteration, device):

    os.makedirs(root_folder, exist_ok=True)
    model_full_path = os.path.join(root_folder, f'after{iteration}feedback.pth.tar')
    base_dir = Path(__file__).resolve().parents[0]
    metrics_file_path = base_dir / "evaluation_metrics.txt"
    if not metrics_file_path.exists():
        metrics_file_path.touch() 
    test_set_small = base_dir / "test"
    test_set_all = base_dir / "testV2"
    dataset = eval_datasets(directory=test_set_small)
    dataset_all = eval_datasets(directory=test_set_all)
    dataloader = DataLoader(dataset, batch_size=4)
    dataloader_all = DataLoader(dataset_all, batch_size=4)
    eval_loss, eval_miou, eval_acc = eval_fn(model, dataloader, device)
    eval_loss_all, eval_miou_all, eval_acc_all = eval_fn(model, dataloader_all, device)
    eval_loss = float(eval_loss)
    eval_miou = float(eval_miou)
    eval_acc = float(eval_acc)
    eval_loss_all = float(eval_loss_all)
    eval_miou_all = float(eval_miou_all)
    eval_acc_all = float(eval_acc_all)
    if iteration % 15 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_full_path)
    metrics = {
        "iteration": iteration,
        "eval_loss": eval_loss,
        "eval_miou": eval_miou,
        "eval_acc": eval_acc,
        "eval_loss_all": eval_loss_all,
        "eval_miou_all": eval_miou_all,
        "eval_acc_all": eval_acc_all
    }
    with open(metrics_file_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")