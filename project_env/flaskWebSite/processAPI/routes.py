from flask import Blueprint, request, jsonify
from flaskWebSite.processAPI.utils import convert_json_to_pil, model_loader, generate, convert_pil_to_base64, save_model
from flaskWebSite.processAPI.rlhfutils import PlanDataset, train_start, augment_img_bbox
from pathlib import Path
import torch
from PIL import Image
import io
import base64

# Create Blueprint
pep = Blueprint('pep', __name__)

base_dir = Path(__file__).resolve().parents[1]
model_path = base_dir / "modelsTrained"

from flaskWebSite.modelARCH.vgg19 import VGGUNET19
model = VGGUNET19()
model = model_loader(model, model_path / "DaftNew.pth.tar")

feedback_counter = 0
#assign device mps or cuda or cpu
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


print(f"Device: {device}")

# get a prediction
@pep.route('/prediction', methods=['GET', 'POST'])
def prediction():
    data = request.get_json()

    image = convert_json_to_pil(data)

    output_image = generate(image, model, device)

    output_base64 = convert_pil_to_base64(output_image)

    return jsonify({"image": output_base64})

@pep.route('/rlhfprocess', methods=['POST'])
def rlhf_process():
    global feedback_counter
    data = request.get_json()# JSON data
    
    rectangles = data.get('rectangles', [])# Extract bbox
    extracted_data = [ {"boundingBox": rect.get("boundingBox"), "label": rect.get("label", "No label provided")} for rect in rectangles] # reformat the data

    original_image = Image.open(io.BytesIO(base64.b64decode(data.get('image')))).convert('RGB') # Extract original image
    pred_image = Image.open(io.BytesIO(base64.b64decode(data.get('predImage')))).convert('RGB') # Extract predicted image

    aug_image, aug_bboxes_data = augment_img_bbox(original_image, extracted_data)

    images_list = [original_image, aug_image]   # Both are PIL Image objects
    bbox_list = [extracted_data, aug_bboxes_data]
    pred_list = [pred_image, pred_image.rotate(90, expand=True)]

    dataset = PlanDataset(images_list, transform=None)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    train_start(model, train_dataloader, pred_list, bbox_list, device)

    feedback_counter += 1

    # Save model every 10 requests
    if feedback_counter % 1 == 0:
        save_model(model, model_path, feedback_counter, device)


    return jsonify({"success": True, "message": "Bounding box fed to the model."})
   