from flask import Blueprint, request, jsonify
from flaskWebSite.processAPI.utils import convert_json_to_pil, model_loader, generate, convert_pil_to_base64
from flaskWebSite.processAPI.rlhfutils import PlanDataset, train_start, augment_img_bbox
from pathlib import Path
import torch
from PIL import Image
import io
import base64
from pathlib import Path


# Create Blueprint
pep = Blueprint('pep', __name__)

base_dir = Path(__file__).resolve().parents[1] 
  
from flaskWebSite.modelARCH.vgg19 import VGGUNET19
model = VGGUNET19()

model_path = base_dir / "modelsTrained" / "Daft.pth.tar"
model = model_loader(model, model_path)

# get a prediction
@pep.route('/prediction', methods=['GET', 'POST'])
def prediction():
    data = request.get_json()

    image = convert_json_to_pil(data)

    output_image = generate(image, model)

    output_base64 = convert_pil_to_base64(output_image)

    return jsonify({"image": output_base64})

@pep.route('/rlhfprocess', methods=['POST'])
def rlhf_process():

    data = request.get_json()# JSON data
    
    rectangles = data.get('rectangles', [])# Extract bbox
    extracted_data = [ {"boundingBox": rect.get("boundingBox"), "label": rect.get("label", "No label provided")} for rect in rectangles] # reformat the data

    original_image = Image.open(io.BytesIO(base64.b64decode(data.get('image')))).convert('RGB') # Extract original image
    pred_image = Image.open(io.BytesIO(base64.b64decode(data.get('predImage')))).convert('RGB') # Extract predicted image

    aug_image, aug_bboxes_data = augment_img_bbox(original_image, extracted_data)

    images_list = [original_image, aug_image]   # Both are PIL Image objects
    bboxes_list = [extracted_data, aug_bboxes_data]

    dataset = PlanDataset(images_list, transform=None)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    train_start(model, train_dataloader, bboxes_list, torch.device('cpu'))

    return jsonify({"success": True, "message": "Bounding box and label data extracted.", "data": extracted_data})
   