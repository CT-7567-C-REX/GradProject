from flask import Blueprint, request, jsonify
from flaskWebSite.processAPI.utils import convert_json_to_pil, save_picture, model_loader, generate, convert_pil_to_base64, predict_single_image
from flaskWebSite.processAPI.rlhfutils import PlanDataset, train_start
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
    try:
        
        data = request.get_json()# JSON data
        
        rectangles = data.get('rectangles', [])# Extract bbox
        extracted_data = [ {"boundingBox": rect.get("boundingBox"), "label": rect.get("label", "No label provided")} for rect in rectangles] # reformat the data

        original_image = Image.open(io.BytesIO(base64.b64decode(data.get('image')))).convert('RGB') # Extract original image
        pred_image = Image.open(io.BytesIO(base64.b64decode(data.get('predImage')))).convert('RGB') # Extract predicted image

        # Initialize dataset with the in-memory image
        dataset = PlanDataset(image=original_image, transform=None)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        print(extracted_data)
        
        train_start(model, train_dataloader, rectangles, torch.device('cpu'))

        return jsonify({"success": True, "message": "Bounding box and label data extracted.", "data": extracted_data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
