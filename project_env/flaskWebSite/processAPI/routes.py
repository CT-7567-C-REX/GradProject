from flask import Blueprint, request, jsonify
from flaskWebSite.processAPI.utils import convert_json_to_pil, model_loader, generate,convert_pil_to_base64
from flaskWebSite.processAPI.rlhfutils import PlanDataset, train_start, albumentations_transform                                             
from pathlib import Path
import torch
from PIL import Image
import io
import base64

pep = Blueprint('pep', __name__)
base_dir = Path(__file__).resolve().parents[1] 

from flaskWebSite.modelARCH.vgg19 import VGGUNET19
model = VGGUNET19()

model_path = base_dir / "modelsTrained" / "Daft.pth.tar"
model = model_loader(model, model_path)

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
        data = request.get_json()

        rectangles = data.get('rectangles', [])  # bounding boxes
        extracted_data = [{"boundingBox": rect.get("boundingBox"), "label": rect.get("label", "No label provided")} for rect in rectangles]

        original_image = Image.open(io.BytesIO(base64.b64decode(data.get('image')))).convert('RGB')
        pred_image = Image.open(io.BytesIO(base64.b64decode(data.get('predImage')))).convert('RGB')
        print(extracted_data)

        # Build an augmented dataset
        dataset = PlanDataset(image=original_image, rectangles=extracted_data, transform=albumentations_transform)
        

        train_dataloader = torch.utils.data.DataLoader( dataset, batch_size=1, shuffle=False)

        device = torch.device('cpu')
        #train_start(model, train_dataloader, rectangles, device)

        return jsonify({"success": True, 
                        "message": "Bounding box + label data extracted (with augmentation).", 
                        "data": extracted_data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
