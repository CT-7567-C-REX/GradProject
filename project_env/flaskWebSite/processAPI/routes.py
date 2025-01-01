from flask import Blueprint, request, jsonify
from flaskWebSite.processAPI.utils import convert_json_to_pil, model_loader, generate, convert_pil_to_base64
from flaskWebSite.processAPI.rlhfutils import PlanDataset, train_start
from pathlib import Path
import torch
from PIL import Image
import io
import base64
from pathlib import Path
import numpy as np
import albumentations as A

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

        alb_bboxes = []
        alb_labels = []
        for rect in extracted_data:
            bb = rect['boundingBox']
            label = rect['label']
            
            x = bb['topLeftX']
            y = bb['topLeftY']
            w = bb['width']
            h = bb['height']
            
            alb_bboxes.append([x, y, w, h])
            alb_labels.append(label)

        transform_90 = A.Compose(
            [
                A.Rotate(limit=[90, 90], p=1.0)
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
        )
        augmented = transform_90(
            image=np.array(original_image),
            bboxes=alb_bboxes,
            labels=alb_labels
        )
         # Convert augmented image (NumPy array) back to PIL
        aug_image = Image.fromarray(augmented['image'])
        aug_bboxes = augmented['bboxes']  # still in COCO format => [x, y, w, h]

        aug_bboxes_data = []
        for bbox, label in zip(aug_bboxes, alb_labels):
            x, y, w, h = bbox
            aug_bboxes_data.append({
                'boundingBox': {
                    'topLeftX': int(x),
                    'topLeftY': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                'label': label
            })

        images_list = [original_image, aug_image]   # Both are PIL Image objects
        bboxes_list = [extracted_data, aug_bboxes_data]

        images_list[1].show()
        print("Augmented BBoxes:", aug_bboxes_data)


        # Initialize dataset with the in-memory image
        dataset = PlanDataset(images_list, transform=None)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        print(extracted_data)
        
        train_start(model, train_dataloader, bboxes_list, torch.device('cpu'))

        return jsonify({"success": True, "message": "Bounding box and label data extracted.", "data": extracted_data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
