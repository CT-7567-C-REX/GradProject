from flask import Blueprint, request, jsonify
from flaskWebSite.processAPI.utils import (
    convert_json_to_pil,
    model_loader,
    generate,
    convert_pil_to_base64
)
from flaskWebSite.processAPI.rlhfutils import (
    PlanDataset,  # The dataset that yields original+rotated items
    train_start
)
from pathlib import Path
import torch
from PIL import Image
import io
import base64

pep = Blueprint('pep', __name__)
base_dir = Path(__file__).resolve().parents[1]

# Import your model architecture
from flaskWebSite.modelARCH.vgg19 import VGGUNET19
model = VGGUNET19()

# Load the trained model from disk
model_path = base_dir / "modelsTrained" / "Daft.pth.tar"
model = model_loader(model, model_path)

@pep.route('/prediction', methods=['GET','POST'])
def prediction():
    """
    Endpoint: /prediction
    Receives { "image": base64Data }, 
    returns { "image": <base64 of the model's output> }
    """
    data = request.get_json()
    image = convert_json_to_pil(data)
    output_image = generate(image, model)
    output_base64 = convert_pil_to_base64(output_image)
    return jsonify({"image": output_base64})

@pep.route('/rlhfprocess', methods=['POST'])
def rlhf_process():
    """
    Endpoint: /rlhfprocess
    Receives JSON with:
      - "image": base64 data for the original image
      - "predImage": base64 data for the predicted image (not used here, but available)
      - "rectangles": array of { "boundingBox": {...}, "label": "(R,G,B)" }

    We'll:
      1) Convert the base64 to a PIL image
      2) Create our PlanDataset => yields 2 items: 
         - Index 0 => original image + bounding boxes
         - Index 1 => rotated image + bounding boxes
      3) Print bounding boxes in the console to verify
      4) (Optional) run train_start(...) on that data
    """
    try:
        data = request.get_json()
        rectangles = data.get('rectangles', [])

        # Print user-provided rectangles
        print("[ROUTE] Original rectangles from user =>", rectangles)

        # Convert the user image from base64 to PIL
        original_image_data = base64.b64decode(data.get('image', ''))
        original_image = Image.open(io.BytesIO(original_image_data)).convert('RGB')

        # Create the dataset that yields original + rotated items
        dataset = PlanDataset(image=original_image, rectangles=rectangles)

        # We'll use a DataLoader with batch_size=1
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # Example: just iterate to show the bounding boxes printed
        for i, (img_tensor, box_data) in enumerate(dataloader):
            print(f"\n[ROUTE] DataLoader iteration {i}")
            print("Image shape =>", img_tensor.shape)
            # 'box_data' is the bounding boxes for either original or rotated

        # If you want to run a training pass with your model, uncomment below:
        """
        device = torch.device('cpu')
        train_start(model, dataloader, rectangles, device)
        """

        return jsonify({
            "success": True,
            "message": "Printed bounding boxes for original + 90deg image in console."
        })
    except Exception as e:
        # Return the error message so the front end sees "Processing failed"
        return jsonify({"success": False, "message": str(e)})
