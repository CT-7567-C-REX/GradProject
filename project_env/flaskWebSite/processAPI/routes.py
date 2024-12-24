from flask import Blueprint, request, jsonify
from flaskWebSite.processAPI.utils import convert_json_to_pil, save_picture, model_loader, generate, convert_pil_to_base64, predict_single_image
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
# get a prediction
@pep.route('/prediction', methods=['GET', 'POST'])
def prediction():
    data = request.get_json()

    image = convert_json_to_pil(data)

    model = VGGUNET19()

    model_path = base_dir / "modelsTrained" / "Daft.pth.tar"
    model = model_loader(model, model_path)

    output_image = generate(image, model)

    output_base64 = convert_pil_to_base64(output_image)

    return jsonify({"image": output_base64})

@pep.route('/rlhfprocess', methods=['POST'])
def rlhf_process():
    try:
        # Parse JSON data from the request
        data = request.get_json()

        # Extract fields from the JSON
        base64_original_image = data.get('image')  
        base64_pred_image = data.get('predImage')  
        rectangles = data.get('rectangles', [])

        # Convert Base64 to PIL
        if base64_original_image:
            original_image = Image.open(io.BytesIO(base64.b64decode(base64_original_image)))
            original_image.show()

        
        if base64_pred_image:
            pred_image = Image.open(io.BytesIO(base64.b64decode(base64_pred_image)))
            pred_image.show()

        print(f"Number of rectangles: {len(rectangles)}", flush=True)
        for idx, rect_data in enumerate(rectangles, start=1):
            print(f"Rectangle #{idx}:", flush=True)
            print("  Label:", rect_data.get('label'), flush=True)
            print("  Bounding Box:", rect_data.get('boundingBox'), flush=True)
            print("  Corners:", rect_data.get('corners'), flush=True)
            print("-----", flush=True)

       
        return jsonify({"success": True, "message": "Images displayed and rectangle details logged."})

    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

