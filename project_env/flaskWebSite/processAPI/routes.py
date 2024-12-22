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

from flaskWebSite.modelARCH.tinyvgg import TinyVgg 

@pep.route('/classificaiton', methods=['GET', 'POST'])
def classificaiton():
    data = request.get_json()
    image = convert_json_to_pil(data)
    model = TinyVgg(input_shape=3, hidden_units=10, output_shape=20)
    model_path = base_dir / "modelsTrained" / "03_pytorch_computer_vision_model_2.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    output_class = predict_single_image(image, model)
    
    return jsonify({"pred": output_class})

   
from flaskWebSite.modelARCH.vgg19 import VGGUNET19 
# get a prediction
@pep.route('/prediction', methods=['GET', 'POST'])
def prediction():
    data = request.get_json()

    image = convert_json_to_pil(data)

    model = VGGUNET19()

    model_path = base_dir / "modelsTrained" / "newModel.pth.tar"
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
        rectangle = data.get('rectangle')  # Rectangle data

        # Convert Base64 to PIL
        if base64_original_image:
            original_image = Image.open(io.BytesIO(base64.b64decode(base64_original_image)))
            original_image.show()

        
        if base64_pred_image:
            pred_image = Image.open(io.BytesIO(base64.b64decode(base64_pred_image)))
            pred_image.show()

        # Print rectangle
        if rectangle:
            print(rectangle)

       
        return jsonify({"success": True, "message": "Images displayed and rectangle details logged."})

    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500
