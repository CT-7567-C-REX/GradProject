from flask import Blueprint, request, jsonify
from flaskWebSite.processAPI.utils import convert_json_to_pil, save_picture, model_loader, generate, convert_pil_to_base64, predict_single_image
from pathlib import Path
import torch


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

@pep.route('/RLHFprocess', methods=['POST'])
def RLHFprocess():
    try:
        data = request.get_json()

        image = convert_json_to_pil(data.get('canvasImage'))
        
        image.save('c:\\Users\\90555\\Desktop\\klasor\\enes.jpg', 'JPEG')

        return jsonify({"status": "success", "message": "Endpoint is working"}), 200
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500