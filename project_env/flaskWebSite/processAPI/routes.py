from flask import Blueprint, request, jsonify
import os
import random
from flaskWebSite.processAPI.utils import convert_json_to_pil, save_picture, model_loader, generate, convert_pil_to_base64, predict_single_image
from pathlib import Path
import torch


# Create Blueprint
pep = Blueprint('pep', __name__)

base_dir = Path(__file__).resolve().parents[1] 
# Route for testing blueprint
@pep.route("/testforblueprint", methods=['GET', 'POST'])
def test():
    return "Blueprint test successful!"


from flaskWebSite.frontend.tinyvgg import TinyVgg 
# get a clasificaition
@pep.route('/classificaiton', methods=['GET', 'POST'])
def classificaiton():
    data = request.get_json()
    image = convert_json_to_pil(data)
    model = TinyVgg(input_shape=3, hidden_units=10, output_shape=20)
    model_path = base_dir / "models" / "03_pytorch_computer_vision_model_2.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    output_class = predict_single_image(image, model)
    
    return jsonify({"pred": output_class})

   
from flaskWebSite.frontend.vgg19 import VGGUNET19 
# get a prediction
@pep.route('/prediction', methods=['GET', 'POST'])
def prediction():
    data = request.get_json()

    # Convert the incoming JSON image data to a PIL image
    image = convert_json_to_pil(data)

    # Load the model
    model = VGGUNET19()

    model_path = base_dir / "models" / "VGGUnet19_Segmentation_best.pth.tar"
    model = model_loader(model, model_path)

    # Generate the prediction
    output_image = generate(image, model)

    # Convert the output image to base64
    output_base64 = convert_pil_to_base64(output_image)

    # Return the base64-encoded image as JSON
    return jsonify({"image": output_base64})

