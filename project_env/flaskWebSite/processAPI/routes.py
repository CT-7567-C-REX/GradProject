from flask import Blueprint, request, jsonify, url_for, send_from_directory
import os
import random
from flaskWebSite.processAPI.utils import convert_json_to_pil, save_picture, model_loader, generate, convert_pil_to_base64
from pathlib import Path


# Create Blueprint
pep = Blueprint('pep', __name__)

# Define the upload folder relative to this file
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'UPLOAD_FOLDER')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for testing blueprint
@pep.route("/testforblueprint", methods=['GET', 'POST'])
def test():
    return "Blueprint test successful!"

# Route for uploading image
@pep.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Parse JSON data
        data = request.get_json()

        # Convert JSON image to PIL Image
        image = convert_json_to_pil(data)

        # Define the save directory
        save_directory = Path(__file__).parent / 'UPLOAD_FOLDER'

        # Save the image and get the filename
        filename = save_picture(image, save_directory)

        return jsonify({
            'message': 'Image uploaded and saved successfully',
            'filename': filename,
            'path': str(save_directory / filename)
        }), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

# Route to fetch a random image
@pep.route('/random-image', methods=['GET'])
def random_image():
    try:
        # Get all files in UPLOAD_FOLDER with valid extensions
        image_files = [
            f for f in os.listdir(UPLOAD_FOLDER)
            if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not image_files:
            return jsonify({"error": "No images found"}), 404

        # Select a random image
        random_filename = random.choice(image_files)

        # Generate the file URL
        file_url = url_for('pep.get_image', filename=random_filename, _external=True)
        return jsonify({"image_url": file_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to serve images
@pep.route('/uploads/<filename>')
def get_image(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
from flaskWebSite.frontend.vgg19 import VGGUNET19 
# get a prediction
@pep.route('/prediction', methods=['GET', 'POST'])
def prediction():
    data = request.get_json()

    # Convert the incoming JSON image data to a PIL image
    image = convert_json_to_pil(data)

    # Load the model
    model = VGGUNET19()
    base_dir = Path(__file__).resolve().parents[1] 
    model_path = base_dir / "models" / "VGGUnet19_Segmentation_best.pth.tar"
    model = model_loader(model, model_path)

    # Generate the prediction
    output_image = generate(image, model)

    # Convert the output image to base64
    output_base64 = convert_pil_to_base64(output_image)

    # Return the base64-encoded image as JSON
    return {"image": output_base64}

