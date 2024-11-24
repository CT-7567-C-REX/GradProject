from flask import Blueprint, request, jsonify, url_for, send_from_directory
import os
from PIL import Image
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import random
import uuid

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
        image_base64 = data.get('image')

        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        # Generate a unique filename
        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

        # Save the image
        image.save(file_path)

        return jsonify({'message': 'Image uploaded successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
