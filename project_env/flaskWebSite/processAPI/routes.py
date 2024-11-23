from flask import Blueprint, request, redirect, url_for, current_app, flash, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import random

# Create Blueprint
pep = Blueprint('pep', __name__)

# Define the upload folder relative to this file
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'UPLOAD_FOLDER')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for testing blueprint
@pep.route("/testforblueprint", methods=['GET', 'POST'])
def test():
    return "test"

# Route for uploading image
@pep.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Retrieve uploaded file
        uploaded_file = request.files.get('img')

        if not uploaded_file or uploaded_file.filename == '':
            flash("No file selected or invalid file.", "error")
            return redirect(request.url)

        # Secure filename and save file
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)
        flash("File uploaded successfully.", "success")
        return redirect(url_for('pep.test'))  # Adjust as needed to point to the desired route

    # For GET request
    return "Form submission was successful."

# Route to fetch a random image
@pep.route('/random-image', methods=['GET'])
def random_image():
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

# Route to serve images
@pep.route('/uploads/<filename>')
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
