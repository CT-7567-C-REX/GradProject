import { formDataToJson, sendToEndpoint, handleFilePreview } from './utils.js';
import { setupCanvas } from './canvas.js';

// File Preview
handleFilePreview("input[type=file]", "#file-preview"); // file preview

// Canvas Setup
const canvasElement = document.getElementById('canvas');
let canvas;
if (canvasElement) {
    canvas = setupCanvas('canvas');
}

// Upload Form for Prediction
function initializeUploadForm() {
    const uploadForm = document.getElementById('uploadForm');
    if (!uploadForm) return; // Exit if the form doesn't exist

    const statusElement = document.getElementById('status');

    uploadForm.addEventListener('submit', async function (event) {
        event.preventDefault(); // Stop form from refreshing the page

        const fileInput = document.getElementById('img'); // Form image data

        statusElement.textContent = 'Uploading...'; // Set the status

        if (fileInput.files.length === 0) {
            statusElement.textContent = 'Please select an image.';
            return;
        }

        try {
            const payload = await formDataToJson(fileInput); // Convert form data to JSON
            const data = await sendToEndpoint('/prediction', payload); // Send data to server
            const predictedImageBase64 = data.image; // Server results

            statusElement.textContent = 'Upload successful!';

            // Set the predicted image as the background of the canvas
            fabric.Image.fromURL(
                'data:image/png;base64,' + predictedImageBase64,
                function (img) {
                    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
                        scaleX: canvas.width / img.width,
                        scaleY: canvas.height / img.height,
                    });
                }
            );
        } catch (error) {
            statusElement.textContent = 'Error: ' + error.message;
        }
    });
}

// Initialize Forms
initializeUploadForm();
