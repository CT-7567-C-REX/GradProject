import { formDataToJson, sendToEndpoint, handleFilePreview, createPayloadForBackEnd } from './utils.js';
import { setupCanvas } from './canvas.js';
import { setCanvasBackground, clearCanvas } from "./canvas_utils.js";


handleFilePreview("input[type=file]", "#file-preview"); // file preview

// Canvas Setup
let predimage;
let { canvas, rectangleTool } = setupCanvas("canvas");

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
            predimage = 'data:image/png;base64,' + predictedImageBase64,
            statusElement.textContent = 'Upload successful!';

            // Set the predicted image as the background of the canvas
            setCanvasBackground(canvas, predimage);
        } catch (error) {
            statusElement.textContent = 'Error: ' + error.message;
        }
    });
}

// Add event listener for the "Done" button
document.addEventListener("DOMContentLoaded", () => {
    const saveCanvasButton = document.getElementById("save-canvas");

    if (saveCanvasButton) { // Check if the button exists
        saveCanvasButton.onclick = async function () {

            const fileInput = document.getElementById('img'); // Form image data
            const statusElement = document.getElementById('status'); // Status feedback

            try {
                statusElement.textContent = 'Processing...';

                // Create payload for backend
                const payload = await createPayloadForBackEnd(fileInput, predimage, rectangleTool);

                // Send payload to backend endpoint
                const response = await sendToEndpoint('/rlhfprocess', payload);

                if (response.success) {
                    statusElement.textContent = 'Process completed successfully!';
                } else {
                    statusElement.textContent = 'Processing failed. Please try again.';
                }
            } catch (error) {
                if (statusElement) {
                    statusElement.textContent = 'Error: ' + error.message;
                }
            }
        };
    } else {
        console.warn('Save canvas button not found. Ensure the button is defined in your HTML.');
    }
});


const clearEl = document.getElementById("clear-canvas");
clearEl.onclick = () => { clearCanvas(canvas, predimage)}; // keep the canvas while clearing the objects

initializeUploadForm();
