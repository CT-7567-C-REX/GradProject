import { formDataToJson, sendToEndpoint, handleFilePreview } from './utils.js';
import { setupCanvas } from './canvas.js';
import { setCanvasBackground, clearCanvas } from "./canvas_utils.js";


handleFilePreview("input[type=file]", "#file-preview"); // file preview

// Canvas Setup
let predimage;
let { canvas, rectangleTool } = setupCanvas("canvas");

// Function to log rectangle data
function logRectangleData() {
  console.log(rectangleTool.drawnRectangles);
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
            predimage = 'data:image/png;base64,' + predictedImageBase64,
            statusElement.textContent = 'Upload successful!';

            // Set the predicted image as the background of the canvas
            setCanvasBackground(canvas, predimage);
        } catch (error) {
            statusElement.textContent = 'Error: ' + error.message;
        }
    });
}
const clearEl = document.getElementById("clear-canvas");
clearEl.onclick = () => { clearCanvas(canvas, predimage)}; // keep the canvas while clearing the objects

initializeUploadForm();
logRectangleData(); // just for log
