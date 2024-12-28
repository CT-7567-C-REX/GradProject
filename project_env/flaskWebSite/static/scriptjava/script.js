import {
  formDataToJson,
  sendToEndpoint,
  handleFilePreview,
  createPayloadForBackEnd
} from "./utils.js";
import { setupCanvas } from "./canvas.js";
import { setCanvasBackground, clearCanvas } from "./canvas_utils.js";

handleFilePreview("input[type=file]", "#file-preview"); // file preview

// Canvas Setup
let predimage;
let { canvas, rectangleTool } = setupCanvas("canvas");

/**
 * Upload Form for Prediction
 */
function initializeUploadForm() {
  const uploadForm = document.getElementById("uploadForm");
  if (!uploadForm) return; // Exit if the form doesn't exist

  const statusElement = document.getElementById("status");

  uploadForm.addEventListener("submit", async function (event) {
    event.preventDefault(); // Stop form from refreshing the page

    const fileInput = document.getElementById("img"); // Form image data

    statusElement.textContent = "Uploading..."; // Set the status

    if (fileInput.files.length === 0) {
      statusElement.textContent = "Please select an image.";
      return;
    }

    try {
      const payload = await formDataToJson(fileInput); // Convert form data to JSON
      const data = await sendToEndpoint("/prediction", payload); // Send data to server
      const predictedImageBase64 = data.image; // Server results

      predimage = "data:image/png;base64," + predictedImageBase64;
      statusElement.textContent = "Upload successful!";

      // Set the predicted image as the background of the canvas
      setCanvasBackground(canvas, predimage);
    } catch (error) {
      statusElement.textContent = "Error: " + error.message;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  // "Done" button => submit rectangles, etc.
  const saveCanvasButton = document.getElementById("save-canvas");
  const statusElement = document.getElementById("status");

  if (saveCanvasButton) {
    saveCanvasButton.onclick = async function () {
      const fileInput = document.getElementById("img"); // Form image data

      try {
        statusElement.textContent = "Processing...";

        // Create payload for backend
        const payload = await createPayloadForBackEnd(
          fileInput,
          predimage,
          rectangleTool
        );

        // Send payload to backend endpoint
        const response = await sendToEndpoint("/rlhfprocess", payload);

        if (response.success) {
          statusElement.textContent = "Process completed successfully!";
        } else {
          statusElement.textContent = "Processing failed. Please try again.";
        }
      } catch (error) {
        statusElement.textContent = "Error: " + error.message;
      }
    };
  } else {
    console.warn(
      "Save canvas button not found. Ensure the button is defined in your HTML."
    );
  }

  // Clear Canvas
  const clearEl = document.getElementById("clear-canvas");
  clearEl.onclick = () => {
    clearCanvas(canvas, predimage);
    rectangleTool = {}; // clear the box datas
  };

  // Label Buttons => set label in rectangleTool
  const btnWhite = document.getElementById("btn-label-white");
  const btnBlack = document.getElementById("btn-label-black");
  const btnRed   = document.getElementById("btn-label-red");
  const btnGreen = document.getElementById("btn-label-green");
  const btnBlue  = document.getElementById("btn-label-blue");

  if (btnWhite && btnBlack && btnRed && btnGreen && btnBlue) {
    btnWhite.onclick = () => {
      rectangleTool.setLabel("(255,255,255)"); // background color
    };
    btnBlack.onclick = () => {
      rectangleTool.setLabel("(0,0,0)"); // walls
    };
    btnRed.onclick = () => {
      rectangleTool.setLabel("(255,80,80)"); // ıwan
    };
    btnGreen.onclick = () => {
      rectangleTool.setLabel("(255,255,0)"); // stairs
    };
    btnBlue.onclick = () => {
      rectangleTool.setLabel("(80,80,255)"); // room
    };
  }
});

initializeUploadForm();
