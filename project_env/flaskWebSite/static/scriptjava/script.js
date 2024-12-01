import { formDataToJson, sendToEndpoint, handleFilePreview } from './utils.js';
import { setupCanvas } from './canvas.js';


handleFilePreview("input[type=file]", "#file-preview"); // file preview


const canvas = setupCanvas('canvas');


document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault(); // Stop form from refreshing the page

    const fileInput = document.getElementById('img'); // form image data
    const statusElement = document.getElementById('status'); // status section

    statusElement.textContent = 'Uploading...'; // set the status

    if (fileInput.files.length === 0) { // this part might be removed
        statusElement.textContent = 'Please select an image.';
        return;
    }

    try {
        // Convert form data to JSON
        const payload = await formDataToJson(fileInput);

        // send data to server
        const data = await sendToEndpoint('/prediction', payload);

        const predictedImageBase64 = data.image; // server results

        statusElement.textContent = 'Upload successful!';

        // Set the predicted image as the background of the canvas
        fabric.Image.fromURL('data:image/png;base64,' + predictedImageBase64, function (img) { // this fucking canvas stuff is annoying, I hated it
            canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), { // we need to devide this one into the functions as well but I had some problems
                scaleX: canvas.width / img.width,
                scaleY: canvas.height / img.height
            });
        });
    } catch (error) {
        statusElement.textContent = 'Error: ' + error.message; // if error hapens will apear in the status bar
    }
});
