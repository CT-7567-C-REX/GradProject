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

// Upload Form for Classification
function initializeClassificationForm() {
    const uploadFormClassificaiton = document.getElementById('uploadFormClassificaiton');
    const predictionResult = document.getElementById('prediction-result');
    const feedbackSection = document.getElementById('feedback-section');

    if (!uploadFormClassificaiton) return; // Exit if the form doesn't exist

    uploadFormClassificaiton.addEventListener('submit', async function (event) {
        event.preventDefault(); // Stop default form submission

        const fileInput = document.getElementById('img'); // Form image data

        try {
            const payload = await formDataToJson(fileInput); // Convert form data to JSON
            const data = await sendToEndpoint('/classificaiton', payload); // Send data to server

            const predictedClass = data.pred; // Server result (string)
            predictionResult.textContent = `Prediction: ${predictedClass}`;
            predictionResult.style.display = 'block';
            feedbackSection.style.display = 'block';
        } catch (error) {
            console.error('Error during submission:', error);
            predictionResult.textContent = 'An error occurred. Please try again.';
            predictionResult.style.display = 'block';
        }
    });
}

function initializeFeedbackForm() {
    const feedbackForm = document.getElementById('feedback-form');
    const feedbackMessage = document.getElementById('feedback-message');
    const classSelection = document.getElementById('class-selection');
    const feedbackOptions = document.querySelectorAll('input[name="feedback"]');

    const classes = [
        'Not sure', 'T-Shirt', 'Shoes', 'Shorts', 'Shirt', 'Pants', 'Skirt',
        'Other', 'Top', 'Outwear', 'Dress', 'Body', 'Longsleeve',
        'Undershirt', 'Hat', 'Polo', 'Blouse', 'Hoodie', 'Skip', 'Blazer'
    ];

    // Populate the class selection dropdown with options
    if (classSelection) {
        classes.forEach(cls => {
            const option = document.createElement('option');
            option.value = cls;
            option.textContent = cls;
            classSelection.appendChild(option);
        });
    }

    if (!feedbackForm) return; // Exit if the feedback form doesn't exist

    feedbackForm.addEventListener('submit', function (event) {
        event.preventDefault(); // Stop default form submission

        const feedbackValue = document.querySelector('input[name="feedback"]:checked'); // Get selected feedback

        if (!feedbackValue) return; // Exit if no feedback option is selected

        if (feedbackValue.value === 'correct') {
            feedbackMessage.textContent = 'Thank you for confirming the prediction!';
            feedbackMessage.style.color = 'green';
            feedbackMessage.style.display = 'block';
            feedbackForm.reset(); // Reset the form after submission
            
            return; // Do not send any request if the prediction is correct
        }

        // Validate that a class is selected when the prediction is incorrect
        const selectedClass = classSelection.value;
        if (!selectedClass) {
            feedbackMessage.textContent = 'Please select a class if the prediction is incorrect.';
            feedbackMessage.style.color = 'red';
            feedbackMessage.style.display = 'block';
            return; // Exit if no class is selected
        }

        // Prepare feedback data
        const feedbackData = { correctClass: selectedClass };

        // Send the feedback to the endpoint
        fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedbackData),
        })
        .then(response => {
            if (response.ok) {
                feedbackMessage.textContent = 'Thank you for your feedback!';
                feedbackMessage.style.color = 'green';
            } else {
                feedbackMessage.textContent = 'Error submitting feedback. Please try again.';
                feedbackMessage.style.color = 'red';
            }
            feedbackMessage.style.display = 'block';
        })
        .catch(error => {
            feedbackMessage.textContent = 'Network error. Please try again.';
            feedbackMessage.style.color = 'red';
            feedbackMessage.style.display = 'block';
        });

        feedbackForm.reset(); // Reset the form after submission
        classSelection.disabled = true; // Re-disable the dropdown
    });
}

// Initialize Feedback Form
initializeFeedbackForm();




// Initialize Forms
initializeUploadForm();
initializeClassificationForm();
