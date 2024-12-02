// Convert form data to JSON with Base64 conversion for file input
export async function formDataToJson(fileInput) { // nut sure how this one works but HEY it is working :)
    return new Promise((resolve, reject) => {
        const file = fileInput.files[0];
        const reader = new FileReader();

        reader.onload = () => {
            const base64Image = reader.result.split(',')[1]; // Extract the Base64 data
            resolve({ image: base64Image });
        };

        reader.onerror = (error) => reject(error);

        reader.readAsDataURL(file); // Start reading the file
    });
}

// Send data to an endpoint
export async function sendToEndpoint(endpoint, data) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Server responded with an error.');
        }

        return await response.json(); // Assuming the server returns JSON
    } catch (error) {
        console.error('Error sending to endpoint:', error);
        throw error;
    }
}

// Function to handle file preview
export function handleFilePreview(fileInputSelector, previewElementSelector) {
    const fileInput = document.querySelector(fileInputSelector);
    const filePreview = document.querySelector(previewElementSelector);

    if (filePreview) {
        filePreview.style.display = "none";

        fileInput.addEventListener("change", function (e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function (event) {
                    filePreview.src = event.target.result;
                    filePreview.style.display = "block";
                };
                reader.readAsDataURL(file);
            }
        });
    }
}
