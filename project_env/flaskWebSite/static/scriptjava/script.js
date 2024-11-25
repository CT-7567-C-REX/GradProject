const filePreview = document.getElementById("file-preview");

if (filePreview) {
    filePreview.style.display = "none";
    document.querySelector("input[type=file]").addEventListener("change", function (e) {
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
let canvas;

(function() {
  canvas = new fabric.Canvas('canvas', {
    isDrawingMode: false  // Set drawing mode to off initially
  });

  // Set the background image for the canvas
  fabric.Image.fromURL('/static/assets/KHAS.jpg', function(img) {
    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
      scaleX: canvas.width / img.width,
      scaleY: canvas.height / img.height
    });
  });

  const drawingColorEl = document.getElementById('drawing-color');
  const drawingLineWidthEl = document.getElementById('drawing-line-width');
  const clearEl = document.getElementById('clear-canvas');
  const toggleDrawModeEl = document.getElementById('toggle-draw-mode');
  const zoomInEl = document.getElementById('zoom-in');
  const zoomOutEl = document.getElementById('zoom-out');

  canvas.freeDrawingBrush.color = drawingColorEl.value;
  canvas.freeDrawingBrush.width = parseInt(drawingLineWidthEl.value, 10) || 1;

  drawingColorEl.onchange = function() {
    canvas.freeDrawingBrush.color = this.value;
  };

  drawingLineWidthEl.onchange = function() {
    canvas.freeDrawingBrush.width = parseInt(this.value, 10) || 1;
  };

  toggleDrawModeEl.onclick = function() {
    canvas.isDrawingMode = !canvas.isDrawingMode;
    toggleDrawModeEl.textContent = canvas.isDrawingMode ? 'Exit Draw Mode' : 'Enter Draw Mode';
  };

  clearEl.onclick = function() {
    canvas.clear();
    fabric.Image.fromURL('/static/assets/KHAS.jpg', function(img) {
      canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
        scaleX: canvas.width / img.width,
        scaleY: canvas.height / img.height
      });
    });
  };

  // Zoom controls
  $(function () {
      $('#zoom-in').click(function () {
          canvas.setZoom(canvas.getZoom() * 1.1);
      });

      $('#zoom-out').click(function () {
          const minZoomLevel = 0.5;  // Minimum zoom level
          const newZoom = canvas.getZoom() / 1.1;
          canvas.setZoom(newZoom > minZoomLevel ? newZoom : minZoomLevel);
      });
  });

  // Pan functionality
  let panning = false;
  canvas.on('mouse:down', function(e) {
      if (!canvas.isDrawingMode) {  // Only enable panning when not in drawing mode
          panning = true;
      }
  });

  canvas.on('mouse:up', function() {
      panning = false;
  });

  canvas.on('mouse:move', function(e) {
      if (panning && e && e.e) {
          const delta = new fabric.Point(e.e.movementX, e.e.movementY);
          canvas.relativePan(delta);
      }
  });
})();


 // Handle image upload and prediction request
document.getElementById('uploadForm').addEventListener('submit', async function (event) {
  event.preventDefault(); // Stop form from refreshing the page

  const fileInput = document.getElementById('img');
  const statusElement = document.getElementById('status');


  statusElement.textContent = 'Uploading...';

  if (fileInput.files.length === 0) {
      statusElement.textContent = 'Please select an image.';
      return;
  }

  const file = fileInput.files[0];
  const reader = new FileReader();

  reader.onload = async function () {
      const base64Image = reader.result.split(',')[1]; // Extract the base64 data
      const payload = { image: base64Image };

      try {
          // Send the base64 image to the /prediction endpoint for prediction
          const response = await fetch('/prediction', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify(payload)
          });

          if (response.ok) {
              const data = await response.json();
              const predictedImageBase64 = data.image; // Assuming the server returns the base64 image

              statusElement.textContent = 'Upload successful!';

              // Set the predicted image as the background of the canvas
              fabric.Image.fromURL('data:image/png;base64,' + predictedImageBase64, function (img) {

                  canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
                      scaleX: canvas.width / img.width,
                      scaleY: canvas.height / img.height
                  });
              });
          } else {
              statusElement.textContent = 'Upload failed!';
          }
      } catch (error) {
          statusElement.textContent = 'Error: ' + error.message;
      }
  };

  reader.readAsDataURL(file); // Read the file as a data URL
});
