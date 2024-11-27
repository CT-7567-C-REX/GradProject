// canvas.js

export function setupCanvas(canvasId) {
    // Initialize canvas
    const canvas = new fabric.Canvas(canvasId, {
      isDrawingMode: false // Set drawing mode to off initially
    });
  
    // Set the background image for the canvas
    fabric.Image.fromURL('/static/assets/KHAS.jpg', function (img) {
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
  
    drawingColorEl.onchange = function () {
      canvas.freeDrawingBrush.color = this.value;
    };
  
    drawingLineWidthEl.onchange = function () {
      canvas.freeDrawingBrush.width = parseInt(this.value, 10) || 1;
    };
  
    toggleDrawModeEl.onclick = function () {
      canvas.isDrawingMode = !canvas.isDrawingMode;
      toggleDrawModeEl.textContent = canvas.isDrawingMode ? 'Exit Draw Mode' : 'Enter Draw Mode';
    };
  
    clearEl.onclick = function () {
      canvas.clear();
      fabric.Image.fromURL('/static/assets/KHAS.jpg', function (img) {
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
    canvas.on('mouse:down', function (e) {
      if (!canvas.isDrawingMode) {  // Only enable panning when not in drawing mode
        panning = true;
      }
    });
  
    canvas.on('mouse:up', function () {
      panning = false;
    });
  
    canvas.on('mouse:move', function (e) {
      if (panning && e && e.e) {
        const delta = new fabric.Point(e.e.movementX, e.e.movementY);
        canvas.relativePan(delta);
      }
    });
  
    return canvas;  // Return the canvas object
  }
  