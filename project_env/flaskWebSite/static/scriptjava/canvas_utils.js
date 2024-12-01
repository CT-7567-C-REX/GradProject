

export function setCanvasBackground(canvas, imageUrl) {  // set the background image for canvas
    fabric.Image.fromURL(imageUrl, function (img) {
        canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
        scaleX: canvas.width / img.width,
        scaleY: canvas.height / img.height,
        });
    });
}

export function updateColorPickerFromObject(canvas, colorEl) {
    const activeObject = canvas.getActiveObject();
    if (activeObject) {
      const currentColor = activeObject.fill || activeObject.stroke || '#000000';
      colorEl.value = currentColor;
    }
}


export function enablePanZoom(canvas, togglePanZoomEl, zoomInEl, zoomOutEl, panZoomMode) {
    const minZoom = 0.5; // Minimum zoom level
    const maxZoom = 3.0; // Maximum zoom level
  
    // Toggle Pan/Zoom mode
    togglePanZoomEl.onclick = function () {
      panZoomMode = !panZoomMode;
      togglePanZoomEl.textContent = panZoomMode ? 'Exit Pan/Zoom Mode' : 'Enter Pan/Zoom Mode';
  
      // Disable drawing mode when entering Pan/Zoom mode
      if (panZoomMode) {
        canvas.isDrawingMode = false;
      }
    };
  
    // Mouse Down: Start Panning
    canvas.on('mouse:down', function (e) {
      if (panZoomMode && !canvas.isDrawingMode) {
        canvas.__panning = true;
        canvas.__panStart = canvas.getPointer(e.e);
      }
    });
  
    // Mouse Move: Handle Panning
    canvas.on('mouse:move', function (e) {
      if (canvas.__panning) {
        const delta = new fabric.Point(e.e.movementX, e.e.movementY);
        canvas.relativePan(delta);
      }
    });
  
    // Mouse Up: End Panning
    canvas.on('mouse:up', function () {
      canvas.__panning = false;
    });
  
    // Zoom In
    zoomInEl.onclick = function () {
      if (panZoomMode) {
        const newZoom = Math.min(canvas.getZoom() * 1.1, maxZoom);
        canvas.setZoom(newZoom);
      }
    };
  
    // Zoom Out
    zoomOutEl.onclick = function () {
      if (panZoomMode) {
        const newZoom = Math.max(canvas.getZoom() / 1.1, minZoom);
        canvas.setZoom(newZoom);
      }
    };
  }
  
  
  