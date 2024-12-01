

export function setCanvasBackground(canvas, imageUrl) {  // set the background image for canvas
    fabric.Image.fromURL(imageUrl, function (img) {
        canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
        scaleX: canvas.width / img.width,
        scaleY: canvas.height / img.height,
        });
    });
}

export function updateColorPickerFromObject(canvas, colorEl) { // if an object is selected update the color picker at the fronend
    const activeObject = canvas.getActiveObject();
    if (activeObject) {
      const currentColor = activeObject.fill || activeObject.stroke || '#000000';
      colorEl.value = currentColor;
    }
}


export function enablePanZoom(canvas, togglePanZoomEl, zoomInEl, zoomOutEl, panZoomMode, toggleDrawModeEl) {

    const minZoom = 0.5; 
    const maxZoom = 3.0; 
  
    // change the button depend on the selected mode
    togglePanZoomEl.onclick = function () {
      panZoomMode = !panZoomMode;
      togglePanZoomEl.textContent = panZoomMode ? 'Exit Pan/Zoom Mode' : 'Enter Pan/Zoom Mode';
  
      // Disable drawing mode
      if (panZoomMode) {
        canvas.isDrawingMode = false;
        toggleDrawModeEl.textContent = 'Enter Draw Mode'; // change the drawing mode btn state
      }
    };
  
    // When Mouse Down
    canvas.on('mouse:down', function (e) {
      if (panZoomMode && !canvas.isDrawingMode) {
        canvas.__panning = true;
        canvas.__panStart = canvas.getPointer(e.e);
      }
    });
  
    // When Mouse Moves
    canvas.on('mouse:move', function (e) {
      if (canvas.__panning) {
        const delta = new fabric.Point(e.e.movementX, e.e.movementY);
        canvas.relativePan(delta);
      }
    });
  
    // When Mouse Up not pressing
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

export function saveCanvas(canvas) {
  
    // Generate the image
    const dataURL = canvas.toDataURL({
      format: 'png', 
      quality: 1.0,  // Adjust this for compression
    });
  
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'canvas-image.png'; // the file name
    link.click();
}
  
  
  
  