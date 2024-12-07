

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


export function enablePanZoom(canvas, togglePanZoomEl, zoomInEl, zoomOutEl, panZoomMode, toggleDrawModeEl) { // allow zoom in/out and pan

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

export function saveCanvas(canvas) { // save the image
  
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
  
  
export function updateObjectColor(canvas, drawingColorEl, fillColor) { // this function change the color of the object from color picker after it's created
    drawingColorEl.onchange = function () {
      fillColor = this.value; // Update fillColor dynamically
      if (canvas.isDrawingMode) {
        canvas.freeDrawingBrush.color = fillColor;
      } else {
        const activeObject = canvas.getActiveObject();
        if (activeObject) {
          if (activeObject.type === 'polygon' || activeObject.type === 'circle') {
            activeObject.set({ fill: fillColor, stroke: fillColor }); // Update both polygon and polygon border
          } else if (activeObject.type === 'path' || activeObject.type === 'line') {
            activeObject.set({ stroke: fillColor });
          }
          canvas.renderAll();
        }
      }
    };
}
    

export function updateCirclesForSelectedPolygon(canvas) { // show the red circles of the selected polygon only
    // Hide all circles first
    canvas.getObjects('circle').forEach(circle => (circle.visible = false));
  
    const activeObject = canvas.getActiveObject();
    if (activeObject && activeObject.type === 'polygon') {
      // Show the circles for the currently selected polygon
      const circles = canvas.getObjects('circle').filter(c => c.polygonNo === activeObject.polygonNo);
      circles.forEach(circle => (circle.visible = true));
  
      // Prevent the polygon from being moved while editing
      activeObject.selectable = false;
    }
  
    canvas.renderAll();
}
  
export function drawGrid(canvas, grid) {
  const canvasWidth = canvas.getWidth();
  const canvasHeight = canvas.getHeight();
  const cellWidth = canvasWidth / grid; // Horizontal cell size
  const cellHeight = canvasHeight / grid; // Vertical cell size

  for (let i = 0; i <= grid; i++) {
    // Vertical lines
    canvas.add(new fabric.Line([i * cellWidth, 0, i * cellWidth, canvasHeight], {
      stroke: '#ccc',
      selectable: false
    }));

    // Horizontal lines
    canvas.add(new fabric.Line([0, i * cellHeight, canvasWidth, i * cellHeight], {
      stroke: '#ccc',
      selectable: false
    }));
  }
}
  