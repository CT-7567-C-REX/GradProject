export function setupCanvas(canvasId) {
  // Initialize canvas
  const canvas = new fabric.Canvas(canvasId, {
    isDrawingMode: false,
  });

  // Set the background image for the canvas
  fabric.Image.fromURL('/static/assets/KHAS.jpg', function (img) {
    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
      scaleX: canvas.width / img.width,
      scaleY: canvas.height / img.height,
    });
  });

  // Variables
  let polygonCount = 1;
  let startDrawingPolygon = false;
  let circleCount = 1;
  let points = [];
  let fillColor = "#000000"; // Default color
  let panZoomMode = false;

  // Element references
  const drawingColorEl = document.getElementById('drawing-color');
  const drawingLineWidthEl = document.getElementById('drawing-line-width');
  const clearEl = document.getElementById('clear-canvas');
  const toggleDrawModeEl = document.getElementById('toggle-draw-mode');
  const togglePanZoomEl = document.getElementById('toggle-pan-zoom');
  const addPolygonBtn = document.getElementById('add-polygon');
  const createPolygonBtn = document.getElementById('create-polygon');

  // Initialize drawing brush
  canvas.freeDrawingBrush.color = drawingColorEl.value;
  canvas.freeDrawingBrush.width = parseInt(drawingLineWidthEl.value, 10) || 1;

  // Event Handlers
  drawingColorEl.onchange = function () {
    fillColor = this.value; // Update fillColor dynamically
    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.color = fillColor;
    } else {
      const activeObject = canvas.getActiveObject();
      if (activeObject) {
        if (activeObject.type === 'polygon' || activeObject.type === 'circle') {
          activeObject.set({ fill: fillColor });
        } else if (activeObject.type === 'path' || activeObject.type === 'line') {
          activeObject.set({ stroke: fillColor });
        }
        canvas.renderAll();
      }
    }
  };

  drawingLineWidthEl.onchange = function () {
    canvas.freeDrawingBrush.width = parseInt(this.value, 10) || 1;
  };

  addPolygonBtn.onclick = function () {
    startDrawingPolygon = true;
    points = [];
    circleCount = 1;
  };

  createPolygonBtn.onclick = function () {
    if (points.length < 3) return;
    const polygon = new fabric.Polygon(points, {
      fill: fillColor, // Use current color picker value
      stroke: fillColor, // Use current color picker value
      selectable: true,
      objectCaching: false,
    });
    canvas.add(polygon);
    polygonCount++;
    startDrawingPolygon = false;
  };

  canvas.on('mouse:down', function (option) {
    if (startDrawingPolygon) {
      const pointer = canvas.getPointer(option.e);
      const circle = new fabric.Circle({
        left: pointer.x,
        top: pointer.y,
        radius: 7,
        hasBorders: false,
        hasControls: false,
        polygonNo: polygonCount,
        name: "draggableCircle",
        circleNo: circleCount,
        fill: "rgba(0, 0, 0, 0.5)",
        originX: 'center',
        originY: 'center',
      });

      canvas.add(circle);
      points.push({ x: circle.left, y: circle.top });
      circleCount++;
    }
  });

  canvas.on('selection:created', updateColorPicker);
  canvas.on('selection:updated', updateColorPicker);
  canvas.on('selection:cleared', () => {
    if (canvas.isDrawingMode) {
      drawingColorEl.value = canvas.freeDrawingBrush.color;
    }
  });

  function updateColorPicker() {
    const activeObject = canvas.getActiveObject();
    if (activeObject) {
      const currentColor = activeObject.fill || activeObject.stroke || '#000000';
      drawingColorEl.value = currentColor;
    }
  }

  toggleDrawModeEl.onclick = function () {
    canvas.isDrawingMode = !canvas.isDrawingMode;
    toggleDrawModeEl.textContent = canvas.isDrawingMode ? 'Exit Draw Mode' : 'Enter Draw Mode';
    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.color = fillColor;
    }
  };

  // Zoom controls
  $(function () {
    $('#zoom-in').click(function () {
    // Check if Pan/Zoom mode is active
    if (panZoomMode) {
        const newZoom = canvas.getZoom() * 1.1;
        canvas.setZoom(newZoom);
    }
    });

    $('#zoom-out').click(function () {
    // Check if Pan/Zoom mode is active
    if (panZoomMode) {
        const minZoomLevel = 0.5;  // Minimum zoom level
        const newZoom = canvas.getZoom() / 1.1;
        canvas.setZoom(newZoom > minZoomLevel ? newZoom : minZoomLevel);
    }
    });
  });

  togglePanZoomEl.onclick = function () {
    panZoomMode = !panZoomMode;
    togglePanZoomEl.textContent = panZoomMode ? 'Exit Pan/Zoom Mode' : 'Enter Pan/Zoom Mode';
  };

  clearEl.onclick = function () {
    const activeObjects = canvas.getActiveObjects(); // Get all selected objects
  if (activeObjects.length > 0) {
    // Remove all selected objects
    activeObjects.forEach(obj => canvas.remove(obj));
    } else {
      // Clear the entire canvas
      canvas.clear();
      fabric.Image.fromURL('/static/assets/KHAS.jpg', function (img) {
        canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
          scaleX: canvas.width / img.width,
          scaleY: canvas.height / img.height,
        });
      });
    }
  };

  canvas.on('selection:created', function () {
    clearEl.textContent = "Delete Selection"; // Update button text on selection
  });
  
  canvas.on('selection:updated', function () {
    clearEl.textContent = "Delete Selection"; // Ensure button text remains updated
  });
  
  canvas.on('selection:cleared', function () {
    clearEl.textContent = "Clear Canvas"; // Revert button text when no selection
  });
  
  canvas.on('mouse:down', function (e) {
    if (panZoomMode && !canvas.isDrawingMode) {
      canvas.__panning = true;
    }
  });

  canvas.on('mouse:up', function () {
    canvas.__panning = false;
  });

  canvas.on('mouse:move', function (e) {
    if (canvas.__panning && e && e.e) {
      const delta = new fabric.Point(e.e.movementX, e.e.movementY);
      canvas.relativePan(delta);
    }
  });

  return canvas;
}
