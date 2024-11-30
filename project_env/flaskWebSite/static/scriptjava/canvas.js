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
  const centerCanvasBtn = document.getElementById('center-canvas');

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
          activeObject.set({ fill: fillColor, stroke: fillColor }); // Update both fill and stroke
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
  
    // Create the polygon
    const polygon = new fabric.Polygon(points, {
      fill: fillColor,
      stroke: fillColor,
      selectable: true,
      objectCaching: false,
      polygonNo: polygonCount,
    });
  
    // Hide the circles after creating the polygon
    const circles = canvas.getObjects('circle').filter(c => c.polygonNo === polygonCount);
    circles.forEach(circle => circle.visible = false);
  
    canvas.add(polygon);
    polygonCount++;
    startDrawingPolygon = false;
  
    // Reset points array to prevent duplicate polygons
    points = [];
  };
  
  canvas.on('selection:created', function (e) {
    const activeObject = canvas.getActiveObject();
  
    if (activeObject && activeObject.type === 'polygon') {
      // Show the circles for the selected polygon
      const circles = canvas.getObjects('circle').filter(c => c.polygonNo === activeObject.polygonNo);
      circles.forEach(circle => circle.visible = true);
  
      // Prevent moving the polygon while editing
      activeObject.selectable = false;
    }
    canvas.renderAll();
  });
  
  canvas.on('selection:cleared', function () {
    // Hide all circles when selection is cleared
    canvas.getObjects('circle').forEach(circle => (circle.visible = false));
  
    // Allow all polygons to be selectable again
    canvas.getObjects('polygon').forEach(polygon => (polygon.selectable = true));
    canvas.renderAll();
  });
  
  // Prevent the polygon from moving when adjusting points
  canvas.on('object:moving', function (event) {
    const movedCircle = event.target;
  
    if (movedCircle.name === 'draggableCircle') {
      const polygon = canvas.getObjects('polygon').find(p => p.polygonNo === movedCircle.polygonNo);
      if (polygon) {
        // Update the points of the polygon dynamically
        const updatedPoints = polygon.points.map((point, index) => {
          return index === movedCircle.circleNo - 1
            ? { x: movedCircle.left, y: movedCircle.top }
            : point;
        });
        polygon.set({ points: updatedPoints });
        canvas.renderAll();
      }
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
      if (panZoomMode) {
        const newZoom = canvas.getZoom() * 1.1;
        canvas.setZoom(newZoom);
      }
    });

    $('#zoom-out').click(function () {
      if (panZoomMode) {
        const minZoomLevel = 0.5;
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
    const activeObjects = canvas.getActiveObjects();
    if (activeObjects.length > 0) {
      activeObjects.forEach(obj => canvas.remove(obj));
    } else {
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
    clearEl.textContent = "Delete Selection";
  });

  canvas.on('selection:updated', function () {
    clearEl.textContent = "Delete Selection";
  });

  canvas.on('selection:cleared', function () {
    clearEl.textContent = "Clear Canvas";
  });

  canvas.on('mouse:down', function (e) {

    if (panZoomMode && !canvas.isDrawingMode) {
      canvas.__panning = true;
    }
    if (startDrawingPolygon && !panZoomMode) {
      const pointer = canvas.getPointer(e.e);
      const circle = new fabric.Circle({
        left: pointer.x,
        top: pointer.y,
        radius: 5,
        fill: 'red',
        stroke: 'red',
        strokeWidth: 1,
        originX: 'center',
        originY: 'center',
        selectable: true,
        name: 'draggableCircle',
        polygonNo: polygonCount, // Associate with the current polygon
        circleNo: circleCount,   // Unique identifier for the circle in this polygon
      });

      points.push({ x: pointer.x, y: pointer.y }); // Add the point to the polygon
      canvas.add(circle); // Add the circle to the canvas
      circleCount++;
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
