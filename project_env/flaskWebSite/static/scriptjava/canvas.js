import { setCanvasBackground, updateColorPickerFromObject, enablePanZoom, saveCanvas, updateObjectColor, updateCirclesForSelectedPolygon } from './canvas_utils.js';

export function setupCanvas(canvasId) {
  // Initialize canvas
  const canvas = new fabric.Canvas(canvasId, {
    isDrawingMode: false,
  });

  var grid = 6;
  var canvasSize = 512; 
  var cellSize = canvasSize / grid; 

  for (var i = 0; i <= grid; i++) {
    // Vertical lines
    canvas.add(new fabric.Line([i * cellSize, 0, i * cellSize, canvasSize], {
      stroke: '#ccc',
      selectable: false
    }));
  
    // Horizontal lines
    canvas.add(new fabric.Line([0, i * cellSize, canvasSize, i * cellSize], {
      stroke: '#ccc',
      selectable: false
    }));
  }


  setCanvasBackground(canvas, '/static/assets/KHAS.jpg');

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
  const zoomInEl = document.getElementById('zoom-in');
  const zoomOutEl = document.getElementById('zoom-out');
  const addPolygonBtn = document.getElementById('add-polygon');
  const createPolygonBtn = document.getElementById('create-polygon');
  const centerCanvasBtn = document.getElementById('center-canvas');

  // Initialize drawing brush
  canvas.freeDrawingBrush.color = drawingColorEl.value;
  canvas.freeDrawingBrush.width = parseInt(drawingLineWidthEl.value, 10) || 1;

  updateObjectColor(canvas, drawingColorEl, fillColor);

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

    // Add the polygon to the canvas
    canvas.add(polygon);

    // Calculate the center of the polygon
    const centerX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const centerY = points.reduce((sum, point) => sum + point.y, 0) / points.length;

    // Move circles slightly outward from the center
    const offset = 20; // Adjust this value for how far the circles should be moved
    const circles = canvas.getObjects('circle').filter(c => c.polygonNo === polygonCount);
    circles.forEach((circle, index) => {
        const dx = circle.left - centerX;
        const dy = circle.top - centerY;
        const magnitude = Math.sqrt(dx * dx + dy * dy);
        if (magnitude > 0) {
            circle.left += (dx / magnitude) * offset;
            circle.top += (dy / magnitude) * offset;
            circle.setCoords(); // Update the circle's bounding box
        }
        circle.visible = true; // Ensure circles are visible after creation
    });

    // Update the canvas after moving circles
    canvas.renderAll();

    polygonCount++;
    startDrawingPolygon = false;

    // Reset points array to prevent duplicate polygons
    points = [];
};

  canvas.on('selection:created', () => updateCirclesForSelectedPolygon(canvas));
  canvas.on('selection:updated', () => updateCirclesForSelectedPolygon(canvas));


  canvas.on('selection:cleared', function () {
    // Hide all circles when selection is cleared
    canvas.getObjects('circle').forEach(circle => (circle.visible = false));
  
    // Allow all polygons to be selectable again
    canvas.getObjects('polygon').forEach(polygon => (polygon.selectable = true));
    canvas.renderAll();
  });

  canvas.on('object:moving', function (event) { // needs a fix here I gues.
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

  canvas.on('selection:created', () => updateColorPickerFromObject(canvas, drawingColorEl));
  canvas.on('selection:updated', () => updateColorPickerFromObject(canvas, drawingColorEl));

  toggleDrawModeEl.onclick = function () {
    canvas.isDrawingMode = !canvas.isDrawingMode;
    toggleDrawModeEl.textContent = canvas.isDrawingMode ? 'Exit Draw Mode' : 'Enter Draw Mode';
    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.color = fillColor;
    }
  };

  clearEl.onclick = function () {
    const activeObjects = canvas.getActiveObjects();
    if (activeObjects.length > 0) {
      activeObjects.forEach(obj => canvas.remove(obj));
    } else {
      canvas.clear();
      setCanvasBackground(canvas, '/static/assets/KHAS.jpg');
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

  document.getElementById('save-canvas').onclick = function () {
    saveCanvas(canvas);
  };
  
  // Enable Pan/Zoom
  enablePanZoom(canvas, togglePanZoomEl, zoomInEl, zoomOutEl, panZoomMode, toggleDrawModeEl);

  return canvas;
}
