import { setCanvasBackground, updateColorPickerFromObject, enablePanZoom, saveCanvas, updateObjectColor, updateCirclesForSelectedPolygon, drawGrid, createPolyControls, createObjectDefaultControls } from './canvas_utils.js';
export function setupCanvas(canvasId) {
  // Initialize canvas
  const canvas = new fabric.Canvas(canvasId, {
    isDrawingMode: false,
  });

  var grid = 5;
  drawGrid(canvas, grid);

  setCanvasBackground(canvas, '/static/assets/KHAS.jpg');

  // Variables
  let polygonCount = 1;
  let startDrawingPolygon = false;
  let circleCount = 1;
  let points = [];
  let fillColor = "#000000"; // Default color
  let panZoomMode = false;
  let editing = false;

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

  // This branch will be related to polygon stuff.
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
    canvas.getObjects('circle').forEach(circle => (circle.visible = false));

    polygonCount++;
    startDrawingPolygon = false;

    // Reset points array to prevent duplicate polygons
    points = [];


  
    
    // Double-click to toggle edit mode for the polygon
    polygon.on('mousedblclick', () => {
      editing = !editing;
      if (editing) {
        // Enter edit mode
        polygon.cornerStyle = 'circle';
        polygon.cornerColor = 'rgba(0,0,255,0.5)';
        polygon.hasBorders = false;
        polygon.controls = createPolyControls(polygon);
      } else {
        // Exit edit mode
        polygon.cornerColor = 'blue';
        polygon.cornerStyle = 'rect';
        polygon.hasBorders = true;
        polygon.controls = createObjectDefaultControls();
      }
      polygon.setCoords();
      canvas.requestRenderAll();
    });
  };

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
        selectable: true, // Allow selection for circles
        name: 'draggableCircle', // Identify as draggable circle
        polygonNo: polygonCount, // Associate with the current polygon
        circleNo: circleCount,   // Unique identifier for the circle in this polygon
      });

      points.push({ x: pointer.x, y: pointer.y }); // Add the point to the polygon
      canvas.add(circle); // Add the circle to the canvas
      circleCount++;
    }
  });

  // canvas.on('object:moving', function (event) {
  //   const movedCircle = event.target;

  //   // Only proceed if the moved object is a draggable circle
  //   if (editing && movedCircle.name === 'draggableCircle') {
  //     const polygon = canvas.getObjects('polygon').find(p => p.polygonNo === movedCircle.polygonNo);
  //     if (polygon) {
  //       // Update the polygon points based on the moved circle
  //       const updatedPoints = polygon.points.map((point, index) => {
  //         if (index === movedCircle.circleNo - 1) {
  //           return { x: movedCircle.left, y: movedCircle.top }; // Update moved point
  //         }
  //         return point; // Keep other points unchanged
  //       });
  //       polygon.set({ points: updatedPoints });
  //       canvas.renderAll();
  //     }
  //   }
  // });

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

  document.getElementById('save-canvas').onclick = function () {
    saveCanvas(canvas);
  };

  // Enable Pan/Zoom
  enablePanZoom(canvas, togglePanZoomEl, zoomInEl, zoomOutEl, panZoomMode, toggleDrawModeEl);

  return canvas;
}
