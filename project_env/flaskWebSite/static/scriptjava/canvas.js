import { setCanvasBackground, updateColorPickerFromObject, enablePanZoom, saveCanvas, updateObjectColor, updateCirclesForSelectedPolygon, drawGrid, createPolyControls, createObjectDefaultControls, initializeCenterCanvas } from './canvas_utils.js';
export function setupCanvas(canvasId) {
  const canvas = new fabric.Canvas(canvasId, {
    isDrawingMode: false,
  });

  setCanvasBackground(canvas, '/static/assets/KHAS.jpg');

  let polygonCount = 1;
  let startDrawingPolygon = false;
  let circleCount = 1;
  let points = [];
  let fillColor = "#000000"; // Default color
  let panZoomMode = false;
  let editing = false;

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

  updateObjectColor(canvas, drawingColorEl, fillColor);  // Call the update function

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
      fill: fillColor,
      stroke: fillColor,
      selectable: true,
      objectCaching: false,
      polygonNo: polygonCount,
    });

    canvas.add(polygon);
    canvas.getObjects('circle').forEach(circle => (circle.visible = false));

    polygonCount++;
    startDrawingPolygon = false;

    points = [];
    
    polygon.on('mousedblclick', () => {
      editing = !editing;
      if (editing) {
        polygon.cornerStyle = 'circle';
        polygon.cornerColor = 'rgba(0,0,255,0.5)';
        polygon.hasBorders = false;
        polygon.controls = createPolyControls(polygon);
      } else {
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
        selectable: true,
        name: 'draggableCircle',
        polygonNo: polygonCount,
        circleNo: circleCount,
      });

      points.push({ x: pointer.x, y: pointer.y });
      canvas.add(circle);
      circleCount++;
    }
  });

  initializeCenterCanvas(canvas, centerCanvasBtn);

  toggleDrawModeEl.onclick = function () {
    canvas.isDrawingMode = !canvas.isDrawingMode;
    toggleDrawModeEl.textContent = canvas.isDrawingMode ? 'Exit Draw Mode' : 'Enter Draw Mode';

    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.color = fillColor;  // Use selected fill color in draw mode
    } else {
      canvas.freeDrawingBrush.color = "#000000";  // Set color to black when exiting draw mode
      document.getElementById('drawing-color').value = "#000000";
    }
  };

  canvas.on('selection:created', () => updateColorPickerFromObject(canvas, drawingColorEl));
  canvas.on('selection:updated', () => updateColorPickerFromObject(canvas, drawingColorEl));

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

  enablePanZoom(canvas, togglePanZoomEl, zoomInEl, zoomOutEl, panZoomMode, toggleDrawModeEl);

  return canvas;
}
