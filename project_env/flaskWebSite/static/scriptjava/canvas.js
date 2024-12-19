import { setCanvasBackground, updateColorPickerFromObject, enablePanZoom, saveCanvas, updateObjectColor, initializeCenterCanvas, RectangleTool } from './canvas_utils.js';

export function setupCanvas(canvasId) {
  const canvas = new fabric.Canvas(canvasId, {
    isDrawingMode: false,
  });

  setCanvasBackground(canvas, '/static/assets/KHAS.jpg');



  let fillColor = "#000000"; // Default color
  let panZoomMode = false;


  const drawingColorEl = document.getElementById('drawing-color');
  const drawingLineWidthEl = document.getElementById('drawing-line-width');
  const clearEl = document.getElementById('clear-canvas');
  const toggleDrawModeEl = document.getElementById('toggle-draw-mode');
  const togglePanZoomEl = document.getElementById('toggle-pan-zoom');
  const zoomInEl = document.getElementById('zoom-in');
  const zoomOutEl = document.getElementById('zoom-out');
  
  const centerCanvasBtn = document.getElementById('center-canvas');
  const toggleRectangleModeEl = document.getElementById('toggle-rectangle-mode'); // Reintroduced for rectangle mode

  // Initialize drawing brush
  canvas.freeDrawingBrush.color = drawingColorEl.value;
  canvas.freeDrawingBrush.width = parseInt(drawingLineWidthEl.value, 10) || 1;

  updateObjectColor(canvas, drawingColorEl, fillColor);  // Call the update function

  drawingLineWidthEl.onchange = function () {
    canvas.freeDrawingBrush.width = parseInt(this.value, 10) || 1;
  };

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

  // Initialize the rectangle tool
  const rectangleTool = new RectangleTool(canvas);

  if (toggleRectangleModeEl) {
    toggleRectangleModeEl.onclick = function() {
      if (rectangleTool.isEnable()) {
        // If rectangle mode is currently enabled, disable it
        rectangleTool.disable();
        toggleRectangleModeEl.textContent = 'Enter Rectangle Mode';
      } else {
        // Before enabling rectangle mode, ensure other modes are disabled
        canvas.isDrawingMode = false;
        panZoomMode = false;


        rectangleTool.enable();
        toggleRectangleModeEl.textContent = 'Exit Rectangle Mode';
      }
    };
  }

  return canvas;
}
