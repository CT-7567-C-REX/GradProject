import {
  setCanvasBackground,
  updateColorPickerFromObject,
  enablePanZoom,
  saveCanvas,
  updateObjectColor,
  initializeCenterCanvas,
  RectangleTool,
} from "./canvas_utils.js";

export function setupCanvas(canvasId) {
  const canvas = new fabric.Canvas(canvasId, {
    isDrawingMode: false,
  });

  setCanvasBackground(canvas, "/static/assets/logo.svg");

  let fillColor = "#000000";
  let panZoomMode = false;

  // DOM references
  const drawingColorEl = document.getElementById("drawing-color");
  const drawingLineWidthEl = document.getElementById("drawing-line-width");
  
  const toggleDrawModeEl = document.getElementById("toggle-draw-mode");
  const togglePanZoomEl = document.getElementById("toggle-pan-zoom");
  const zoomInEl = document.getElementById("zoom-in");
  const zoomOutEl = document.getElementById("zoom-out");
  const centerCanvasBtn = document.getElementById("center-canvas");
  const toggleRectangleModeEl = document.getElementById("toggle-rectangle-mode");

  // Set up brush
  canvas.freeDrawingBrush.color = drawingColorEl.value;
  canvas.freeDrawingBrush.width = parseInt(drawingLineWidthEl.value, 10) || 1;
  updateObjectColor(canvas, drawingColorEl, fillColor);

  drawingLineWidthEl.onchange = function () {
    canvas.freeDrawingBrush.width =
      parseInt(this.value, 10) || 1;
  };

  // Initialize center canvas logic
  initializeCenterCanvas(canvas, centerCanvasBtn);

  // Toggle free-draw mode
  toggleDrawModeEl.onclick = function () {
    canvas.isDrawingMode = !canvas.isDrawingMode;
    toggleDrawModeEl.textContent = canvas.isDrawingMode
      ? "Exit Draw Mode"
      : "Enter Draw Mode";

    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.color = fillColor;
    } else {
      canvas.freeDrawingBrush.color = "#000000";
      document.getElementById("drawing-color").value = "#000000";
    }
  };

  // Color picker updates
  canvas.on("selection:created", () =>
    updateColorPickerFromObject(canvas, drawingColorEl)
  );
  canvas.on("selection:updated", () =>
    updateColorPickerFromObject(canvas, drawingColorEl)
  );


  // Save canvas
  document.getElementById("save-canvas").onclick = function () {
    saveCanvas(canvas);
  };

  // Initialize the rectangle tool
  const rectangleTool = new RectangleTool(canvas);
  

  // Pass rectangleTool & toggleRectangleModeEl into enablePanZoom
  enablePanZoom(
    canvas,
    togglePanZoomEl,
    zoomInEl,
    zoomOutEl,
    panZoomMode,
    toggleDrawModeEl,
    rectangleTool,
    toggleRectangleModeEl
  );

  // Toggle rectangle mode button
  if (toggleRectangleModeEl) {
    toggleRectangleModeEl.onclick = function () {
      if (rectangleTool.isEnable()) {
        // Disable rectangle mode
        rectangleTool.disable();
        toggleRectangleModeEl.textContent = "Enter Rectangle Mode";
      } else {
        // If panZoomMode is on => turn it off so user can't pan while drawing
        if (!togglePanZoomEl.disabled) {
          // but we must also sync the button text & disable zoom
          panZoomMode = false;
          togglePanZoomEl.textContent = "Switch to Pan/Zoom Mode";
          zoomInEl.disabled = true;
          zoomOutEl.disabled = true;
        }

        // Turn off free-draw if needed
        canvas.isDrawingMode = false;
        toggleDrawModeEl.textContent = "Enter Draw Mode";

        // Enable rectangle mode
        rectangleTool.enable();
        toggleRectangleModeEl.textContent = "Exit Rectangle Mode";
      }
    };
  }
  
  return { canvas, rectangleTool };
}
