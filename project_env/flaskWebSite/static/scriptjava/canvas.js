import {
  setCanvasBackground,
  enablePanZoom,
  initializeCenterCanvas,
  RectangleTool,
} from "./canvas_utils.js";

export function setupCanvas(canvasId) {
  const canvas = new fabric.Canvas(canvasId, {
    isDrawingMode: false,
  });

  // Set a default background on load
  setCanvasBackground(canvas, "/static/assets/logo.svg");

  let fillColor = "#000000";
  let panZoomMode = false;

  // DOM references
  const togglePanZoomEl = document.getElementById("toggle-pan-zoom");
  const zoomInEl = document.getElementById("zoom-in");
  const zoomOutEl = document.getElementById("zoom-out");
  const centerCanvasBtn = document.getElementById("center-canvas");
  const toggleRectangleModeEl = document.getElementById("toggle-rectangle-mode");
  
  // Initialize center canvas logic
  initializeCenterCanvas(canvas, centerCanvasBtn);

  // Initialize the rectangle tool
  const rectangleTool = new RectangleTool(canvas);

  // Enable pan/zoom with reference to rectangleTool
  enablePanZoom(
    canvas,
    togglePanZoomEl,
    zoomInEl,
    zoomOutEl,
    panZoomMode,
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
          panZoomMode = false;
          togglePanZoomEl.textContent = "Switch to Pan/Zoom Mode";
          zoomInEl.disabled = true;
          zoomOutEl.disabled = true;
        }
        // Enable rectangle mode
        rectangleTool.enable();
        toggleRectangleModeEl.textContent = "Exit Rectangle Mode";
      }
    };
  }

  return { canvas, rectangleTool };
}
