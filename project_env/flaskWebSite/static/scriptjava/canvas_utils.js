export function initializeCenterCanvas(canvas, centerCanvasBtn) {
  const initialZoom = canvas.getZoom();
  const initialViewportTransform = [...canvas.viewportTransform];

  centerCanvasBtn.onclick = function () {
    canvas.setZoom(initialZoom);
    canvas.viewportTransform = [...initialViewportTransform];
    canvas.renderAll();
  };
}

export function setCanvasBackground(canvas, imageUrl) {
  fabric.Image.fromURL(imageUrl, function (img) {
    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
      scaleX: canvas.width / img.width,
      scaleY: canvas.height / img.height,
    });
  });
}

export function updateColorPickerFromObject(canvas, colorEl) {
  const activeObject = canvas.getActiveObject();
  if (activeObject) {
    const currentColor = activeObject.fill || activeObject.stroke || "#000000";
    colorEl.value = currentColor;
  }
}

/**
 * Updated enablePanZoom that:
 * - Toggles panZoomMode
 * - Enables/disables zoomIn/zoomOut buttons
 * - Also disables/enables the rectangle mode button when panZoomMode is on/off
 */
export function enablePanZoom(
  canvas,
  togglePanZoomEl,
  zoomInEl,
  zoomOutEl,
  panZoomMode,
  toggleDrawModeEl,
  rectangleTool,
  toggleRectangleModeEl,
  image
) {
  let minZoom = 1;

  function calculateMinZoom() {
    if (image && image.width && image.height) {
      minZoom = Math.max(
        canvas.width / image.width,
        canvas.height / image.height
      );
      enforceZoomBoundaries();
      centerImage();
    }
  }

  function centerImage() {
    if (image && minZoom !== undefined) {
      const offsetX = (canvas.width - image.width * minZoom) / 2;
      const offsetY = (canvas.height - image.height * minZoom) / 2;
      canvas.viewportTransform = [minZoom, 0, 0, minZoom, offsetX, offsetY];
      canvas.requestRenderAll();
    }
  }

  function enforceZoomBoundaries() {
    const currentZoom = canvas.getZoom();
    if (currentZoom < minZoom) {
      canvas.setZoom(minZoom);
      centerImage();
    }
  }

  if (image) {
    image.onload = () => {
      calculateMinZoom();
    };
  } else {
    calculateMinZoom();
  }

  const maxZoom = 10.0;

  // Initial: Pan/Zoom off => disable zoom buttons
  zoomInEl.disabled = true;
  zoomOutEl.disabled = true;

  togglePanZoomEl.onclick = function () {
    // Toggle the panZoomMode boolean
    panZoomMode = !panZoomMode;

    // Update button text
    togglePanZoomEl.textContent = panZoomMode
      ? "Exit Pan/Zoom Mode"
      : "Enter Pan/Zoom Mode";

    // Enable or disable the zoom in/out buttons
    zoomInEl.disabled = !panZoomMode;
    zoomOutEl.disabled = !panZoomMode;

    // Also enable or disable the rectangle button, so user can't enter rectangle if panZoom is active
    if (toggleRectangleModeEl) {
      toggleRectangleModeEl.disabled = panZoomMode; // <--- Key line
    }

    if (panZoomMode) {
      // Turn off free-draw
      canvas.isDrawingMode = false;
      toggleDrawModeEl.textContent = "Enter Draw Mode";

      // Also disable rectangle mode if it's on
      if (rectangleTool && rectangleTool.isEnable()) {
        rectangleTool.disable();
        toggleRectangleModeEl.textContent = "Enter Rectangle Mode";
      }
    } else {
      // We just turned off pan/zoom => recheck boundaries
      enforceZoomBoundaries();
    }
  };

  canvas.on("mouse:wheel", function (opt) {
    if (!panZoomMode) return;

    const delta = opt.e.deltaY;
    let zoom = canvas.getZoom();
    zoom *= 0.999 ** delta;
    zoom = Math.min(maxZoom, Math.max(minZoom, zoom));

    const pointer = canvas.getPointer(opt.e);
    canvas.zoomToPoint(pointer, zoom);

    opt.e.preventDefault();
    opt.e.stopPropagation();

    enforceZoomBoundaries();
  });

  zoomInEl.onclick = function () {
    if (panZoomMode) {
      const center = canvas.getCenter();
      const newZoom = Math.min(canvas.getZoom() * 1.1, maxZoom);
      canvas.zoomToPoint(new fabric.Point(center.left, center.top), newZoom);
      enforceZoomBoundaries();
    }
  };

  zoomOutEl.onclick = function () {
    if (panZoomMode) {
      const center = canvas.getCenter();
      const newZoom = Math.max(canvas.getZoom() / 1.1, minZoom);
      canvas.zoomToPoint(new fabric.Point(center.left, center.top), newZoom);
      enforceZoomBoundaries();
    }
  };

  canvas.on("mouse:down", function (e) {
    if (panZoomMode && !canvas.isDrawingMode) {
      canvas.__panning = true;
      canvas.__panStart = canvas.getPointer(e.e);
    }
  });

  canvas.on("mouse:move", function (e) {
    if (canvas.__panning) {
      const delta = new fabric.Point(e.e.movementX, e.e.movementY);
      canvas.relativePan(delta);
    }
  });

  canvas.on("mouse:up", function () {
    canvas.__panning = false;
  });

  canvas.on("after:render", function () {
    enforceZoomBoundaries();
  });
}

export function saveCanvas(canvas) {
  const dataURL = canvas.toDataURL({
    format: "png",
    quality: 1.0,
  });

  const link = document.createElement("a");
  link.href = dataURL;
  link.download = "canvas-image.png";
  link.click();
}

export function updateObjectColor(canvas, drawingColorEl, fillColor) {
  drawingColorEl.onchange = function () {
    fillColor = this.value;
    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.color = fillColor;
    } else {
      const activeObject = canvas.getActiveObject();
      if (activeObject) {
        if (
          activeObject.type === "polygon" ||
          activeObject.type === "circle"
        ) {
          activeObject.set({ fill: fillColor, stroke: fillColor });
        } else if (
          activeObject.type === "path" ||
          activeObject.type === "line"
        ) {
          activeObject.set({ stroke: fillColor });
        }
        canvas.renderAll();
      }
    }
  };
}

/**
 * RectangleTool draws a red rectangle, then upon mouse up:
 * - Removes the red rect
 * - Draws two green bounding boxes (scaling normally)
 * - Logs bounding box data: topLeft, width, height, corners
 */
export class RectangleTool {
  constructor(canvas) {
    this.canvas = canvas;
    this.isDrawing = false;
    this.origX = 0;
    this.origY = 0;
    this.drawnRectangles = [];

    this.bindEvents();
  }

  bindEvents() {
    this.canvas.on("mouse:down", (o) => this.onMouseDown(o));
    this.canvas.on("mouse:move", (o) => this.onMouseMove(o));
    this.canvas.on("mouse:up", (o) => this.onMouseUp(o));
  }

  onMouseDown(o) {
    if (!this.isDrawing) return;
    const pointer = this.canvas.getPointer(o.e);
    this.origX = pointer.x;
    this.origY = pointer.y;

    const rect = new fabric.Rect({
      left: this.origX,
      top: this.origY,
      originX: "left",
      originY: "top",
      width: 0,
      height: 0,
      angle: 0,
      transparentCorners: false,
      hasBorders: false,
      hasControls: false,
      stroke: "red",
      strokeWidth: 5,
      fill: "transparent",
    });

    this.canvas.add(rect).setActiveObject(rect);
  }

  onMouseMove(o) {
    if (!this.isDrawing) return;
    const activeObj = this.canvas.getActiveObject();
    if (!activeObj) return;

    const pointer = this.canvas.getPointer(o.e);

    if (this.origX > pointer.x) {
      activeObj.set({ left: Math.abs(pointer.x) });
    }
    if (this.origY > pointer.y) {
      activeObj.set({ top: Math.abs(pointer.y) });
    }

    activeObj.set({
      width: Math.abs(this.origX - pointer.x),
      height: Math.abs(this.origY - pointer.y),
    });

    activeObj.setCoords();
    this.canvas.renderAll();
  }

  onMouseUp(o) {
    if (!this.isDrawing) return;

    const activeObj = this.canvas.getActiveObject();
    if (!activeObj) return;

    // Compute bounding box
    const left = activeObj.left;
    const top = activeObj.top;
    const width = activeObj.width * activeObj.scaleX;
    const height = activeObj.height * activeObj.scaleY;

    // Corner coordinates
    const topLeft = { x: left, y: top };
    const topRight = { x: left + width, y: top };
    const bottomLeft = { x: left, y: top + height };
    const bottomRight = { x: left + width, y: top + height };

    // Prompt user
    const userLabel = prompt(
      `Coordinates of the drawn rectangle:
       Top Left: (${topLeft.x.toFixed(2)}, ${topLeft.y.toFixed(2)})
       Width: ${width.toFixed(2)}, Height: ${height.toFixed(2)}

Please enter a label or description for this rectangle:`
    );

    // Store data
    const rectData = {
      label: userLabel || "No label provided",
      boundingBox: {
        topLeftX: left,
        topLeftY: top,
        width: width,
        height: height,
      },
      corners: {
        topLeft,
        topRight,
        bottomLeft,
        bottomRight,
      },
    };

    this.drawnRectangles.push(rectData);
    console.log("Stored rectangle data:", rectData);

    // Remove the original red rectangle
    this.canvas.remove(activeObj);

    // Add green bounding boxes that scale with the canvas
    const outerRect = new fabric.Rect({
      left: left,
      top: top,
      width: width,
      height: height,
      fill: "transparent",
      stroke: "green",
      strokeWidth: 3,
      selectable: false,
      evented: false,
    });

    const innerRect = new fabric.Rect({
      left: left + 5,
      top: top + 5,
      width: width - 10,
      height: height - 10,
      fill: "transparent",
      stroke: "green",
      strokeWidth: 3,
      selectable: false,
      evented: false,
    });

    this.canvas.add(outerRect);
    this.canvas.add(innerRect);
    this.canvas.renderAll();

    // Discard selection
    this.canvas.discardActiveObject();
    this.canvas.renderAll();
  }

  isEnable() {
    return this.isDrawing;
  }

  enable() {
    this.isDrawing = true;
  }

  disable() {
    this.isDrawing = false;
  }
}
