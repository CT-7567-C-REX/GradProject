
export function initializeCenterCanvas(canvas, centerCanvasBtn) {
  // Başlangıç pozisyonu ve zoom seviyesini kaydet
  const initialZoom = canvas.getZoom(); // Mevcut zoom seviyesini al
  const initialViewportTransform = [...canvas.viewportTransform]; // Canvas'ın başlangıç transform matrisini al

  // Center Canvas işlevi
  centerCanvasBtn.onclick = function () {
    canvas.setZoom(initialZoom); // Zoom'u başlangıç seviyesine sıfırla
    canvas.viewportTransform = [...initialViewportTransform]; // Transform'u başlangıç değerine döndür
    canvas.renderAll(); // Canvas'ı yeniden çiz
  };
}


export function setCanvasBackground(canvas, imageUrl) {  // set the background image for canvas
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
    const currentColor = activeObject.fill || activeObject.stroke || '#000000';
    colorEl.value = currentColor;
  }
}

export function enablePanZoom(canvas, togglePanZoomEl, zoomInEl, zoomOutEl, panZoomMode, toggleDrawModeEl, image) {
  let minZoom = 1; // Initialize minZoom with a default fallback

  function calculateMinZoom() {
      if (image && image.width && image.height) {
          minZoom = Math.max(canvas.width / image.width, canvas.height / image.height);
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

  const maxZoom = 10.0; // Maximum zoom level

  togglePanZoomEl.onclick = function () {
      panZoomMode = !panZoomMode;
      togglePanZoomEl.textContent = panZoomMode ? 'Exit Pan/Zoom Mode' : 'Enter Pan/Zoom Mode';

      if (panZoomMode) {
          canvas.isDrawingMode = false;
          toggleDrawModeEl.textContent = 'Enter Draw Mode';
      }
  };

  canvas.on('mouse:wheel', function (opt) {
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

  canvas.on('mouse:down', function (e) {
      if (panZoomMode && !canvas.isDrawingMode) {
          canvas.__panning = true;
          canvas.__panStart = canvas.getPointer(e.e);
      }
  });

  canvas.on('mouse:move', function (e) {
      if (canvas.__panning) {
          const delta = new fabric.Point(e.e.movementX, e.e.movementY);
          canvas.relativePan(delta);
      }
  });

  canvas.on('mouse:up', function () {
      canvas.__panning = false;
  });

  canvas.on('after:render', function () {
      enforceZoomBoundaries();
  });
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
  
export function updateObjectColor(canvas, drawingColorEl, fillColor) {
  // Bu fonksiyon, renk seçici (input) değiştiğinde çağrılacak
  drawingColorEl.onchange = function () {
    fillColor = this.value;  // Yeni rengi al
    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.color = fillColor;  // Çizim fırçasının rengini güncelle
    } else {
      const activeObject = canvas.getActiveObject();
      if (activeObject) {
        if (activeObject.type === 'polygon' || activeObject.type === 'circle') {
          activeObject.set({ fill: fillColor, stroke: fillColor });  // Poligon ve dairelerin rengini güncelle
        } else if (activeObject.type === 'path' || activeObject.type === 'line') {
          activeObject.set({ stroke: fillColor });  // Çizgilerin rengini güncelle
        }
        canvas.renderAll();  // Tüm objeleri yeniden render et
      }
    }
  };
}

  // Define the function to create poly controls
export function createPolyControls(polygon, options = {}) {
  const controls = {};
  for (let i = 0; i < polygon.points.length; i++) {
    controls[`p${i}`] = new fabric.Control({
      positionHandler: function (dim, finalMatrix, fabricObject) {
        const point = fabricObject.points[i];
        return fabric.util.transformPoint(
          { x: point.x - fabricObject.pathOffset.x, y: point.y - fabricObject.pathOffset.y },
          fabricObject.calcTransformMatrix()
        );
      },
      actionHandler: function (eventData, transform, x, y) {
        const fabricObject = transform.target;
        fabricObject.points[i].x = x;
        fabricObject.points[i].y = y;
        fabricObject._calcDimensions();
        fabricObject.setCoords();
        return true;
      },
      actionName: 'modifyPolygonPoint',
    });
  }

  // Ensure the coordinates are recalculated for the control points
  polygon.setCoords();
  return controls;
}

// Define the function to create default object controls
export function createObjectDefaultControls() {
  return {
    tl: new fabric.Control({
      x: -0.5,
      y: -0.5,
      cursorStyle: 'nwse-resize',
      actionHandler: fabric.controlsUtils.scalingEqually,
    }),
    tr: new fabric.Control({
      x: 0.5,
      y: -0.5,
      cursorStyle: 'nesw-resize',
      actionHandler: fabric.controlsUtils.scalingEqually,
    }),
    bl: new fabric.Control({
      x: -0.5,
      y: 0.5,
      cursorStyle: 'nesw-resize',
      actionHandler: fabric.controlsUtils.scalingEqually,
    }),
    br: new fabric.Control({
      x: 0.5,
      y: 0.5,
      cursorStyle: 'nwse-resize',
      actionHandler: fabric.controlsUtils.scalingEqually,
    }),
    mtr: new fabric.Control({
      x: 0,
      y: -0.5,
      offsetY: -40,
      cursorStyle: 'crosshair',
      actionHandler: fabric.controlsUtils.rotationWithSnapping,
    }),
  };
}
export class RectangleTool {
  constructor(canvas) {
      this.canvas = canvas;
      this.isDrawing = false;
      this.origX = 0;
      this.origY = 0;
      
      // We'll store drawn rectangles data here
      // In a production scenario, you might store this elsewhere or pass it to backend.
      this.drawnRectangles = [];

      this.bindEvents();
  }

  bindEvents() {
      this.canvas.on('mouse:down', (o) => this.onMouseDown(o));
      this.canvas.on('mouse:move', (o) => this.onMouseMove(o));
      this.canvas.on('mouse:up', (o) => this.onMouseUp(o));
  }

  onMouseDown(o) {
      if (!this.isDrawing) return;
      const pointer = this.canvas.getPointer(o.e);
      this.origX = pointer.x;
      this.origY = pointer.y;

      const rect = new fabric.Rect({
          left: this.origX,
          top: this.origY,
          originX: 'left',
          originY: 'top',
          width: 0,
          height: 0,
          angle: 0,
          transparentCorners: false,
          hasBorders: false,
          hasControls: false,
          stroke: 'red',
          strokeWidth: 5,
          fill: 'transparent'
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
          height: Math.abs(this.origY - pointer.y)
      });

      activeObj.setCoords();
      this.canvas.renderAll();
  }

  onMouseUp(o) {
      if (!this.isDrawing) return;
      
      const activeObj = this.canvas.getActiveObject();
      if (!activeObj) return;

      // Compute the coordinates of the corners of the rectangle
      const left = activeObj.left;
      const top = activeObj.top;
      const width = activeObj.width * activeObj.scaleX;   // In Fabric.js, width/height might be scaled
      const height = activeObj.height * activeObj.scaleY; // so we multiply by scaleX/scaleY to get actual size

      const topLeft = { x: left, y: top };
      const topRight = { x: left + width, y: top };
      const bottomLeft = { x: left, y: top + height };
      const bottomRight = { x: left + width, y: top + height };

      // Prompt the user for a label or confirmation
      const userLabel = prompt(
        `Coordinates of the drawn rectangle:
        Top Left: (${topLeft.x.toFixed(2)}, ${topLeft.y.toFixed(2)})
        Top Right: (${topRight.x.toFixed(2)}, ${topRight.y.toFixed(2)})
        Bottom Left: (${bottomLeft.x.toFixed(2)}, ${bottomLeft.y.toFixed(2)})
        Bottom Right: (${bottomRight.x.toFixed(2)}, ${bottomRight.y.toFixed(2)})

Please enter a label or description for this rectangle:`
      );

      // Store the data
      const rectData = {
          label: userLabel || 'No label provided',
          coordinates: {
              topLeft,
              topRight,
              bottomLeft,
              bottomRight
          }
      };

      this.drawnRectangles.push(rectData);

      // At this point, you could send rectData to your backend via fetch() or AJAX if you wish.
      // For now, we just log it:
      console.log('Stored rectangle data:', rectData);

      // Discard the active selection so it doesn't move around with the cursor
      this.canvas.discardActiveObject();
      this.canvas.renderAll();

      // Do NOT disable the tool here. The user can continue drawing rectangles
      // until they click the "Exit Rectangle Mode" button.
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
