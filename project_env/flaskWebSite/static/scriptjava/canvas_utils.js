
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

export function updateCirclesForSelectedPolygon(canvas) { // show the red circles of the selected polygon only
    // Hide all circles first
    canvas.getObjects('circle').forEach(circle => (circle.visible = false));
  
    const activeObject = canvas.getActiveObject();
    if (activeObject && activeObject.type === 'polygon') {
      // Show the circles for the currently selected polygon
      const circles = canvas.getObjects('circle').filter(c => c.polygonNo === activeObject.polygonNo);
      circles.forEach(circle => (circle.visible = true));
  
      // Prevent the polygon from being moved while editing
      activeObject.selectable = false;
    }
  
    canvas.renderAll();
}
  
export function drawGrid(canvas, grid) {
  const canvasWidth = canvas.getWidth();
  const canvasHeight = canvas.getHeight();
  const cellWidth = canvasWidth / grid;
  const cellHeight = canvasHeight / grid;

  const gridMatrix = Array.from({ length: grid }, () => Array(grid).fill(0));

  // Reference to the HTML table
  const gridTable = document.getElementById('gridTable');

  // Populate the table dynamically
  gridTable.innerHTML = '';
  for (let i = 0; i < grid; i++) {
    const row = gridTable.insertRow();
    for (let j = 0; j < grid; j++) {
      const cell = row.insertCell();
      cell.textContent = gridMatrix[i][j]; // Initialize with 0
      cell.className = 'align-middle'; // Center-align Bootstrap class
      cell.style.minWidth = '50px'; // Optional: Adjust cell size
    }
  }

  for (let i = 0; i < grid; i++) {
    for (let j = 0; j < grid; j++) {
      const rect = new fabric.Rect({
        left: i * cellWidth,
        top: j * cellHeight,
        width: cellWidth,
        height: cellHeight,
        fill: 'transparent',
        stroke: '#ccc',
        selectable: false,
        hasBorders: false,
        hasControls: false,
        data: { row: i, col: j }
      });

      rect.on('mousedown', function () {
        const row = this.data.col;
        const col = this.data.row;
        const currentScore = gridMatrix[row][col];

        const newScore = parseInt(prompt(`Assign a score for cell (${row}, ${col}):`, currentScore), 10);

        if (!isNaN(newScore)) {
          gridMatrix[row][col] = newScore;
          console.log(`Updated cell (${row}, ${col}) with score: ${newScore}`);

          // Update the corresponding table cell
          gridTable.rows[row].cells[col].textContent = newScore;
        }
        // Explicitly reset canvas state
        canvas.selection = false; // Disable selection temporarily
        canvas.discardActiveObject(); // Deselect any active objects
        canvas.renderAll(); // Re-render canvas to reflect changes
      });

      canvas.add(rect);
    }
  }
  
  return gridMatrix;
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
