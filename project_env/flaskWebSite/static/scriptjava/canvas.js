export function setupCanvas(canvasId) {
    // Initialize canvas
    const canvas = new fabric.Canvas(canvasId, {
      isDrawingMode: false // Set drawing mode to off initially
    });
  
    // Set the background image for the canvas
    fabric.Image.fromURL('/static/assets/KHAS.jpg', function (img) {
      canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
        scaleX: canvas.width / img.width,
        scaleY: canvas.height / img.height
      });
    });
  
    // Setup for drawing a polygon
    let polygonCount = 1;
    let startDrawingPolygon = false;
    let circleCount = 1;
    let points = [];
    let fillColor = "#000000"; // Default color
  
    // Get references to the color picker and polygon buttons
    const drawingColorEl = document.getElementById('drawing-color');
    const drawingLineWidthEl = document.getElementById('drawing-line-width');
    const clearEl = document.getElementById('clear-canvas');
    const toggleDrawModeEl = document.getElementById('toggle-draw-mode');
    const togglePanZoomEl = document.getElementById('toggle-pan-zoom');
  
    const addPolygonBtn = document.getElementById('add-polygon'); // Button to start drawing polygon
    const createPolygonBtn = document.getElementById('create-polygon'); // Button to finalize polygon
  
    // Set initial drawing color and brush width
    canvas.freeDrawingBrush.color = drawingColorEl.value;
    canvas.freeDrawingBrush.width = parseInt(drawingLineWidthEl.value, 10) || 1;
  
    // Color picker update
    drawingColorEl.onchange = function () {
      canvas.freeDrawingBrush.color = this.value; // Update brush color for drawing
    };
  
    // Line width update
    drawingLineWidthEl.onchange = function () {
      canvas.freeDrawingBrush.width = parseInt(this.value, 10) || 1;
    };
  
    // Start drawing polygon
    addPolygonBtn.onclick = function () {
      startDrawingPolygon = true;
      points = []; // Clear previous points
      circleCount = 1; // Reset circle counter
    };
  
    // Finalize polygon drawing
    createPolygonBtn.onclick = function () {
      if (points.length < 3) return; // Ensure at least 3 points to form a polygon
      const polygon = new fabric.Polygon(points, {
        fill: fillColor,
        stroke: fillColor,
        selectable: true,
        objectCaching: false,
      });
      canvas.add(polygon);
      polygonCount++;
      startDrawingPolygon = false; // Stop drawing after polygon is created
    };
  
    // Handle mouse down event for polygon points
    canvas.on('mouse:down', function (option) {
      if (startDrawingPolygon) {
        var pointer = canvas.getPointer(option.e);
        var circle = new fabric.Circle({
          left: pointer.x,
          top: pointer.y,
          radius: 7,
          hasBorders: false,
          hasControls: false,
          polygonNo: polygonCount,
          name: "draggableCircle",
          circleNo: circleCount,
          fill: "rgba(0, 0, 0, 0.5)",
          hasRotatingPoint: false,
          originX: 'center',
          originY: 'center'
        });
  
        // Add the circle to the canvas
        canvas.add(circle);
  
        // Store the point for polygon creation
        points.push({
          x: circle.left,
          y: circle.top
        });
  
        circleCount++;
      }
    });
  
    // Zoom controls
    $(function () {
        $('#zoom-in').click(function () {
        // Check if Pan/Zoom mode is active
        if (panZoomMode) {
            const newZoom = canvas.getZoom() * 1.1;
            canvas.setZoom(newZoom);
        }
        });
    
        $('#zoom-out').click(function () {
        // Check if Pan/Zoom mode is active
        if (panZoomMode) {
            const minZoomLevel = 0.5;  // Minimum zoom level
            const newZoom = canvas.getZoom() / 1.1;
            canvas.setZoom(newZoom > minZoomLevel ? newZoom : minZoomLevel);
        }
        });
    });
  
  
    // Pan functionality
    let panning = false;
    let panZoomMode = false; // Track pan/zoom mode state
  
    canvas.on('mouse:down', function (e) {
      if (panZoomMode && !canvas.isDrawingMode) {  // Only enable panning when not in drawing mode
        panning = true;
      }
    });
  
    canvas.on('mouse:up', function () {
      panning = false;
    });
  
    canvas.on('mouse:move', function (e) {
      if (panning && e && e.e) {
        const delta = new fabric.Point(e.e.movementX, e.e.movementY);
        canvas.relativePan(delta);
      }
    });
  
    // Clear canvas functionality
    clearEl.onclick = function () {
      canvas.clear();
      // Optionally, reset zoom or background image here
      fabric.Image.fromURL('/static/assets/KHAS.jpg', function (img) {
        canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
          scaleX: canvas.width / img.width,
          scaleY: canvas.height / img.height
        });
      });
    };
  
    // Toggle drawing mode
    toggleDrawModeEl.onclick = function () {
      canvas.isDrawingMode = !canvas.isDrawingMode;
      toggleDrawModeEl.textContent = canvas.isDrawingMode ? 'Exit Draw Mode' : 'Enter Draw Mode';
    };
  
    // Toggle pan/zoom mode
    togglePanZoomEl.onclick = function () {
      panZoomMode = !panZoomMode;
      togglePanZoomEl.textContent = panZoomMode ? 'Exit Pan/Zoom Mode' : 'Enter Pan/Zoom Mode';
    };
  
    return canvas;  // Return the canvas object
  }
  