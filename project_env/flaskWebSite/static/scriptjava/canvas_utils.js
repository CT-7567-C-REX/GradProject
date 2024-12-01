

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
  
  