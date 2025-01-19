import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the folder containing images
image_folder = '/Users/salmantas/Desktop/Py_Enviroments/vgg19_env/Heritage-Vision/tomiray'

# Get all image file names and sort them numerically
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))], key=lambda x: int(os.path.splitext(x)[0]))

# Number of images per row
images_per_row = 11

# Calculate the number of rows needed
num_rows = (len(image_files) + images_per_row - 1) // images_per_row

# Create the plot with reduced figure size
fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 4 * num_rows))

# Flatten axes for easy iteration (even if there's only one row)
axes = axes.ravel()

for i, image_file in enumerate(image_files):
    img_path = os.path.join(image_folder, image_file)
    img = mpimg.imread(img_path)
    axes[i].imshow(img)
    axes[i].axis('off')

# Hide unused axes if the number of images is not a multiple of images_per_row
for j in range(len(image_files), len(axes)):
    axes[j].axis('off')

# Adjust layout and reduce spacing
plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Reduce space between images
plt.tight_layout(pad=0.1)  # Reduce padding around the plot
plt.show()
