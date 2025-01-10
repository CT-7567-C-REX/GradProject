import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the folder containing images
image_folder = '/Users/salmantas/Desktop/newdataset/seg'

# Get all image file names
image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

# Number of images per row
images_per_row = 11

# Calculate number of rows needed
num_rows = (len(image_files) + images_per_row - 1) // images_per_row

# Create the plot
fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, 20))
axes = axes.ravel()  # Flatten the axes array for easy iteration

for i, image_file in enumerate(image_files):
    img_path = os.path.join(image_folder, image_file)
    img = mpimg.imread(img_path)
    axes[i].imshow(img)
    axes[i].axis('off')

# Hide any unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
