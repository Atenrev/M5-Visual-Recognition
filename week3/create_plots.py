import os
import matplotlib.pyplot as plt

# Define the directories containing the images
out_of_context_dir = 'out_of_context'
mask_rcnn_dir = 'mask_rcnn'
faster_rcnn_dir = 'faster_rcnn'

# Define the output directory
output_dir = 'composition'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all the image files in the out_of_context directory
image_files = os.listdir(out_of_context_dir)

# Loop over each image file
for image_file in image_files:
    # Load the images from each directory
    out_of_context_image = plt.imread(os.path.join(out_of_context_dir, image_file))
    mask_rcnn_image = plt.imread(os.path.join(mask_rcnn_dir, image_file))
    faster_rcnn_image = plt.imread(os.path.join(faster_rcnn_dir, image_file))

    # Create the plot with the three images
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].imshow(out_of_context_image)
    ax[1].imshow(mask_rcnn_image)
    ax[2].imshow(faster_rcnn_image)

    # Add a title with the name of the image
    plt.suptitle(image_file)

    # Add a text below the images with the name of the directory
    ax[0].text(0.5, -0.1, 'original', ha='center', transform=ax[0].transAxes)
    ax[1].text(0.5, -0.1, 'mask_rcnn', ha='center', transform=ax[1].transAxes)
    ax[2].text(0.5, -0.1, 'faster_rcnn', ha='center', transform=ax[2].transAxes)

    # Hide the axes and adjust the image size
    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box')
        a.set_aspect('equal')
        a.margins(4,4)

    # Save the plot to the output directory with the same name as the input image
    output_path = os.path.join(output_dir, image_file)
    dpi = 300
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)

    # Close the figure
    plt.close(fig)