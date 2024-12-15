import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def plot_images_grid(image_dir, grid_shape=(5, 6), output_file=None):
    """
    Plot a grid of images from a directory with tighter spacing.

    Args:
        image_dir (str): Directory containing the .png files.
        titles (list): List of titles for the images.
        grid_shape (tuple): Shape of the grid (rows, columns).
        output_file (str): If provided, saves the plot to this file.
    """

    molecule = image_dir.split('/')[-2]

    print('We are using this molecule ',molecule)

    # List all .png files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    try:
        names = [xx.strip("_"+molecule+"_new.png") for xx in image_files]
    except:
        names = [xx.strip("_"+molecule+"_new.png") for xx in image_files]


    rows, cols = grid_shape
    if len(image_files) > rows * cols:
        raise ValueError("Grid size is too small for the number of images.")

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(grid_shape[1]*3, grid_shape[0]*3))
    axes = axes.flatten()  # Flatten axes array for easy iteration


    for ax, img_file, title in zip(axes, image_files, names):
        img = mpimg.imread(os.path.join(image_dir, img_file))
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')  # Turn off axes

    # Turn off axes for unused grid spaces
    for ax in axes[len(image_files):]:
        ax.axis('off')

    # Adjust the spacing between images
    plt.subplots_adjust(wspace=0.005, hspace=0.005)  # Reduce horizontal and vertical spacing

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to {output_file}.")
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    image_directory = "./Figures/Moment_maps/C18O/"  # Replace with the directory containing your .png files
    grid_shape = (5, 5)  # Grid of 5 rows and 6 columns

    plot_images_grid(image_directory,  grid_shape, output_file="grid_plot_C18O.png")
