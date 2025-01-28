import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import glob

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

def read_file(folder,filename):
    # Read the file with three columns: strings, float, and float
    data = np.genfromtxt(os.path.join(folder,filename), dtype=[('col1', 'U20'), ('col2', 'f8'), ('col3', 'f8')],
                         delimiter=None, encoding='utf-8', comments='#')

    # Accessing data
    star_name = data['col1']  # First column (strings)
    one_sigma_c_factor = data['col2']  # Second column (numerical)
    three_sigma_c_factor = data['col3']  # Third column (numerical)


    return star_name,one_sigma_c_factor,three_sigma_c_factor

def plot_images_grid3(image_dir, num_columns=3, max_sources=None,ini_num=None, output_file=None):
    """
    Creates a single plot with a grid of Nx3 images, ensuring consistent size and specific order.
    Displays blank plots for missing images and adds subfolder names rotated on the left of the grid.

    Args:
        image_dir (str): Path to the main directory containing subfolders of images.
        num_columns (int): Number of columns for the grid (default is 3).
        max_sources (int): Maximum number of sources (subfolders) to process. If None, process all sources.
        ini_num (int): indicate the initial one If None, process all sources.
        output_file (str): Path to save the output plot. If None, the plot will be displayed.
    """
    # Get the list of subfolders
    subfolders = sorted([sf for sf in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, sf))])

    # Limit the number of subfolders if max_sources is specified
    if max_sources is not None:
        subfolders = subfolders[ini_num:max_sources+ini_num]
        print(subfolders)

    # Initialize data for the grid
    ordered_images = []
    folder_labels = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(image_dir, subfolder)
        # Get .png files from the subfolder
        image_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.png')])

        # Order images: Shortest name -> Contains "HCO+" -> Contains "C18O"
        short_name = min(image_files, key=len, default=None)
        hco_file = next((f for f in image_files if "HCO+" in f), None)
        c18o_file = next((f for f in image_files if "C18O" in f), None)

        # Add ordered images for this subfolder (with placeholders for None)
        row_images = [short_name, hco_file, c18o_file]
        ordered_images.extend([os.path.join(subfolder_path, img) if img else None for img in row_images])
        folder_labels.append(subfolder)  # Store subfolder name for labeling

    # Calculate the number of rows
    num_rows = len(folder_labels)
    total_cells = num_rows * num_columns

    # Create the figure and axes
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 3.0, num_rows * 3.2), constrained_layout=False)
    axes = axes.flatten()  # Flatten for easier iteration

    # Plot the images
    for i, ax in enumerate(axes):
        if i < len(ordered_images):
            img_path = ordered_images[i]
            if img_path:  # If an image exists, display it
                img = mpimg.imread(img_path)
                ax.imshow(img)  # Ensure consistent image size
            else:  # Display a blank plot for missing images
                ax.set_facecolor("white")
        ax.axis('off')  # Turn off axes

    # Add subfolder names rotated 90 degrees on the left
    for row_idx, folder_label in enumerate(folder_labels):
        fig.text(
            x=0.1,  # X-position of the label (adjust if needed)
            y=1 - (row_idx + 1.5) / (num_rows+2),  # Dynamically align to row center
            s=folder_label,
            va='center',  # Vertically align the text to the center of the row
            ha='center',  # Horizontally center the text
            rotation=90,  # Rotate 90 degrees
            fontsize=10,
            color='black'
        )

    # Adjust the spacing between images
    plt.subplots_adjust(wspace=0.005, hspace=0.01)
    # Adjust the spacing between images
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.02, hspace=0.01)

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to {output_file}.")

    plt.show()


def plot_images_grid(image_dir, grid_shape=(5, 6), output_file=None, c_factor_sourced=False):
    """
    Plot a grid of images from a directory with tighter spacing.

    Args:
        image_dir (str): Directory containing the .png files.
        titles (list): List of titles for the images.
        grid_shape (tuple): Shape of the grid (rows, columns).
        output_file (str): If provided, saves the plot to this file.
    """

    molecule = image_dir.split('/')[-2]
    print('We are using this molecule: ',molecule)

    # List all .png files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])


    rows, cols = grid_shape
    if len(image_files) > rows * cols:
        raise ValueError("Grid size is too small for the number of images. There are " + str(len(image_files))+' images')

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(grid_shape[1]*4, grid_shape[0]*3.5))
    axes = axes.flatten()  # Flatten axes array for easy iteration


    unsorted_source_name, c_s1, c_s3 = read_file(folder=c_factor_folder, filename=c_factor_file)
    sorted_names = [[x,y] for x, y  in sorted(zip(c_s1, unsorted_source_name),reverse=True)]


    if c_factor_sourced:
        for ax, img_file in zip(axes, sorted_names):
            source_name = img_file[1]
            concentration = round(img_file[0],2)
            png_name = [name for name in image_files if source_name in name][0]

            img = mpimg.imread(os.path.join(image_dir, png_name))
            ax.imshow(img)

            if molecule == 'HCO+':
                ax.set_title(source_name +r' $C_{HCO^+}$=' +str(concentration), fontsize=10)
            else:
                ax.set_title(source_name +r' $C_{C^{18}O}$=' +str(concentration), fontsize=10)

            ax.axis('off')  # Turn off axes

    else:
        counter = 0
        for ax, img_file in zip(axes, image_files):
            png_name = image_files[counter]
            source_name = png_name.split(molecule)[0][:-1]
            img = mpimg.imread(os.path.join(image_dir, png_name))
            ax.imshow(img)

            # if molecule == 'HCO+':
            #     ax.set_title(source_name , fontsize=10)
            # else:
            #     ax.set_title(source_name, fontsize=10)

            ax.axis('off')  # Turn off axes
            counter = counter + 1

    # Turn off axes for unused grid spaces
    for ax in axes[len(image_files):]:
        ax.axis('off')

    # Adjust the spacing between images
    plt.subplots_adjust(wspace=0.005, hspace=0.01)  # Reduce horizontal and vertical spacing

    if output_file:
        plt.savefig(output_file+'_'+molecule+'_spectra.png', dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to {output_file}.")
    else:
        plt.show()



# Example usage
if __name__ == "__main__":

    c_factor_folder= 'text_files'
    c_factor_file='concentrations_C18O.txt'
    # c_factor_file='concentrations_HCO+.txt'

    # image_directory = "./Figures/Spectra/HCO+/central/"  # Replace with the directory containing your .png files
    image_directory = "./Figures/gallery_images/"  # Replace with the directory containing your .png files

    grid_shape = (6, 3)  # Grid of 5 rows and 6 columns

    # plot_images_grid(image_directory,  grid_shape, output_file="./Figures/grid_plot")
    # plot_same_source_grid(image_directory,  grid_shape, output_file="./Figures/grid_plot_per_source")
    plot_images_grid3(image_directory, num_columns=3, max_sources=5, ini_num=5, output_file='second_batch')

