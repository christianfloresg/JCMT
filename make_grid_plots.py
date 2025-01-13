import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

def read_file(folder,filename):
    # Read the file with three columns: strings, float, and float
    data = np.genfromtxt(os.path.join(folder,filename), dtype=[('col1', 'U20'), ('col2', 'f8'), ('col3', 'f8')],
                         delimiter=None, encoding='utf-8', comments='#')

    # Accessing data
    star_name = data['col1']  # First column (strings)
    one_sigma_c_factor = data['col2']  # Second column (numerical)
    three_sigma_c_factor = data['col3']  # Third column (numerical)


    return star_name,one_sigma_c_factor,three_sigma_c_factor

def plot_same_source_grid(image_dir, grid_shape=(5, 6), output_file=None,order=False):
    

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
    image_directory = "./Figures/Moment_maps/moment-eight/C18O/"  # Replace with the directory containing your .png files

    grid_shape = (7, 4)  # Grid of 5 rows and 6 columns

    plot_images_grid(image_directory,  grid_shape, output_file="./Figures/grid_plot")
