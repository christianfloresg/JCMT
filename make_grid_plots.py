import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import glob
from datetime import date,datetime
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps

today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)

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

def pad_to_square(img, fill_color='white'):
    """Pads an image to make it square without distortion."""
    max_dim = max(img.size)
    padding = [(max_dim - s) // 2 for s in img.size]
    extra = [(max_dim - s) % 2 for s in img.size]
    return ImageOps.expand(img, (padding[0], padding[1], padding[0] + extra[0], padding[1] + extra[1]), fill=fill_color)

def plot_images_grid3(image_dir, num_columns=3, max_sources=None, ini_num=0, output_file=None):
    subfolders = sorted([sf for sf in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, sf))])
    subfolders = subfolders[ini_num:ini_num + max_sources] if max_sources else subfolders

    image_paths = []
    folder_labels = []
    for subfolder in subfolders:
        path = os.path.join(image_dir, subfolder)
        images = sorted([f for f in os.listdir(path) if f.endswith('.png')])
        row = [min(images, key=len, default=None),
               next((f for f in images if "HCO+" in f), None),
               next((f for f in images if "C18O" in f), None)]
        image_paths.extend([os.path.join(path, f) if f else None for f in row])
        folder_labels.append(subfolder)

    num_rows = len(folder_labels)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2.5, num_rows * 3.2), gridspec_kw={'width_ratios': [0.8, 1, 1]})
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < len(image_paths) and image_paths[i]:
            img = Image.open(image_paths[i])
            img = pad_to_square(img).resize((900, 900))
            ax.imshow(np.asarray(img))
            ax.set_aspect('equal')

    # for idx, label in enumerate(folder_labels):
        # fig.text(0.11, 0.95 - 0.775*(idx + 1.0) / num_rows, label, va='center', ha='right', rotation=90, fontsize=10)
        # fig.text(0.2, 0.95 - 0.775*(idx + 1.0) / num_rows, label, va='center', ha='right', rotation=0, fontsize=10)

    fig.subplots_adjust(wspace=0.00, hspace=0.00)

    if output_file:
        now = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(f"{output_file}{now}.png", dpi=600, bbox_inches='tight')
        print(f"Grid plot saved to {output_file}{now}.png")

    plt.show()



def plot_images_grid(image_dir, c_factor_folder, c_factor_file, grid_shape=(5, 6), output_file=None, c_factor_sorted=False):
    """
    Plot a grid of images from a directory with tighter spacing.

    Args:
        image_dir (str): Directory containing the .png files.
        titles (list): List of titles for the images.
        grid_shape (tuple): Shape of the grid (rows, columns).
        output_file (str): If provided, saves the plot to this file.
    """


    if 'HCO+' in c_factor_file:
        molecule = 'HCO+'
    elif 'C18O' in c_factor_file:
        molecule = 'C18O'
    print('We are using this molecule: ',molecule)

    # List all .png files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    # print(image_files)

    rows, cols = grid_shape
    if len(image_files) > rows * cols:
        raise ValueError("Grid size is too small for the number of images. There are " + str(len(image_files))+' images')

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(grid_shape[1]*4, grid_shape[0]*3.5))
    axes = axes.flatten()  # Flatten axes array for easy iteration


    unsorted_source_name, c_s1, c_s3 = read_file(folder=c_factor_folder, filename=c_factor_file)
    sorted_names = [[x,y] for x, y  in sorted(zip(c_s1, unsorted_source_name),reverse=True)]

    base_names = set()
    for filename in image_files:
        if molecule == 'HCO+':
            base_name = filename.split('_HCO')[0]
            base_names.add(base_name)
        else:
            base_name = filename.split('_C18')[0]
            base_names.add(base_name)
    # Step 2: Filter sorted_names based on matching names
    filtered_sorted_names = [entry for entry in sorted_names if entry[1] in base_names]

    print(len(filtered_sorted_names))

    if c_factor_sorted:
        for ii, (ax, img_file) in enumerate(zip(axes, filtered_sorted_names)):
            source_name = img_file[1]
            print(source_name)
            # print(filtered_sorted_names)
            try:
                concentration = round(img_file[0],2)
                png_name = [name for name in image_files if source_name in name][0]
                img = mpimg.imread(os.path.join(image_dir, png_name))
                ax.imshow(img)

                if molecule == 'HCO+':
                    ax.set_title(source_name +r' $C_{HCO^+}$=' +str(concentration), fontsize=10)
                else:
                    ax.set_title(source_name +r' $C_{C^{18}O}$=' +str(concentration), fontsize=10)

                ax.axis('off')  # Turn off axes
            except:
                print("I don't have image corresponding to " +source_name+ " in folder")

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
        plt.savefig(output_file+'_'+molecule+'_'+today+'_'+hour_now+'.png', dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to {output_file}.")
        # plt.show()

    else:
        plt.show()



# Example usage
if __name__ == "__main__":

    c_factor_folder= 'text_files'
    # c_factor_file='concentrations_C18O.txt'
    c_factor_file='concentrations_HCO+.txt'

    # image_directory = "./Figures/Spectra/HCO+/central/"  # Replace with the directory containing your .png files
    image_directory = "./Figures/gallery_images/"  # Replace with the directory containing your .png files
    # image_directory = "./Figures/Moment_maps/moment-zero/C18O_offset_1-sigma/outside_center_emission/"  # Replace with the directory containing your .png files

    # grid_shape = (5,4)  # Grid of 5 rows and 6 columns

    # plot_images_grid(image_directory, c_factor_folder, c_factor_file, grid_shape,
    #                  output_file="./Figures/grid_plots/grid_plot_no_source",c_factor_sorted=True)
    # plot_same_source_grid(image_directory,  grid_shape, output_file="./Figures/grid_plot_per_source")

    # y_offsets = [[0.01, 0, 0] for _ in range(5)]
    plot_images_grid3(image_directory, num_columns=3, max_sources=6, ini_num=18, output_file='test4_')

