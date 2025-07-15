# import astropy
from astropy.visualization import *#make_lupton_rgb#, make_rgb
# from astropy import visualization.make_rgb
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from plot_generation_JCMT import find_simbad_source_in_file, get_icrs_coordinates
import os

'''
2MASS images need to be donwloaded from here. typically sub-image size is 120''
https://irsa.ipac.caltech.edu/applications/2MASS/IM/interactive.html
'''

def offset_coordinates(ax,skycoord_object):
    ra = ax.coords['ra']
    ra.set_auto_axislabel(False)
    dec = ax.coords['dec']
    # ra.set_coord_type('longitude', 180)
    overlay = ax.get_coords_overlay(skycoord_object.skyoffset_frame())
    ra.set_ticklabel_visible(False)
    dec.set_ticklabel_visible(False)
    ra.set_ticks_visible(False)
    dec.set_ticks_visible(False)

    lon = overlay['lon']
    # lon.set_coord_type('longitude', 180)
    lon.set_format_unit(u.arcsec)
    # lon.set_ticklabel(rotation=, pad=2)
    lon.set_ticklabel(size=12)
    lon.set_ticks_position('b')
    lon.set_ticklabel_position('b')
    lon.set_axislabel_position('b')
    lon.set_ticks(spacing=30. * u.arcsec)

    lat = overlay['lat']
    lat.set_format_unit(u.arcsec)
    lat.set_ticklabel(size=12)
    lat.set_ticks_position('l')
    lat.set_ticklabel_position('l')
    lat.set_axislabel_position('l')

    # lon.set_axislabel(r'$\Delta$RA',size=12)
    # lat.set_axislabel(r'$\Delta$DEC',size=12)

    lon.set_axislabel(None)
    lat.set_axislabel(None)

    lon.set_auto_axislabel(False)
    lat.set_auto_axislabel(False)
    lon.set_axislabel('')
    lat.set_axislabel('')
    ra.set_axislabel('')
    dec.set_axislabel('')
    # lat.set_ticklabel_visible(False)
    # lon.set(xlabel=None)


    # fig1.set_xticklabels([])
    # fig1.set_yticklabels([])


def apply_threshold_correction(array, lower_threshold):
    """
    Sets values in a 2D array to NaN if they are outside the given threshold range.

    Args:
        array (numpy.ndarray): The input 2D array.
        lower_threshold (float): The lower bound of the threshold.
        upper_threshold (float): The upper bound of the threshold.

    Returns:
        numpy.ndarray: The modified array with values outside the thresholds set to NaN.
    """
    # Create a copy of the array to avoid modifying the original
    array_copy = array.copy()

    # Apply conditions to set values outside the threshold to NaN
    # array_copy[(array < np.percentile(array.reshape(-1),lower_threshold) )] = np.percentile(array.reshape(-1),lower_threshold)
    # array_copy[(array > np.percentile(array.reshape(-1),upper_threshold))] = np.percentile(array.reshape(-1),upper_threshold)

    return array_copy - np.nanpercentile(array.reshape(-1),lower_threshold)

def open_files(folder_fits,source_file):

    # List all .fits files in the directory
    file_directory = os.path.join(folder_fits,source_file)
    search_files_in_directory = sorted([f for f in os.listdir(file_directory) if f.endswith('.fits')])

    J_name = os.path.join(folder_fits,source_file,search_files_in_directory[1])
    H_name = os.path.join(folder_fits,source_file,search_files_in_directory[0])
    K_name = os.path.join(folder_fits,source_file,search_files_in_directory[2])

    ###safety check
    if 'aJ' not in search_files_in_directory[1] or 'aH' not in search_files_in_directory[0]:
        print('something is wrong with the name files')
        print(search_files_in_directory[1],search_files_in_directory[0])
        return

    Jfile_header = fits.getheader(J_name)

    J_data = fits.getdata(J_name)
    H_data = fits.getdata(H_name)
    K_data = fits.getdata(K_name)

    wcs = WCS(Jfile_header)

    return J_data,H_data,K_data,wcs


def histograms():
    reshaped_i = i.reshape(-1)
    bins = np.arange(400, 600, 10)
    print('median', np.nanmedian(reshaped_i))
    print('1%', np.percentile(reshaped_i, 0.5))
    print('99%', np.percentile(reshaped_i, 99.5))
    print(len(reshaped_i))
    plt.hist(reshaped_i, bins)
    plt.yscale('log')
    plt.show()

def twomass_image_to_png(fits_files,source_name,plot=True, pctl= 100):

    J_data,H_data,K_data,wcs = open_files(fits_files,source_name)

    new_g = apply_threshold_correction(J_data, lower_threshold=5)
    new_r = apply_threshold_correction(H_data, lower_threshold=5)
    new_i = apply_threshold_correction(K_data, lower_threshold=5)

    #### First transform folder name to simbad name
    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt', search_word=source_name)
    #### Then get the coordinates based on the SIMBAD name
    skycoord_object = get_icrs_coordinates(object_name=simbad_name)

    # x_center, y_center =wcs.world_to_pixel(skycoord_object) ## This one if 2D cube

    plt.figure(figsize=(6, 7))
    fig1 = plt.subplot(projection=wcs)

    # pctl = 99.0
    maximum = 0

    for img in [new_i,new_r,new_g]:
        val = np.nanpercentile(img,pctl)
        print(val)
        if val > maximum:
            maximum = val

    rgb = make_rgb(new_i, new_r, new_g, interval=ManualInterval(vmin=0, vmax=maximum))

    fig1.imshow(rgb, origin='lower')

    offset_coordinates(fig1, skycoord_object)


    plt.savefig('Figures/2MASS/'+source_name+'_upper_'+str(pctl).replace('.','p')+'.png', bbox_inches='tight',dpi=300)
    # plt.axis('square')

    if plot:
        plt.show()

def prepare_sources_to_download(source_name):

    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt', search_word=source_name)
    skycoord_object = get_icrs_coordinates(object_name=simbad_name)


    # find_simbad_source_in_file()
    ra_deg = round(skycoord_object.ra.deg,11)
    dec_deg = round(skycoord_object.dec.deg,11)

    def save_to_file(save_filename, new_values):

        formatted_entry = (
            f"{'':<1}"  # Source name in 20 bytes
            f"{new_values[0]:<15}"  # Source name in 20 bytes
            f"{new_values[1]:<16}"  # 
            f"{new_values[2]:<16}"  #
            f"{new_values[3]:<9}"  #
            f"{new_values[4]:<9}"  #

        )
        # Read the file if it exists, otherwise start with a header
        try:
            with open(save_filename, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:

            # If the file doesn't exist, start with a formatted header line
            header = (
                f"{'|  id   ':<14}"  # Source name in 20 bytes
                f"{'|    ra ':<16}"  # 
                f"{'|     dec ':<16}"  #
                f"{'| best':<9}"  #
                f"{'|  size':<9}"  #
                f"{' |':<1}\n"  #
                f"{'|   char    ':<14}"  # Source name in 20 bytes
                f"{'|     double ':<16}"  # 
                f"{'|     double  ':<16}"  #
                f"{'| char':<9}"  #
                f"{'| double':<9}"  #
                f"{' |':<1}\n"  #
                f"{'|       ':<14}"  # Source name in 20 bytes
                f"{'|     deg ':<16}"  # 
                f"{'|     deg  ':<16}"  #
                f"{'|  ':<9}"  #
                f"{'| arcsec':<9}"  #
                f"{' |':<1}\n"  #
                f"{'|  null   ':<14}"  # Source name in 20 bytes
                f"{'|  null ':<16}"  # 
                f"{'|  null  ':<16}"  #
                f"{'|  null':<9}"  #
                f"{'|  null':<9}"  #
                f"{' |':<1}\n"  #

            )

            lines = [header]

        first_value = new_values[0]
        found = False

        # Check if the first value is already in the file and update the line if it exists
        for i, line in enumerate(lines):
            # Check if this line starts with the source name
            # if line.startswith(f"{# first_value:<17}"):
            if first_value in line:
                print('This source already exists ',first_value)
                lines[i] = formatted_entry + "\n"
                found = True
                break

        # If the source name was not found, append the new formatted entry
        if not found:
            lines.append(formatted_entry + "\n")

        # Write the updated content back to the file
        with open(save_filename, 'w') as file:
            file.writelines(lines)

    values=[simbad_name.strip(),ra_deg,dec_deg,'y',180]

    save_to_file(save_filename='2mass_prep_table2.txt',new_values=values)

def mass_produce_2mass_images(folder, pctl= 100):
    folder_list = sorted(next(os.walk(folder))[1])  # List of subfolders
    print("Folders found:", folder_list)

    for sources in folder_list:
        try:
            file_directory = os.path.join(folder, sources)
            search_files_in_directory = sorted([f for f in os.listdir(file_directory) if f.endswith('.fits')])
            print(file_directory)

            # Check if the necessary file exists before running the function
            if len(search_files_in_directory) <1:
                print(f"No files in : {sources}. Skipping this folder.")
                continue  # Move to the next folder if the file doesn't exist

            # Generate 2MASS image
            twomass_image_to_png(folder, sources,plot=False,pctl=pctl)

        except IndexError as err:
            print(f"Map for {sources} was not produced. Check the moment maps.")
            print(f"An error occurred: {err}")

        except Exception as e:
            print(f"An unexpected error occurred with {sources}: {e}")


"""
To run this program need to actvivate the astropy environment
conda activate astropy
python3 2MASS_images.py
"""
if __name__ == "__main__":
    fits_files_folder='2MASS_files'
    source_name='T-Tauri'
    twomass_image_to_png(fits_files_folder,source_name,pctl=99.0)
    # mass_produce_2mass_images(folder=fits_files_folder,pctl=98.5)

    # sources_to_download_file='text_files/names_to_simbad_names.txt'
    # prepare_sources_to_download(source_name)

