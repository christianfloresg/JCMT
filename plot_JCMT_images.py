import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Astropy modules to deal with coordinates
from astropy.wcs import WCS
from astropy.io import fits

from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
import BTS
import seaborn as sns

class DataAnalysis:
    """
    Class used to obtain the important parameters from the data
    and the physical or meaningful quantities.
    INPUT
    -----
        path - path to the image directory
        filename- name of the file
        continuum - if True the image will be treated as a single 2D array
                    if false, the image will be treated as a cube with the 3rd axis
                    the spectral axis
    """

    def __init__(self, path, filename):

        self.image = 1

        try:
            data_cube = fits.open(os.path.join(path, filename))
        except:
            data_cube = fits.open(os.path.join(path, filename + '.fits'))

        self.filename = filename
        self.header = data_cube[0].header
        self.ppv_data = data_cube[0].data


        # If the data has a 4 dimension, turn it into 3D
        if (np.shape(data_cube[0].data)[0] == 1):
            self.ppv_data = data_cube[0].data[0, :, :, :]

        self.nx = self.header['NAXIS1']
        self.ny = self.header['NAXIS2']

        self.cdelt_ra = self.header['CDELT1'] * 3600



        try:
            self.nz = self.header['NAXIS3']
            self.vel = self.get_vel(self.header)
            dv = self.vel[1] - self.vel[0]
            if (dv < 0):
                dv = dv * -1

            self.molecule = self.header['MOLECULE'].split(' ')[0].replace('-', '')

            s_name = self.header['OBJECT'].strip()
            self.source_name = s_name.replace(' ', '_')

        except:
            print('This is a 2D image')

        self.wcs = WCS(self.header)

    def get_vel(self, head):


        if "v" in head['CTYPE3'].lower():

            refnv = head["CRPIX3"]
            refv = head["CRVAL3"]
            dv = head["CDELT3"]
            ### Construct the velocity axis

            vel = np.zeros(head["NAXIS3"])
            for ii in range(0, len(vel)):
                vel[ii] = refv + (ii - refnv + 1) * dv

            return vel

        else:

            print("The CTYPE3 variable in the fitsfile header does not start with F for frequency or V for velocity")
            return

def create_shell_script_moment_maps(path_to_folder,sdf_name,source_name,molec):
    """
    Create the shell script that runs all the necessary tasks in kappa
    to go from antenna temperature to Main Beam temperature
    change the units to Km/s
    re-sample the datacube

    :param path_to_folder: folder of the .sdf files starting with PROGRAM NUMBER
    :param sdf_name: name of the reduced data (NO .sdf extension)
    :param source_name: source name
    :param molec: molecule name - either HCO+ or C18O
    :return:
    """
    if '.sdf' in sdf_name:
        sdf_name = sdf_name.strip('.sdf')

    file = open(os.path.join('.',"moment_map_preparation.sh"), "w")
    file.write('#!/usr/bin/env bash \n')
    file.write('export STARLINK_DIR=/Users/christianflores/Programs/star-2023A \n')
    file.write('source $STARLINK_DIR/etc/profile  \n')
    file.write('cd /Users/christianflores/Documents/work/Astronomy_data/JCMT/'+path_to_folder+'\n')
    file.write('kappa \n')
    file.write('INPUTNAME="'+sdf_name+'" \n')
    file.write('SOURCE="'+source_name+'" \n')
    file.write('MOLEC="'+molec+'" \n')
    file.write('RESA="_resampled" \n')
    file.write('SOURCENAME=$SOURCE"_"$MOLEC \n')
    file.write('RESAMP=$SOURCE"_"$MOLEC$RESA \n')
    # file.write('RESAMP=$SOURCE"_"$MOLEC \n')
    file.write('cdiv in=$INPUTNAME.sdf scalar=0.63 out=$SOURCENAME.sdf \n')
    file.write('setunits $SOURCENAME.sdf units=\"K km/s \" \n')
    file.write('sqorst in=$SOURCENAME.sdf out=$RESAMP.sdf factors="[6,6,1] conserve" \n')
    file.write('convert \n')
    file.write('ndf2fits in=$RESAMP.sdf out=$RESAMP.fits \n')
    file.write('mkdir /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/'+source_name+ '\n')
    file.write('mv $RESAMP.sdf /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/'+source_name+ '\n')
    file.write('mv $RESAMP.fits /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/'+source_name+ ' \n')
    file.write('cd /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/'+source_name+ ' \n')
    file.write('mv $RESAMP.sdf $SOURCENAME.sdf \n')
    file.write('mv $RESAMP.fits $SOURCENAME.fits \n')
    file.close()

def run_moment_map_shell_script(path_to_folder):
    """
    Run the script moment_map_preparation.sh
    :param path_to_folder:
    :return:
    """
    subprocess.check_call(['bash',os.path.join(path_to_folder,"moment_map_preparation.sh")], cwd=path_to_folder)


def create_moment_masking_parameterfile(source_name, fits_file_name,cube_param_name='Fit_cube.param'):
    '''
    Copy the parameter file needed to run BTS and create moment maps
    Modify the files themselves so they have the appropriate input data
    source
    folder_file: the directory of the folder
    fits_file_name: name of the fits file datacube that wants to be used
    for moment map creating
    cube_param_name: default is Fit_cube.param in the same directory as this file
    '''

    folder_file= os.path.join('sdf_and_fits',source_name)#  sdf_and_fits/'+source_name+'/'
    data_cube = DataAnalysis(folder_file, fits_file_name + '.fits')
    molecule = data_cube.molecule
    # source_name =data_cube.source_name

    ### copying the file
    moment_param_filename = source_name + '_' + molecule + '_moments.param'  ## Name of the cube.param file

    full_path_moment_param_filename = os.path.join(folder_file, moment_param_filename)  ## full name including path
    copy_text_files(cube_param_name, full_path_moment_param_filename)

    ### modifying the file
    new_fits_path = os.path.join(folder_file, fits_file_name+'.fits')
    replace_line(full_path_moment_param_filename, 'data_in_file_name', new_fits_path)
    save_folder = os.path.join('moment_maps', source_name)
    output_base = os.path.join(save_folder, source_name + '_' + molecule)
    replace_line(full_path_moment_param_filename, 'output_base', output_base)

    ### make directory to save file if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    return full_path_moment_param_filename


def copy_text_files(source, destination):
    '''
    # Copy the content of the moment masking parameter to the folder of the fits files
    # source to destination is the folder
    '''

    try:
        shutil.copyfile(source, destination)
        print("File copied successfully.")

    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

    # If destination is a directory.
    except IsADirectoryError:
        print("Destination is a directory.")

    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

    # For other errors
    except:
        print("Error occurred while copying file.")


def replace_line(file_name, key_text, new_text):
    lines = open(file_name, 'r').readlines()
    for count, line in enumerate(lines):
        # for line in lines:
        if key_text in line:
            text_to_change = line.split()[2]
            replaced_line = line.replace(text_to_change, new_text)
            line_num = count
            # print(text_to_change)
    lines[line_num] = replaced_line
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def run_BTS(param_file):
    # Read in the relevant parameter file
    param = BTS.read_parameters(param_file)
    print(param)
    # Run the function to make the moments using the moment-masking technique
    BTS.make_moments(param)

def integrated_intensity(path, filename):
    '''
    Get the flux density (Jy) at the position center over 1 beam
    :param path:
    :param filename:
    :return:
    '''

    return 1


def get_icrs_coordinates(object_name):
    # Initialize Simbad object and customize output to include coordinates
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('coordinates')

    # Query Simbad for the object
    result_table = custom_simbad.query_object(object_name)

    # Check if the query returned any results
    if result_table is None:
        print(f"Object '{object_name}' not found in SIMBAD.")
        return None

    # Extract the RA and DEC columns from the result
    ra = result_table['RA'][0]  # Right Ascension in HMS (hours, minutes, seconds)
    dec = result_table['DEC'][0]  # Declination in DMS (degrees, arcminutes, arcseconds)

    # Convert RA and DEC to a SkyCoord object in the ICRS frame
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='icrs')

    # Return the ICRS coordinates in degrees
    return coord


def make_averaged_spectrum_data(source_name, molecule):
    """
    Average spectrum of the whole cube.
    """

    filename = source_name+'_'+molecule
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')
    moment_0 = DataAnalysis(os.path.join('moment_maps',source_name), filename+'_mom0.fits')

    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05
    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635
    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    pix_per_beam = aperture_radius**2*np.pi / (4*np.log(2)*data_cube.cdelt_ra**2) # pix-per-beam = beam_size/pix_area
    velocity = data_cube.vel
    cube = data_cube.ppv_data

    averaged_spectrum = np.nanmean(cube, axis=(1,2)) * pix_per_beam

    return averaged_spectrum, velocity


def make_central_spectrum_data(source_name, molecule):
    """
    Average spectrum of the central beam.

    """

    filename = source_name+'_'+molecule
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')
    moment_0 = DataAnalysis(os.path.join('moment_maps',source_name), filename+'_mom0.fits')

    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05
    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635
    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    pix_per_beam = aperture_radius**2*np.pi / (4*np.log(2)*data_cube.cdelt_ra**2) # pix-per-beam = beam_size/pix_area
    velocity = data_cube.vel

    x_center, y_center = moment_0.wcs.world_to_pixel(skycoord_object)
    # print(x_center, y_center)

    # Initialize a list to store the pixel values within the aperture
    center_beam_values = []

    # Iterate over a square region, but filter by distance to make it circular
    for xx in range(int(x_center - aperture_radius), int(x_center + aperture_radius) + 1):
        for yy in range(int(y_center - aperture_radius), int(y_center + aperture_radius) + 1):
            # Calculate the distance from the center
            distance = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)

            # Check if the distance is within the aperture radius
            if distance <= aperture_radius:
                # Append the data at this pixel position
                center_beam_values.append(data_cube.ppv_data[:, xx, yy])

    # Convert center_beam_values to a NumPy array for easy manipulation
    center_beam_values = np.array(center_beam_values)

    average_spectrum = np.nanmean(center_beam_values, axis=0) * pix_per_beam

    return average_spectrum, velocity

def plot_spectrum(source_name, molecule,save=False):
    """
    This one plots the average spectrum
    """
    # spectrum, velocity = make_central_spectrum_data(source_name, molecule)
    spectrum, velocity =make_averaged_spectrum_data(source_name,molecule)

    plt.figure()
    # plt.title("Averaged Spectrum ("+mole_name+") @"+dir_each)
    plt.xlabel("velocity (km/s)")
    plt.ylabel("Averaged Spectrum (K)")
    # Set the value for horizontal line
    y_horizontal_line = 0
    # plt.axhline(y_horizontal_line, color='red', linestyle='-')
#     plt.axvline(Vsys, color='red', linestyle='--')
    plt.plot(velocity,spectrum,"-",color="black",lw=1)
    plt.tick_params(axis='both', direction='in')
    plt.xlim(-10,30)
    if save:
        plt.savefig(os.path.join('Figures', 'spectrum_'+filename), bbox_inches='tight')
    plt.show()


def plot_moment_zero_map(filename,source_name,sky_cord_object=False,save=False):
    '''
    Create moment maps using the python package bettermoments.
    Currently only moment 0 and 8 work. Some unknown issues with the velocity
    ones.
    Need to give the data, velocity, and rms levels.
    The moment maps will be computed using a given velocity position
    previously calculated and a velocity dispersion given from gaussian fit.
    A 3 sigma is what we will use for now.
    We adopt a sigma clipping of 1*rms.
    :param data:
    :param velax:
    :param rms:
    :param x0:
    :param sigma:
    :param moment_number:
    :param save:
    :return:
    '''


    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')
    moment_0 = DataAnalysis(os.path.join('moment_maps',source_name), filename+'_mom0.fits')

    print('molecule ',data_cube.molecule)
    ### Here I can go from sky position to pixel coordinates

    image_mom_0 = moment_0.ppv_data

    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05
        cmap = sns.color_palette("YlOrBr",as_cmap=True)

    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635
        cmap = sns.color_palette("YlGnBu",as_cmap=True)

    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    pix_per_beam = aperture_radius**2*np.pi / (4*np.log(2)*data_cube.cdelt_ra**2) # pix-per-beam = beam_size/pix_area
    image_mom_0 = image_mom_0 * pix_per_beam
    # image_mom_0[image_mom_0 == 0] = np.nan

    peak = np.nanmax(image_mom_0)
    levels = np.array([0.2,  0.5, 0.8, 0.95])
    levels = levels * peak


    ## Moment zero
    fig1 = plt.subplot(projection=moment_0.wcs)
    mom0_im = fig1.imshow(image_mom_0, cmap=cmap, origin='lower')#,vmax=1)
    # divider = make_axes_locatable(fig1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mom0_im, fraction=0.048, pad=0.04, label='Integrated Intensity (K * km/s)')
    contour = fig1.contour(image_mom_0, levels=levels, colors="black")
    # plt.clabel(contour, inline=True, fontsize=8)

    fig1.set_xlabel('RA',size=12)
    fig1.set_ylabel('DEC',size=12)

    if sky_cord_object:
        s = SphericalCircle(skycoord_object, aperture_radius * u.arcsec,
                            edgecolor='white', facecolor='none',
                            transform=fig1.get_transform('fk5'))

        fig1.add_patch(s)

    if save:
        plt.savefig(os.path.join('Figures',filename), bbox_inches='tight')
        # plt.savefig(os.path.join('Figures',filename+'_transparent'), bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == "__main__":

    ### Step 1 source name
    containing_folder='M23AP008'
    source_name = 'V347_Aur'
    molecule ='HCO+'
    fits_file_name=source_name+'_'+molecule #'V347_Aur_HCO+'

    ### Step 2
    ## Get the shell script for moment map preparation
    # path_to_folder=containing_folder+'/'+source_name+'/'+molecule+'/reduced/'
    # create_shell_script_moment_maps(path_to_folder,sdf_name='ga20230703_53_1_0p20bin001.sdf',
    #                                 source_name=source_name,molec=molecule)
    # run_moment_map_shell_script(path_to_folder='.')

    ### Step 3
    #### run the BTS to create moment maps
    # BTS_param_file = create_moment_masking_parameterfile(source_name, fits_file_name)
    # run_BTS(BTS_param_file)

    ### Step 3
    ### Plot the maps
    # skycoord_object = SkyCoord('04 56 57.0 +51 30 50.88', unit=(u.hourangle, u.deg))
    # skycoord_object = get_icrs_coordinates('V347 Aur')
    # plot_moment_zero_map(fits_file_name,source_name=source_name,save=True,sky_cord_object=True)

    ### Step 4
    ### Get an averaged spectrum over 1 beam of the map. You MUST give the SIMBAD-searchable name
    skycoord_object = get_icrs_coordinates('V347 Aur')
    plot_spectrum(source_name, molecule)

    ## Step 5
    ### fit the spectrum with a 2D gaussian.
