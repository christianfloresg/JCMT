import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Astropy modules to deal with coordinates
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
import BTS

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
                print('We had to flip the velocity axis')
                dv = dv * -1
                self.vel = np.flip(self.vel)
                self.ppv_data = np.flip(self.ppv_data, axis=0)

            self.molecule = self.header['MOLECULE'].split(' ')[0].replace('-', '')

            s_name = self.header['OBJECT'].strip()
            self.source_name = s_name.replace(' ', '_')

            # temp_change = self.header['SPECSYS'] = 'LSRK'  # Be cautious with this change if the original frame is essential
            # data_cube.wcs = WCS(temp_change)

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
    file.write('convert \n')
    file.write('cdiv in=$INPUTNAME.sdf scalar=0.63 out=$SOURCENAME.sdf \n')
    file.write('setunits $SOURCENAME.sdf units=\"K km/s \" \n')
    file.write('ndf2fits in=$SOURCENAME.sdf out=$SOURCENAME"_original".fits \n')
    file.write('sqorst in=$SOURCENAME.sdf out=$RESAMP.sdf factors="[6,6,1] conserve" \n')
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


if __name__ == "__main__":

    ### Step 1 source name
    containing_folder='M24AH15A'
    source_name = 'DG-Tau'
    # molecule ='HCO+'
    molecule ='C18O'
    fits_file_name=source_name+'_'+molecule #'V347_Aur_HCO+'

    ### Step 2
    # Get the shell script for moment map preparation
    path_to_folder=containing_folder+'/'+source_name+'/'+molecule+'/reduced/'
    create_shell_script_moment_maps(path_to_folder,sdf_name='ga20240919_40_1_0p20bin001.sdf',
                                    source_name=source_name,molec=molecule)
    run_moment_map_shell_script(path_to_folder='.')


    ### Step 3
    ### run the BTS to create moment maps
    # BTS_param_file = create_moment_masking_parameterfile(source_name, fits_file_name)
    # run_BTS(BTS_param_file)


