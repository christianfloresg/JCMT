import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import scipy.ndimage.interpolation as interpol
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits as pyfits
import matplotlib.colors as colors

#Astropy modules to deal with coordinates
from astropy.wcs import WCS
from astropy.wcs import Wcsprm
from astropy.io import fits
from astropy.wcs import utils

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle


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

            self.molecule = self.header['MOLECULE']


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


def integrated_intensity(path, filename):
    '''
    Get the flux density (Jy) at the position center over 1 beam
    :param path:
    :param filename:
    :return:
    '''

    return 1

def make_average_spectrum_data(path, filename):
    """
    Average spectrum of the whole cube.
    """
    data_cube = DataAnalysis(path, filename)
    moment_0 = DataAnalysis(path, filename+'_mom0.fits')

    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05
    elif 'C18O ' in data_cube.molecule:
        aperture_radius = 7.635
    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    pix_per_beam = aperture_radius**2*np.pi / (4*np.log(2)*data_cube.cdelt_ra**2) # pix-per-beam = beam_size/pix_area
    x_center,y_center = moment_0.wcs.world_to_pixel(skycoord_object)
    # print(x,y)
    velocity = data_cube.vel

    ### This needs to be changed by selecting pixels within 1 aperture,
    ### Need to work on the code below
    ### I could use the same for the integration of the emission.
    # center_beam_values=[]
    # for xx in range():
    #     for yy in range():
    #         if xx < x_center + aperture_radius and yy < y_center + aperture_radius:
    #             center_beam_values.append(data_cube.ppv_data[:,xx,yy])

    image = data_cube.ppv_data[:,int(y)-6:int(y)+6,int(x)-6:int(x)+6]* pix_per_beam
    average_spectrum = np.nanmedian(image, axis=(1, 2))

    return average_spectrum, velocity

def plot_average_spectrum(path,filename,save=False):
    """
    This one plots the average spectrum
    """
    spectrum, velocity = make_average_spectrum_data(path,filename)
    plt.figure()
    # plt.title("Averaged Spectrum ("+mole_name+") @"+dir_each)
    plt.xlabel("velocity [km/s]")
    plt.ylabel("Intensity")
    # Set the value for horizontal line
    y_horizontal_line = 0
    plt.axhline(y_horizontal_line, color='red', linestyle='-')
#     plt.axvline(Vsys, color='red', linestyle='--')
    plt.plot(velocity,spectrum,"-",color="black",lw=1)
    plt.tick_params(axis='both', direction='in')
    plt.xlim(-10,30)
    if save:
        plt.savefig(os.path.join('Figures', 'spectrum_'+filename), bbox_inches='tight')
    plt.show()


def plot_moment_zero_map(path,filename,save=False):
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


    data_cube = DataAnalysis(path, filename+'.fits')
    moment_0 = DataAnalysis(path, filename+'_mom0.fits')

    ### Here I can go from sky position to pixel coordinates

    image_mom_0 = moment_0.ppv_data

    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05
    elif 'C18O ' in data_cube.molecule:
        aperture_radius = 7.635
    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    pix_per_beam = aperture_radius**2*np.pi / (4*np.log(2)*data_cube.cdelt_ra**2) # pix-per-beam = beam_size/pix_area
    image_mom_0 = image_mom_0 * pix_per_beam

    peak = np.nanmax(image_mom_0)
    levels = np.array([0.2,  0.5, 0.8, 0.95])
    levels = levels * peak

    ## Moment zero
    fig1 = plt.subplot(projection=moment_0.wcs)
    mom0_im = fig1.imshow(image_mom_0, cmap="viridis", origin='lower',vmax=0.7)
    # divider = make_axes_locatable(fig1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mom0_im, fraction=0.048, pad=0.04, label='Integrated Intensity (K * km/s)')
    contour = fig1.contour(image_mom_0, levels=levels, colors="black")
    plt.clabel(contour, inline=True, fontsize=8)

    # skycoord_object = SkyCoord('04 56 57.0 +51 30 50.88', unit=(u.hourangle, u.deg))
    s = SphericalCircle(skycoord_object, aperture_radius * u.arcsec,
                        edgecolor='white', facecolor='none',
                        transform=fig1.get_transform('fk5'))

    fig1.add_patch(s)

    if save:
        plt.savefig(os.path.join('Figures',filename), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    path = '.'
    filename='V347_AurHCO+_resamp'
    skycoord_object = SkyCoord('04 56 57.0 +51 30 50.88', unit=(u.hourangle, u.deg))

    plot_moment_zero_map(path, filename,save=True)
    # plot_average_spectrum(path, filename,save=True)
    # make_average_spectrum_data(path, filename)