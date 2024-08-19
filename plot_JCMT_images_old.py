import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import scipy.ndimage.interpolation as interpol
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits as pyfits
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.colors as colors

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

    def __init__(self, path, filename, continuum=False, integrated=False):
        joint_path = os.path.join(path, filename)
        try:
            hdulist = pyfits.open(joint_path)
        except:
            hdulist = pyfits.open(joint_path + ".fits")
        prihdr = hdulist[0].header
        self.header = prihdr
        scidata = hdulist[0].data

        if continuum:
            try:
                self.image = scidata[0, :, :]
            except:
                self.image = scidata[:, :]
        else:
            if integrated:
                self.image = scidata[0, :, :]
            else:
                try:
                    self.image = scidata[:, :, :]
                except:
                    self.image = scidata[0, :, :, :]

                self.nchan = prihdr['NAXIS3']
                self.crval_velo = prihdr['CRVAL3']
                self.cdelt_velo = prihdr['CDELT3']
                self.crpix_velo = prihdr['CRPIX3']
                self.ctype_velo = prihdr['CTYPE3']
                try:
                    self.nu = prihdr['RESTFRQ']
                except:
                    if 2.30e11 < self.crval_velo < 2.31e11:
                        self.nu = 2.3053800e11
                    elif 2.20e11 < self.crval_velo < 2.21e11:
                        self.nu = 2.2039868e11
                    elif 2.19e11 < self.crval_velo < 2.199e11:
                        self.nu = 2.1956035e11
                    elif 2.18e11 < self.crval_velo < 2.189e11:  # I added Formaldehyde
                        self.nu = 2.1822219200e11

                velo_ini = (self.crval_velo - self.cdelt_velo * (
                        self.crpix_velo - 1) - self.nu) / self.nu * 2.99792458e10
                velo_fin = (self.crval_velo + self.cdelt_velo * (
                        self.nchan - self.crpix_velo) - self.nu) / self.nu * 2.99792458e10
                self.velo_array = np.linspace(velo_ini, velo_fin, self.nchan)
                self.cdelt_velo = abs(self.velo_array[0] - self.velo_array[1])

        self.nx = prihdr['NAXIS1']
        self.ny = prihdr['NAXIS2']

        try:
            self.bmaj = prihdr['BMAJ'] * 3600  # bmaj in arcsec
            self.bmin = prihdr['BMIN'] * 3600  # bmin in arcsec
            self.bpa = prihdr['BPA']

        except:
            print('This is a non-convolved model or real data')
            self.bmaj = 0  # bmaj in arcsec
            self.bmin = 0  # bmin in arcsec
            self.bpa = 0

        self.crpix_ra = prihdr['CRPIX1']
        self.crpix_dec = prihdr['CRPIX2']
        self.crval_ra = prihdr['CRVAL1']
        self.crval_dec = prihdr['CRVAL2']
        self.cdelt_ra = prihdr['CDELT1']
        self.cdelt_dec = prihdr['CDELT2']
        self.ImsizeRA = abs(self.cdelt_ra) * (self.nx - 1) * 3600  # image size in arcsec
        self.ImsizeDEC = abs(self.cdelt_dec) * (self.ny - 1) * 3600

        ra_ini = self.crval_ra - abs(self.cdelt_ra) * (self.crpix_ra - 1)
        ra_fin = self.crval_ra + abs(self.cdelt_ra) * (self.nx - self.crpix_ra)
        self.ra_array = np.linspace(ra_ini, ra_fin, self.nx)

        dec_ini = self.crval_dec - abs(self.cdelt_dec) * (self.crpix_dec - 1)
        dec_fin = self.crval_dec + abs(self.cdelt_dec) * (self.ny - self.crpix_dec)
        self.dec_array = np.linspace(dec_ini, dec_fin, self.ny)


def masked_array(data, threshold):
    return (data > threshold).astype(int)




def plot_dust_continuum(continuum_path, continuum_filename,
                          figname='standard_name',save=False):
    """
    Create a plot containing the Moment one map in colors and the dust continuum emission
    overalid in black contours
    """

    continuum_image_file = DataAnalysis(continuum_path, continuum_filename, continuum=True, integrated=False)
    continuum_image = continuum_image_file.image
    ra_array_continuum = (continuum_image_file.ra_array - np.nanmedian(continuum_image_file.ra_array))*3600
    dec_array_continuum = (continuum_image_file.dec_array - np.nanmedian(continuum_image_file.dec_array))*3600

    fig = plt.figure(figsize=(8, 7))
    f = fig.add_subplot(1, 1, 1)
    # current_cmap = plt.cm.get_cmap('hot').copy()
    # current_cmap.set_bad(color='white')

    pax = f.imshow(continuum_image, origin='lower',
                   extent=(ra_array_continuum[0], ra_array_continuum[-1], dec_array_continuum[0], dec_array_continuum[-1]),
                   cmap='jet',vmax=0.8,vmin=0)#AsinhNorm(vmin=np.nanmin(continuum_image),vmax=np.nanmax(continuum_image)))

    # pax = f.contour(ra_array_continuum,dec_array_continuum,continuum_image,colors='w',linewidth=0.8,
    #                 levels=[0.12,0.15,0.2,0.3,0.4,0.45])

    divider = make_axes_locatable(f)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = plt.colorbar(pax, cax=cax, format='%.1f', extend='neither')
    cbar.set_label(r'$\int T_{mb} dV \; [\rm K \, km/s] $', size=22)
    cbar.ax.tick_params(labelsize=18)

    # f.set_xlim(-0.75,0.75)
    # f.set_ylim(-0.75,0.75)
    # f.set_xlim(-4.49,4.49)
    # f.set_ylim(-4.49,4.49)

    elips_cont = Ellipse(xy=(-40, -50), width=15, height=15,
                    angle=90, facecolor='gray', edgecolor='k',lw=2.0,alpha=0.6)

    # f.add_artist(elips_cont)

    f.set_xlabel(r'$\rm \Delta \,RA \:(arcsec)$', fontsize=22)
    f.set_ylabel(r'$\rm \Delta \,DEC \:(arcsec)$', fontsize=22)
    f.tick_params(axis='both', which='both', labelsize=22)

    if save == True:
        plt.savefig(os.path.join(save_fig_folder,figname + '.png'), bbox_inches='tight',transparent=True)
    plt.show()


if __name__ == "__main__":
    save_fig_folder = 'Figures/'
    path = 'JCMT_data'
    continuum_path = 'Continuum'

    filename0 = 'IRAS04369+2539_C18O'

    # filename1 = 'OphIRS63_SB_12CO_robust_0.5_rebinned.high_vel1.image.moment.integrated.fits'
    # filename2 = 'OphIRS63_SB_C18O_robust_-0.5.rebinned.image.moment.weighted_coord.fits'

    # plot_mom_zero_map(path,filename0, figname='aa?',save=False)
    plot_dust_continuum(path,filename0,
                        figname=filename0.split('.')[0]+'_8x7', save=True)

    # plot_mom_one_map(folder_path=path, mom_zero_filename = filename1, mom_one_filename= filename2,
    #                  figname='H2CO_r-05_mom1', save=False)

    # plot_dust_levels(continuum_path=continuum_path, continuum_filename=filename0,
    #                     figname='Continuum_levels_large', save=True)

