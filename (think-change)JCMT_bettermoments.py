import bettermoments as bm
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits as pyfits
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
import numpy.ma as ma
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.coordinates import ICRS
from astropy import units as u

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

        wcs = WCS(prihdr)
        self.header = prihdr
        scidata = hdulist[0].data

        # print(wcs)
        # celestial, spectral = wcs.all_pix2world(100., 100., 1)
        # print(celestial)

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection=celestial)
        # ax.imshow(scidata[0,:,:])
        # plt.show()

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

        ra_def = self.crval_ra - (self.cdelt_ra * self.crpix_ra)
        # ra_fin = self.crval_ra + self.cdelt_ra * (self.nx - self.crpix_ra)
        x_array = np.arange(0.5,self.nx+1)
        self.ra_array = self.cdelt_ra*x_array + ra_def
        # print(x_array)

        y_array = np.arange(0.5,self.ny+1)

        dec_def = self.crval_dec - self.cdelt_dec * (self.crpix_dec)
        # dec_fin = self.crval_dec + abs(self.cdelt_dec) * (self.ny - self.crpix_dec)
        self.dec_array = self.cdelt_dec*y_array+dec_def


        # print(self.cdelt_dec*51.5+dec_def)
        # print(self.cdelt_ra*51.5 + ra_def)

def gauss(x, H, A, x0, sigma):
    """
    simple gaussian function
    :param x: variable
    :param H: addition constant
    :param A: multiplicative constant
    :param x0: center of gaussian
    :param sigma: standard deviation
    :return: function
    """
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(xdata, ydata):
    """
    Fit a Gaussian giving some initial parameters
    compute the parameters of the gauss() function
    :param xdata: wavelength array
    :param ydata: flux array
    :return: only sigma and center position
    """
    sigma_guess = 1
    index_position=find_nearest_index(ydata,np.nanmax(ydata))
    position_guess = xdata[index_position]
    popt, pcov = curve_fit(gauss, xdata, ydata, p0=[min(ydata), max(ydata), position_guess, sigma_guess])

    return popt


def spectrum_properties(spectrum,velax,velocity_min, velocity_max,nsigma=3,plot=True,save=False):
    '''
    Compute the std an center position of a gaussian
    starting from the datacube
    :param data:
    :param kwargs: the initial velocity  range
    :return:
    '''

    broad_lower_idx= find_nearest_index(array=velax,value=velocity_max)
    broad_upper_idx= find_nearest_index(array=velax, value=velocity_min)

    shortened_vel=velax[broad_lower_idx:broad_upper_idx]
    shortened_flux=spectrum[broad_lower_idx:broad_upper_idx]


    H, A, x0, sigma = gauss_fit(shortened_vel,shortened_flux)

    if abs(sigma) > 50 or abs(H)>10:
        H = 0
        sigma = 1
        print('THIS FIT DID NOT FIND SOMETHING REALISTIC')

    FWHM = 2.35482 * sigma


    print('The offset of the gaussian baseline is', H)
    print('The center of the gaussian fit is', x0)
    print('The sigma of the gaussian fit is', sigma)
    print('The maximum intensity of the gaussian fit is', H + A)
    print('The Amplitude of the gaussian fit is', A)
    print('The FWHM of the gaussian fit is', FWHM)

    if plot:
        plt.plot(shortened_vel, shortened_flux, 'ko', label='data')
        plt.plot(shortened_vel, gauss(shortened_vel, H, A, x0, sigma), '--r', label='fit')
        plt.axvline(x=x0+nsigma*sigma,color='red',label=str(nsigma)+r'$\sigma$')
        plt.axvline(x=x0-nsigma*sigma,color='red')
        plt.legend()
        plt.title('Gaussian fit,  $f(x) = A e^{(-(x-x_0)^2/(2sigma^2))}$')
        plt.xlabel('velocity')
        plt.ylabel('Intensity (A)')
        if save:
            try:
                plt.savefig(os.path.join('fitting_line', filename.strip('.fits') + '_spectrum_fit.png'),
                            bbox_inches='tight',
                            transparent=False)
            except:
                plt.savefig(os.path.join('fitting_line', filename.strip('.fits') + '_spectrum_fit.png'),
                            bbox_inches='tight',
                            transparent=False)
        plt.show()

    return sigma, x0

def create_spectrum(path, filename, velocity_min, velocity_max, source_position, aperture_radius=7.5):
    """
    create a spectrum centered in a given position using an indicated aperture.
    The position finding of the source could be improved using something like WCS and astropy

    :param path:
    :param filename:
    :param velocity_min:
    :param velocity_max:
    :param source_position:
    :param aperture_radius:
    :return: spectrum, velocity, the width (1 sigma) of the gaussian fit to the spectrum, the center position
    """
    source_position_fk5 = source_position.transform_to('fk5')
    ra_degree,dec_degree = source_position_fk5.ra.degree,source_position_fk5.dec.degree

    smooth_factor = 8

    data, velax = bm.load_cube(os.path.join(path, filename))

    smoothed_data = bm.smooth_data(data=data, smooth=smooth_factor, polyorder=0)
    calculate_signal_to_noise(path, filename, velocity_min, velocity_max) ## Ned to check this one

    # smoothed_rms = bm.estimate_RMS(data=smoothed_data, N=500)

    image_file = DataAnalysis(path, filename, continuum=False, integrated=False)
    pix_size = abs(image_file.cdelt_ra) * 3600 # from degrees to arcsec
    radius_size_in_pixels = round(aperture_radius / pix_size / 2., 0)  ## /2 is because we need radius not diameter

    # find the pixel position in the image close to your source coordinates
    ra_pix_pos = find_nearest_index(array=image_file.ra_array, value=ra_degree)
    dec_pix_pos = find_nearest_index(array=image_file.dec_array, value=dec_degree)

    """
    This way of defining RA and DEC actually does not work close to the edge dues to non-linear stuff I guess
    Need to look back into Astropy WCS but maybe later
    """
    print("pix-position")
    print(ra_pix_pos, dec_pix_pos)

    ## Mask all the pixels outside the central region defined for the aperture
    mask_2d = aperture_definition(smoothed_data, radius=int(radius_size_in_pixels),
                                  center_position=(ra_pix_pos, dec_pix_pos))

    mask_3d = np.repeat(mask_2d[np.newaxis, :, :], np.shape(smoothed_data)[0], axis=0)
    mx = ma.masked_array(smoothed_data, mask=mask_3d)

    spectrum = np.nansum(np.nansum(mx, axis=2), axis=1)

    sigma, position = spectrum_properties(spectrum, velax, velocity_min=velocity_min,
                                          velocity_max=velocity_max, nsigma=6, plot=False)

    return spectrum,velax,sigma,position

def plot_spectra(*argv, velocity_min, velocity_max, source_position, aperture_radius=7.5,save=False):

    """
    Plot the generated spectrum

    :param argv:
    :param velocity_min:
    :param velocity_max:
    :param image_position:
    :param save:
    :return:
    """

    fig = plt.figure(figsize=(8, 7))
    f = fig.add_subplot(1, 1, 1)

    line_colors=['r','k','']
    count=0


    for path,filename in argv:

        spectrum, velax, sigma, position = create_spectrum(path, filename, velocity_min, velocity_max, source_position, aperture_radius)

        lower_idx= find_nearest_index(array=velax,value=position+10)
        upper_idx= find_nearest_index(array=velax, value=position-10)


        if 'C18O' in path or 'C18O' in filename:
            f.step(velax,spectrum,color=line_colors[count],linewidth=2,label=r'C$^{18}$O')
        else:
            f.step(velax,spectrum,color='k',linewidth=2,label=r'HCO$^{+}$')
        count=count+1

    f.legend(fontsize=16)
    f.set_xlim(velax[upper_idx],velax[lower_idx])
    f.set_title('Aperture ' +str(aperture_radius) +' arcsec', fontsize=20),
    f.set_xlabel('Velocity '+r'$\rm (km \, s^{-1})$', fontsize=20)
    f.set_ylabel('Integrated spectrum ' +r'$\rm (K)$', fontsize=20)
    f.tick_params(axis='both', which='both', labelsize=16)

    if save == True:
        plt.savefig(os.path.join(save_folder,filename1.strip('.fits')+'_spectrum_in_K.png'), bbox_inches='tight',transparent=True)

    plt.show()

def spectrum_from_image_position(path,filename,image_position):
    '''
    Give the abbreviation to see what quadrant the spectrum should be chosen from
    the first letter corresponds to y axis, second to x-xis
    :param path:
    :param filename:
    :param abreviation:
    :return:
    '''
    image_file = DataAnalysis(path, filename, continuum=False, integrated=False)

    nx_size = image_file.nx
    zero = nx_size/2
    neg = nx_size/4
    pos = nx_size/4+ zero
    length = nx_size/4-1

    center = int(zero - length), int(zero + length)
    left = int(neg -length), int(neg +length)
    right = int(pos-length), int(pos+length)
    broad = int(zero - length*1.5), int(zero + length*1.5)

    if image_position=='cc':

        print(center,center)
        return center,center

    elif image_position == 'cl':
        return center,left
    elif image_position == 'cr':
        return center,right
    elif image_position == 'tc':
        return right,center
    elif image_position == 'tl':
        return right,left
    elif image_position == 'tr':
        return right,right
    elif image_position == 'bc':
        return left,center
    elif image_position == 'bl':
        return left,left
    elif image_position == 'br':
        return left,right
    elif image_position =='broad':
        return broad,broad
    else:
        raise Exception("Sorry, only certain names are allowed")


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def creates_moment_zero_map(data,velocity,rms):
    moments = bm.collapse_zeroth(velax=velocity, data=data, rms=rms)
    return moments

    # plt.imshow(moments[0],origin='lower',cmap='bone')
    # plt.colorbar()

    # plt.show()

def creates_moment_one_map(data,velocity,rms):
    moments = bm.collapse_first(velax=velocity, data=data, rms=rms)
    # moments = bm.methods.collapse_quadratic(velax=velocity, data=masked_data, rms=smoothed_rms)
    return moments

    # plt.imshow(moments[0],origin='lower',cmap='jet')


    # cube = SpectralCube.read(os.path.join(path,filename))
    # slab = cube.spectral_slab(7 * up.speed, 9 * up.speed)
    # m1 = slab.moment(order=1)
    # masked_slab = slab.with_mask(slab > 1 * Jy / beam)
    # plt.imshow(m1,origin='lower',cmap='jet')

    # plt.colorbar()
    # plt.show()

def creates_moment_eighth_map(data,velocity,rms):
    moments = bm.collapse_eighth(velax=velocity, data=data, rms=rms)
    return moments

    # plt.imshow(moments[0],origin='lower',cmap='bone',vmin=smoothed_rms,vmax=np.nanmax(smoothed_data))
    # plt.colorbar()

    # plt.contour(moments[0],origin='lower',levels=[smoothed_rms*5,smoothed_rms*8],colors='orange')
    # plt.show()

def creates_moment_nine_map(data,velocity,rms):
    moments = bm.collapse_ninth(velax=velocity, data=data, rms=rms)
    return moments

    # plt.imshow(moments[0],origin='lower',cmap='jet')
    # plt.show()

def rms_calculation(spectrum):

    upper_idx= find_nearest_index(array=velax, value=-100)
    lower_idx= find_nearest_index(array=velax,value=-50)

    rms_1 = np.nanstd(spectrum[lower_idx:upper_idx])
    plt.plot(velax[lower_idx:upper_idx],spectrum[lower_idx:upper_idx])

    upper_idx= find_nearest_index(array=velax, value=50)
    lower_idx= find_nearest_index(array=velax,value=100)

    rms_2 = np.nanstd(spectrum[lower_idx:upper_idx])
    plt.plot(velax[lower_idx:upper_idx],spectrum[lower_idx:upper_idx])

    return rms_1,rms_2,np.nanmedian([rms_1,rms_2])


def aperture_definition(data,radius,center_position):
    """
    I should  modify this function to work in arcseconds.
    :param radius: radius in pixels
    :param center_position: center position in pixels
    :return: provide the positions/indices of the all the pixels that will be summed
    """

    (x0,y0) = center_position
    # image_file = DataAnalysis(path, filename, continuum=False, integrated=False)

    print('data shape')
    print(np.shape(data))

    try:
        nx_size = np.shape(data)[1]
        ny_size = np.shape(data)[2]

        aperture_mask = np.ones_like(data[0, :, :])

        print("This aperture is performed on a 3-D array ")

    except:

        nx_size = np.shape(data)[0]
        ny_size = np.shape(data)[1]

        aperture_mask = np.ones_like(data[:, :])

        print("This aperture is performed on a 2-D array ")

    for ii in range(nx_size):
        for jj in range(ny_size):
            if (x0-ii)**2 + (y0-jj)**2 <= radius**2:
                aperture_mask[jj,ii]=0

    # plt.imshow(aperture_mask,origin='lower')
    # plt.show()
    return aperture_mask

def integrated_intensity(path,filename,min_vel,max_vel,source_position,aperture_integration):
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

    image_file, rms_integrated_intensity, moment_map = \
        get_moment_map(path, filename, min_vel, max_vel, source_position, moment_number=0)

    ra_array = (image_file.ra_array - np.nanmedian(image_file.ra_array)) * 3600
    dec_array = (image_file.dec_array - np.nanmedian(image_file.dec_array)) * 3600
    pix_size = abs(image_file.cdelt_ra) * 3600
    pix_per_beam = aperture_radius**2*np.pi / (4*np.log(2)*pix_size**2) # pix-per-beam = beam_size/pix_area


    fig = plt.figure(figsize=(8, 7))
    f = fig.add_subplot(1,1,1)

    # plt.imshow(moment_map)
    # plt.show()
    # find the pixel position in the image close to your source coordinates

    source_position_fk5 = source_position.transform_to('fk5')
    ra_degree,dec_degree = source_position_fk5.ra.degree,source_position_fk5.dec.degree

    ra_pix_pos = find_nearest_index(array=image_file.ra_array, value=ra_degree)
    dec_pix_pos = find_nearest_index(array=image_file.dec_array, value=dec_degree)

    radius_size_in_pixels = round(aperture_integration / pix_size / 2., 0)  ## /2 is because we need radius not diameter
    beam_aperture_in_pixels =round(15 / pix_size / 2., 0)

    mask_2d_small = aperture_definition(moment_map, radius=int(beam_aperture_in_pixels),
                                  center_position=(ra_pix_pos, dec_pix_pos))

    masked_small_intensity_flux = ma.masked_array(moment_map, mask=mask_2d_small)

    max_value_in_beam = np.nanmax(masked_small_intensity_flux)

    mask_2d = aperture_definition(moment_map, radius=int(radius_size_in_pixels),
                                  center_position=(ra_pix_pos, dec_pix_pos))


    masked_intensity_flux = ma.masked_array(moment_map, mask=mask_2d)

    # max_value_in_beam = np.nanmax(masked_intensity_flux)

    masked_intensity =  masked_intensity_flux / pix_per_beam

    integrated_value = np.nansum(np.nansum(masked_intensity, axis=1), axis=0)

    print("Integrated intensity in "+str(aperture_integration)+" arcseconds")
    print(integrated_value)

    print("max value in beam")
    print(max_value_in_beam)

    pax = f.imshow(masked_intensity, origin='lower', cmap='bone',
                   extent=(ra_array[0], ra_array[-1], dec_array[0], dec_array[-1])
                   )
    plt.show()

    return max_value_in_beam, integrated_value

def get_moment_map(path,filename,min_vel,max_vel,source_position, moment_number):
    '''
    Creates either moment zero or eight map
    :param path:
    :param filename:
    :param min_vel:
    :param max_vel:
    :param source_position:
    :param moment_number:
    :return:
    '''


    image_file = DataAnalysis(path, filename, continuum=False, integrated=False)

    data, velax = bm.load_cube(os.path.join(path,filename))
    pix_size = abs(image_file.cdelt_ra) * 3600
    pix_per_beam = aperture_radius**2*np.pi / (4*np.log(2)*pix_size**2) # pix-per-beam = beam_size/pix_area
    data = data * pix_per_beam

    spectrum, velax, sigma, position = create_spectrum(path, filename, min_vel,max_vel, source_position, aperture_radius=7.5)

    # sigma, position = spectrum_properties


    smoothing_factor = 8
    smoothed_data = bm.smooth_data(data=data, smooth=smoothing_factor, polyorder=0)
    # smoothed_rms = bm.estimate_RMS(data=smoothed_data[1000:-1000,20:44,20:44], N=1000)

    smoothed_rms, SNR = calculate_signal_to_noise(path, filename,min_vel,max_vel)

    print('The smoothed rms per pixel is',smoothed_rms)

    upper_idx= find_nearest_index(array=velax, value=position-6*sigma) # 6 sigma is for consistency with Carney
    lower_idx= find_nearest_index(array=velax,value=position+6*sigma)

    number_of_channels_integral=abs(upper_idx-lower_idx)

    print('The number of channels is ', abs(number_of_channels_integral))

    shortened_vel=velax[lower_idx:upper_idx]

    user_mask = bm.get_user_mask(data=smoothed_data, user_mask_path=None)


    channel_mask = bm.get_channel_mask(data=smoothed_data,
                                       firstchannel=lower_idx,
                                       lastchannel=upper_idx)

    threshold_mask = bm.get_threshold_mask(data=smoothed_data,
                                           clip=(-np.inf,1.0),
                                           smooth_threshold_mask=smoothing_factor)

    mask = bm.get_combined_mask(user_mask=user_mask,
                                threshold_mask=threshold_mask,
                                channel_mask=channel_mask,
                                combine='and')

    masked_data = smoothed_data * mask

    # wcs_cord = image_file.wcs

    velocity_resolution_km = abs(image_file.cdelt_velo)*smoothing_factor

    print('velocity resolution in km ', velocity_resolution_km)

    rms_integrated_intensity = smoothed_rms*velocity_resolution_km*(number_of_channels_integral)**0.5

    if moment_number ==0:
        moment = creates_moment_zero_map(masked_data,shortened_vel,smoothed_rms)
        return image_file,rms_integrated_intensity,moment[0]

    elif moment_number ==8:
        moment = creates_moment_eighth_map(masked_data,shortened_vel,smoothed_rms)
        return image_file,rms_integrated_intensity,moment[0]


def plot_moment_map(path,filename,min_vel,max_vel,source_position, moment_number,save=False):
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

    image_file, rms_integrated_intensity, moment_map = \
        get_moment_map(path, filename, min_vel, max_vel, source_position, moment_number)

    ra_array = (image_file.ra_array - np.nanmedian(image_file.ra_array)) * 3600
    dec_array = (image_file.dec_array - np.nanmedian(image_file.dec_array)) * 3600

    fig = plt.figure(figsize=(8, 7))
    f = fig.add_subplot(1,1,1)

    divider = make_axes_locatable(f)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    # cbar.set_label(r'$\int T_{mb} dV \; [\rm K \, km/s] $', size=22)

    if moment_number ==0:

        pax = f.imshow(moment_map, origin='lower', cmap='bone',
                       extent=(ra_array[0], ra_array[-1], dec_array[0], dec_array[-1])
                       # )
                       , vmin=0
                       )
                       # , vmin=rms_integrated_intensity ,
                       # ,vmax=rms_integrated_intensity*SNR*0.5)

        cbar = plt.colorbar(pax, cax=cax, format='%.1f', extend='neither')
        cbar.set_label('Integrated intensity '+r'$(\rm K \, km \,s^{-1}) $', size=22)

    elif moment_number ==8:


        pax = f.imshow(moment_map,origin='lower',cmap='bone',
                   extent=(
                   ra_array[0], ra_array[-1], dec_array[0], dec_array[-1])
                       # )
                   # , vmin=smoothed_rms,vmax=smoothed_rms*5)
                , vmin=0
                       )
                       # ,vmax=smoothed_rms*SNR)

        cbar = plt.colorbar(pax, cax=cax, format='%.1f', extend='neither')
        cbar.set_label('Peak intensity ' + r'$(\rm K) $', size=22)


    cbar.ax.tick_params(labelsize=18)

    elips_cont = Ellipse(xy=(-40, 40), width=14, height=14,
                    angle=90, facecolor='gray', edgecolor='k',lw=2.0,alpha=0.6)

    f.add_artist(elips_cont)


    f.set_xlabel(r'$\rm \Delta \,RA \:(arcsec)$', fontsize=18)
    f.set_ylabel(r'$\rm \Delta \,DEC \:(arcsec)$', fontsize=18)
    f.tick_params(axis='both', which='both', labelsize=18)

    if save == True:
        if moment_number==0:
            plt.savefig(os.path.join(save_folder,filename.strip('.fits')+'_mom1.png'), bbox_inches='tight',transparent=True)
        elif moment_number==8:
            plt.savefig(os.path.join(save_folder,filename.strip('.fits')+'_mom8.png'), bbox_inches='tight',transparent=True)

    plt.show()

def fitting_2D_gaussian(path, filename, min_vel, max_vel, source_position):
    '''

    :return:
    '''

    image_file, rms_integrated_intensity, moment_map = \
        get_moment_map(path, filename, min_vel, max_vel, source_position, moment_number=0)

    source_position_fk5 = source_position.transform_to('fk5')
    ra_degree,dec_degree = source_position_fk5.ra.degree, source_position_fk5.dec.degree

    ra_pix_pos = find_nearest_index(array=image_file.ra_array, value=ra_degree)
    dec_pix_pos = find_nearest_index(array=image_file.dec_array, value=dec_degree)

    pix_size = abs(image_file.cdelt_ra) * 3600 # from degrees to arcsec


    def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        """
        2D gaussian
        :param xy:
        :param amplitude:
        :param xo:
        :param yo:
        :param sigma_x:
        :param sigma_y:
        :param theta:
        :param offset:
        :return:
        """
        x, y = xy
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                           + c * ((y - yo) ** 2)))
        return g.ravel()

    # use your favorite image processing library to load an image
    # im = cv2.imread(r"path_to_load\im.png", -1)
    h, w = moment_map.shape
    data = moment_map.ravel()

    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    x, y = np.meshgrid(x, y)

    # initial guess of parameters
    initial_guess = (1, ra_pix_pos, dec_pix_pos, 20, 20, 0, 0.1)

    # find the optimal Gaussian parameters
    try:
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess)

    except:
        return source_position, 15, 15

    best_amplitude, best_xo, best_yo, best_sigma_x, best_sigma_y, best_theta, best_offset = popt

    FWHM_x_arcsec = 2.35482 * best_sigma_x * pix_size
    FWHM_y_arcsec = 2.35482 * best_sigma_y * pix_size

    print('FWHM of Gaussian X and Y in arcsecs')
    print(FWHM_x_arcsec,FWHM_y_arcsec)

    if abs(FWHM_x_arcsec)>50 or abs(FWHM_y_arcsec)>50:
        print('No Gaussian can be fitted')
        return source_position, 15, 15
    ### Transform the new best gaussian position to a SkyCoord Objects

    new_ra, new_dec = image_file.ra_array[int(best_xo)], image_file.dec_array[int(best_yo)]

    coo = ICRS(new_ra * u.degree, new_dec * u.degree)
    new_coord = str(coo.ra.to_string(u.hour)) + ' ' + str(coo.dec)
    print('best gaussian coordinates')
    print(new_coord)
    new_source_position = SkyCoord(str(coo.ra.to_string(u.hour)), str(coo.dec), frame='icrs') ## GV Tau

    # create new data with these parameters
    data_fitted = twoD_Gaussian((x, y), *popt)

    fig, ax = plt.subplots(1, 1)
    # ax.hold(True) For older versions. This has now been deprecated and later removed
    ax.imshow(moment_map, cmap=plt.cm.jet, origin='lower',
              extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(x, y, data_fitted.reshape(h, w), 8, colors='w')
    plt.show()


    return new_source_position, FWHM_x_arcsec, FWHM_y_arcsec
    # cv2.imwrite(r"path_to_save\data_fitted.png", data_fitted.reshape(h, w))


def concentration_factor(Beam,Robs,integrated_intensity,peak_intensity):
    '''
    Calculate concentration factor as define by Carney et al. 2016
    :param Beam:
    :param Robs:
    :param integrated_intensity:
    :param peak_intensity:
    :return:
    '''
    return 1- 1.13 *Beam**2 * integrated_intensity/(np.pi*Robs**2*peak_intensity)

def calculate_signal_to_noise(path,filename,min_vel,max_vel):

    data, velax = bm.load_cube(os.path.join(path,filename))


    # avgResult = np.average(data.reshape(-1, 3), axis=1)
    # velax = np.average(velax.reshape(-1, 3), axis=1)

    # data = data*72.5177622692351 # pix-per-beam = beam_size/pix_area = 15*15*(pi/4ln2)/(1.875*1.875)
    data = data * 18.129440567  # pix-per-beam = beam_size/pix_area = 7.5*7.5*(pi/4ln2)/(1.875*1.875)
    # data = data * 50.26548245743669

    broad_lower_idx= find_nearest_index(array=velax,value=max_vel)
    broad_upper_idx= find_nearest_index(array=velax, value=min_vel)

    rms = bm.estimate_RMS(data=data[1500:-1500,20:44,20:44], N=1000)
    max_value_data=np.nanmax(data[broad_lower_idx:broad_upper_idx,20:44,20:44])

    print('non-smoothed rms ' + str(rms))
    print('non-smoothed peak ' + str(max_value_data))
    print('non-smoothed S/N ' + str(max_value_data/rms))

    smoothed_data = bm.smooth_data(data=data, smooth=4, polyorder=0)
    smoothed_rms = bm.estimate_RMS(data=smoothed_data[1500:-1500,20:44,20:44], N=1000)
    max_value_smoothed_data=np.nanmax(smoothed_data[broad_lower_idx:broad_upper_idx,20:44,20:44])

    print('smoothed rms ' + str(smoothed_rms))
    print('smoothed peak ' + str(max_value_smoothed_data))
    print('smoothed S/N ' + str(max_value_smoothed_data/smoothed_rms))

    return smoothed_rms, max_value_smoothed_data/smoothed_rms

if __name__ == "__main__":
    save_folder='moment_images'

    path1 = '/Users/christianflores/Documents/work/Astronomy_data/JCMT/M22BH10B/IRAS04591-0856/C18O/'
    filename1='IRAS04591-0856_C18O.fits'

    path2 = '/Users/christianflores/Documents/work/Astronomy_data/JCMT/M22BH10B/IRAS04591-0856/HCO+/'
    filename2='IRAS04591-0856_HCO+.fits'

    # path2 = '/Users/christianflores/Documents/work/Astronomy_data/JCMT/Archival/GV_Tau/HCO+/'
    # filename2='GV_Tau_HCO+_Tmb_resampled_100.fits'

    min_vel,max_vel = 0,40
    aperture_radius=7.5

    # source_position = SkyCoord('04h29m23.7s', '+24d33m02.0s', frame='icrs') ## GV Tau
    # source_position = SkyCoord('04h29m23.7s', '+24d33m02.0s', frame='icrs') ## GV Tau
    # source_position = SkyCoord('04h56m57.0s', '+51d30m50.8s', frame='icrs') ##V347 Aur
    # source_position = SkyCoord('04h21m10.0s', '+27d01m42.0s', frame='icrs') ## IRAS04181+2655
    # source_position = SkyCoord('03h33m12.8s', '+31d21m24.0s', frame='icrs') ## IRAS03301+3111
    # source_position = SkyCoord('03h29m10.0s','+31d21m59.0s', frame='icrs') ## IRAS03260+3111
    # source_position = SkyCoord('19h01m48.1s', '-36d57m22.0s', frame='icrs') ## IRAS5
    # source_position = SkyCoord('18h36m46.3s', '-01d10m29.0s', frame='icrs') ## IRAS18341-0113
    # source_position = SkyCoord('05h28m49.9s','+30d51m29.0s', frame='icrs') ## IRAS05256+3049
    # source_position = SkyCoord('04h39m55.8s', '+25d45m02.0s', frame='icrs') ## IRAS04369+2539
    # source_position = SkyCoord('04h14m26.4s', '+28d05m59.0s', frame='icrs') ## IRAS04113+2758S
    # source_position = SkyCoord('04h52m06.7s', '+30d47m17.0s', frame='icrs') ## IRAS04489+3042
    source_position = SkyCoord('05h01m29.6s','-08d52m16.8s',  frame='icrs') ## IRAS04591-0856

    # calculate_signal_to_noise(path1, filename1,min_vel,max_vel)
    # calculate_signal_to_noise(path2, filename2,min_vel,max_vel)

    # plot_spectra([path1,filename1],[path2, filename2],source_position=source_position,velocity_min=min_vel,
    #              velocity_max=max_vel, aperture_radius= aperture_radius, save=True)

    # plot_spectra([path2, filename2],source_position=source_position,velocity_min=min_vel,
    #              velocity_max=max_vel, aperture_radius= aperture_radius, save=True)

    plot_moment_map(path2,filename2, min_vel=min_vel,max_vel=max_vel, moment_number=0,
                    source_position=source_position,save=True)

    plot_moment_map(path1,filename1, min_vel=min_vel,max_vel=max_vel, moment_number=0,
                    source_position=source_position,save=True)

    # new_source_position, ra_FWHM, dec_FWHM = fitting_2D_gaussian(path2,filename2,min_vel,max_vel,source_position)
    #
    # abs_max_FWHM=max(abs(ra_FWHM),abs(dec_FWHM))
    #
    # print('FWHM values of gaussian')
    #
    # print(ra_FWHM,dec_FWHM,abs_max_FWHM)
    #
    # max_value, integrated = integrated_intensity(path2,filename2,min_vel,max_vel,source_position,
    #                                              aperture_integration=abs_max_FWHM)
    #
    #
    # calculated_concentration_factor=concentration_factor(Beam=15,Robs=abs_max_FWHM,
    #                                                      integrated_intensity=integrated,
    #                                                      peak_intensity=max_value)
    #
    # print('concentrated')
    # print(calculated_concentration_factor)