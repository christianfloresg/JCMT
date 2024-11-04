import numpy as np
from scipy.optimize import curve_fit
from Process_JCMT_data import DataAnalysis
import os
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.visualization import astropy_mpl_style


def calculate_peak_SNR(filename,source_name, velo_limits=[2, 10], separate=False):
    '''
    Calculates the peak SNR over the whole cube.
    It is possible to set velocity limits for the calculation
    of noise in line-free regions.
    The noise is calculated as the std of images in line-free channels,
    averaged over many channels.
    '''


    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')
    velocity = data_cube.vel
    velocity_length = data_cube.nz

    print('molecule ',data_cube.molecule)
    ### Here I can go from sky position to pixel coordinates


    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05

    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635

    else:
        raise Exception("Sorry, I need to calculate such aperture radius")


    pix_per_beam = aperture_radius**2*np.pi / (4*np.log(2)*data_cube.cdelt_ra**2) # pix-per-beam = beam_size/pix_area

    image = data_cube.ppv_data*pix_per_beam

    val_down, val_up = velo_limits[0], velo_limits[1]
    # lower_idx, upper_idx = closest_idx(velocity, val_down), closest_idx(velocity, val_up)
    lower_idx, upper_idx = find_nearest_index(velocity, val_down), find_nearest_index(velocity, val_up)

    try:
        peak_signal_in_cube = np.nanmax(image[lower_idx:upper_idx,:,:])
    except:
        peak_signal_in_cube = np.nanmax(image[upper_idx:lower_idx,:,:])


    ### define the channels to calculate the noise to be 20% of the band on each side
    n_channels_noise = int(velocity_length*0.20)
    array_of_noise_lower = np.nanstd(image[:n_channels_noise, :, :], axis=0)
    array_of_noise_upper = np.nanstd(image[(velocity_length-n_channels_noise):, :, :], axis=0)

    average_noise_images = (np.nanmean(array_of_noise_lower) + np.nanmean(array_of_noise_upper)) / 2.
    print('Image average noise level: ',average_noise_images)

    if separate:
        return peak_signal_in_cube , average_noise_images
    return round(peak_signal_in_cube / average_noise_images, 1)


def integrate_flux_over_velocity(velocities, flux, v_min, v_max):
    """
    Integrates the flux over a specified velocity range using the trapezoidal rule.

    :param velocities: Array of velocities (same length as flux array)
    :param flux: Array of flux values
    :param v_min: Minimum velocity of the range
    :param v_max: Maximum velocity of the range
    :return: The integrated flux over the specified velocity range
    """

    # Find indices corresponding to the velocity range
    indices_in_range = np.where((velocities >= v_min) & (velocities <= v_max))[0]

    # Extract the corresponding velocities and flux values
    velocities_in_range = velocities[indices_in_range]
    flux_in_range = flux[indices_in_range]

    # Perform numerical integration using the trapezoidal rule
    integrated_flux = np.trapz(flux_in_range, velocities_in_range)

    return integrated_flux



def write_or_update_values(file_name, new_values):
    """
    Writes four values to a text file. If the first value already exists in the file,
    it updates the entry; otherwise, it appends the new values.
    If the file does not exist, it adds "## this text" to the first line.

    :param file_name: The name of the text file
    :param new_values: A list of four values to be written to the file
    """

    # values_to_text = [
    #     source_name, image_noise_level, peak_SNR, line_noise, Tmb, line_SNR , rounded_vel_pos, rounded_FHWM,
    #     rounded_sigma, integrated_intensity_main_beam, molecule
    # ]

    # Format the values with specific byte alignment
    formatted_entry = (
        f"{new_values[0]:<17}"  # Source name in 17 bytes
        f"{'':<8}"  # 8 bytes of white space
        f"{new_values[1]:<15.4f}"  # Image noise level with 4 decimal places in 10 bytes
        f"{new_values[2]:<15.1f}"  # Peak image SNR with 1 decimal place in 10 bytess
        f"{new_values[3]:<15.4f}"  # Line noise level with 4 decimal places in 10 bytes
        f"{new_values[4]:<15.3f}"  # Tmb with 3 decimal places in 10 bytes
        f"{new_values[5]:<15.1f}"  # Peak line SNR with 1 decimal place in 10 bytes
        f"{new_values[6]:<15.3f}"  # Position with 3 decimal places in 10 bytes
        f"{new_values[7]:<15.3f}"  # FWHM with 3 decimal places in 10 bytes
        f"{new_values[8]:<15.3f}"  # Sigma with 3 decimal places in 10 bytes
        f"{new_values[9]:<15.3f}"  # Integrated intensity with 3 decimal places in 10 bytes
        f"{new_values[10]:<15}"  # Molecule in 10 bytes
    )

    # Read the file if it exists, otherwise start with a header
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        # If the file doesn't exist, start with a formatted header line
        header = (
            f"{'## Source name':<17}"
            f"{'':<8}"  # 8 bytes of white space
            f"{'Image Noise':<15}"
            f"{'Peak Im. SNR':<15}"
            f"{'Line Noise':<15}"
            f"{'Tmb ':<15}"
            f"{'Peak Line SNR':<15}"
            f"{'Velocity':<15}"
            f"{'FWHM ':<15}"
            f"{'Sigma':<15}"
            f"{'Integ. Int .':<15}"
            f"{'Molecule':<15}\n"
        )

        header2 = (
            f"{'## ':<17}"
            f"{'':<8}"  # 8 bytes of white space
            f"{'(K)':<15}"
            f"{' ':<15}"
            f"{'(K)':<15}"
            f"{'(K)':<15}"
            f"{' ':<15}"
            f"{'(km/s)':<15}"
            f"{'(km/s)':<15}"
            f"{'(km/s)':<15}"
            f"{'(K * km/s)':<15}"
            f"{' ':<15}\n"
        )
        lines = [header,header2]

    first_value = new_values[0]
    found = False

    # Check if the first value is already in the file and update the line if it exists
    for i, line in enumerate(lines):
        # Check if this line starts with the source name
        if line.startswith(f"{first_value:<17}"):
            lines[i] = formatted_entry + "\n"
            found = True
            break

    # If the source name was not found, append the new formatted entry
    if not found:
        lines.append(formatted_entry + "\n")

    # Write the updated content back to the file
    with open(file_name, 'w') as file:
        file.writelines(lines)


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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

    if broad_upper_idx<broad_lower_idx:
        shortened_vel = velax[broad_upper_idx:broad_lower_idx]
        shortened_flux = spectrum[broad_upper_idx:broad_lower_idx]




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
                plt.savefig(os.path.join('fitting_line', filename1.strip('.fits') + '_spectrum_fit.png'),
                            bbox_inches='tight',
                            transparent=False)
            except:
                plt.savefig(os.path.join('fitting_line', filename2.strip('.fits') + '_spectrum_fit.png'),
                            bbox_inches='tight',
                            transparent=False)
        plt.show()

    return  x0, FWHM, sigma

def fit_gaussian_to_spectrum(spectrum,velocity,velo_range=[-30,30],plot=True):

    velocity_min,velocity_max= velo_range[0],velo_range[1]
    position, FWHM, sigma = spectrum_properties(spectrum, velocity, velocity_min=velocity_min,
                                          velocity_max=velocity_max, nsigma=6, plot=plot)

    return position, FWHM, sigma

# Define a 2D Gaussian function
def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    '''
    I am actually forcing a circular gaussian by setting sigma_x = sigma_y
    :param xy:
    :param amplitude:
    :param x0:
    :param y0:
    :param sigma_x:
    :param sigma_y:
    :param theta:
    :param offset:
    :return:
    '''
    (x, y) = xy
    x0 = float(x0)
    y0 = float(y0)
    sigma_y =sigma_x ### we are forcing the circular gaussian
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(- (a * ((x - x0)**2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0)**2)))
    return g.ravel()

# Function to fit the Gaussian and calculate FWHM
def fit_gaussian_2d(image, wcs):

    # image_max =np.nanmax(image)
    # threshold = 0.16096##image_max*0.1
    # image[image < threshold] = np.nan

    # Create x and y indices
    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    x, y = np.meshgrid(x, y)

    # Flatten the image, x, and y, but keep the valid (non-NaN) points only
    mask = ~np.isnan(image)  # True where the image is not NaN
    x_valid = x[mask]
    y_valid = y[mask]
    image_valid = image[mask]
    image_max = np.nanmax(image)

    # Initial guess for the parameters: amplitude, x0, y0, sigma_x, sigma_y, theta, offset
    initial_guess = (image_max, image.shape[1] // 2, image.shape[0] // 2, 1, 1, 0, np.nanmin(image))

    print(image.shape[1] // 2,image.shape[0] // 2 )
    # Fit the Gaussian to the valid data points
    # bounds= ([-np.inf, 40,40,-np.inf,-np.inf,-np.inf,-np.inf],
    #          [np.inf,50,50,np.inf,np.inf,np.inf,np.inf])

    popt, _ = curve_fit(gaussian_2d, (x_valid, y_valid), image_valid.ravel(), p0=initial_guess)#,bounds=bounds)

    # Extract the fit parameters
    # amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt
    amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt
    sigma_y = sigma_x


    # Convert x0, y0 (center) from pixel to WCS (RA, Dec)
    sky_coords = wcs.pixel_to_world(x0, y0)
    print(f"Center in WCS (RA, Dec): {sky_coords}")

    # Calculate pixel scale in arcseconds
    # Get the pixel scale from the WCS. This is approximate and assumes square pixels.
    pixel_scale = np.mean(np.abs(wcs.pixel_scale_matrix)) * 3600  # in arcseconds per pixel
    print(f"Pixel scale: {pixel_scale} arcsec/pixel")


    # Convert x0, y0 from pixels to arcseconds
    x0_arcsec = (x0-image.shape[1] // 2) * pixel_scale
    y0_arcsec = (y0-image.shape[0] // 2) * pixel_scale

    print(f"Pos_offset_x (arcsec): {x0_arcsec}")
    print(f"Pos_offset_y (arcsec): {y0_arcsec}")

    # Convert sigma_x, sigma_y from pixels to arcseconds
    sigma_x_arcsec = sigma_x * pixel_scale
    sigma_y_arcsec = sigma_y * pixel_scale
    print(f"Sigma_x (arcsec): {sigma_x_arcsec}")
    print(f"Sigma_y (arcsec): {sigma_y_arcsec}")

    # Calculate FWHM for both x and y directions in arcseconds
    fwhm_x_arcsec = 2.355 * sigma_x_arcsec
    fwhm_y_arcsec = 2.355 * sigma_y_arcsec

    print(f"FWHM in x (arcsec): {fwhm_x_arcsec}")
    print(f"FWHM in y (arcsec): {fwhm_y_arcsec}")

    plot=True
    if plot:
        fig = plt.figure()
        ax = plt.subplot(projection=wcs)

        ax.imshow(image, origin='lower', cmap='gray')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')

        # Generate RA/Dec coordinates from pixel coordinates for the contour
        world_coords = wcs.pixel_to_world(x, y)

        # Convert to arrays of RA and Dec in degrees
        ra = world_coords.ra.deg
        dec = world_coords.dec.deg

        # Plot the contours, matching the WCS coordinate system
        c1 = ax.contour(ra, dec, gaussian_2d((x, y), *popt).reshape(image.shape),
                   transform=ax.get_transform('world'), linewidths=1,levels=[0.3*amplitude,
                                                                             0.5*amplitude,
                                                                             0.8*amplitude])
        # c1.clabel(c1, inline=True)

        plt.show()

    return x0_arcsec,y0_arcsec,sigma_x_arcsec,fwhm_x_arcsec  # Return the FWHM values and the fit parameters


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

if __name__ == "__main__":

    # Create a 2D Gaussian image for testing
    x = np.linspace(0, 50, 50)
    y = np.linspace(0, 50, 50)
    x, y = np.meshgrid(x, y)
    test_image = gaussian_2d((x, y), 3, 25, 25, 5, 7, 0, 10).reshape(50, 50)

    # Fit the Gaussian to the image
    fwhm_x, fwhm_y, params = fit_gaussian_2d(test_image)
    print(f"FWHM in x: {fwhm_x}")
    print(f"FWHM in y: {fwhm_y}")
