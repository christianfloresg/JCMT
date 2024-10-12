import numpy as np
from Process_JCMT_data import DataAnalysis
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.visualization.wcsaxes import SphericalCircle
from data_cube_analysis import fit_gaussian_to_spectrum, write_or_update_values\
    , calculate_peak_SNR, integrate_flux_over_velocity, fit_gaussian_2d


def find_simbad_source_in_file(file_name, search_word):
    """
    Reads a text file, ignoring lines starting with '#', and searches for a word in the first column.
    If found, returns the rest of the line. If not found, prints a message and exits.

    :param file_name: The name of the text file
    :param search_word: The word to search for in the first column of the file
    :return: The rest of the line if the word is found, otherwise prints a message
    """
    try:
        with open(file_name, 'r') as file:
            for line in file:
                # Skip lines starting with '#' or that are empty
                if line.startswith("#") or not line.strip():
                    continue

                # Split the line into first word and the rest, without unnecessary stripping
                parts = line.split(maxsplit=1)

                if parts[0] == search_word:
                    # Return the rest of the line as soon as the word is found
                    print("You are using the coordinates of " + parts[1])
                    return parts[1] if len(parts) > 1 else ''

        # If no match is found
        print("This source is not in the list.")

    except FileNotFoundError:
        print(f"Error: The file {file_name} does not exist.")

    return None


def find_word_in_file(file_name, search_word, position):
    """
    Reads a text file, ignoring lines starting with '#', and searches for a word in the first column.
    If found, returns the word at the requested position in the same line. If not found, prints a message and exits.

    :param file_name: The name of the text file
    :param search_word: The word to search for in the first column of the file
    :param position: The index of the word in the line to return (0-based)
    :return: The word at the specified position if the word is found, otherwise prints a message
    """
    try:
        with open(file_name, 'r') as file:
            for line in file:
                # Skip lines starting with '#' or that are empty
                if line.startswith("#") or not line.strip():
                    continue

                # Split the line into words
                parts = line.split()

                if parts[0] == search_word:
                    # Check if the requested position is within bounds
                    if position < len(parts):
                        return parts[position]
                    else:
                        print(f"Error: The requested position {position} is out of bounds.")
                        return None

        # If no match is found
        print("This source is not in the list.")

    except FileNotFoundError:
        print(f"Error: The file {file_name} does not exist.")

    return None


def closest_idx(lst, val):
    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argmin()
    return idx



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
    simbad_name = find_simbad_source_in_file(file_name='names_to_simbad_names.txt', search_word=source_name)
    skycoord_object = get_icrs_coordinates(simbad_name)

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


def retrieve_and_write_spectral_properties(source_name, molecule):
    """
    This is a preparatory step where a gaussian is fitted to the spectrum
    so the central wavelength position, the FWHM and other parameters are known
    and then wrote to text
    :param source_name:
    :param molecule:
    :return:
    """

    fits_file_name = source_name+'_'+molecule #'V347_Aur_HCO+'

    #### Get noise & peak SNR from the cube
    peak_signal_in_cube, average_noise_images = calculate_peak_SNR(fits_file_name,source_name=source_name,
                                                                   velo_limits=[5, 10], separate=True)
    noise_level = round(average_noise_images,4)
    peak_SNR = round(peak_signal_in_cube/average_noise_images,1)

    #### Obtain spectral properties from fitting a gaussian
    spectrum, velocity = make_central_spectrum_data(source_name, molecule)
    Tmb = round(np.nanmax(spectrum),3)
    pos, FHWM, sigma = fit_gaussian_to_spectrum(spectrum, velocity,velo_range=[-20,30])
    rounded_pos, rounded_FHWM, rounded_sigma = round(pos,3), round(abs(FHWM),3), round(abs(sigma),3)


    ### Get the integrated intensity from the spectrum
    vmin = pos - 5*abs(sigma)
    vmax = pos + 5*abs(sigma)
    integrated_intensity_main_beam = integrate_flux_over_velocity(velocities=velocity, flux=spectrum,
                                                                  v_min=vmin, v_max=vmax)

    plot=False
    if plot:
        print('the integrated intensity is ', integrated_intensity_main_beam)
        plt.plot(velocity,spectrum)
        plt.axvline(vmin)
        plt.axvline(vmax)
        plt.show()


    values_to_text =[source_name,noise_level,Tmb, peak_SNR, rounded_pos, rounded_FHWM,
                     rounded_sigma, integrated_intensity_main_beam, molecule]
    write_or_update_values(file_name='spectrum_parameters_'+molecule+'.txt', new_values=values_to_text)


def plot_spectrum(source_name, molecule, type='central', save=False):
    """
    This one plots the average spectrum
    """
    if type.lower()=='fov':
        spectrum, velocity = make_averaged_spectrum_data(source_name,molecule)
        title = 'FOV averaged spectrum \n' + source_name + ' for ' + molecule
    else:
        spectrum, velocity = make_central_spectrum_data(source_name, molecule)
        title = 'Central beam  spectrum \n' + source_name + ' for ' + molecule

    try:
        ### I first try to get the velocity centroid from the data saved in text file.
        pos = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name, position=4)
        pos = float(pos)
    except:
        pos, FHWM, sigma = fit_gaussian_to_spectrum(spectrum, velocity,velo_range=[-20,30])

    plt.figure()
    # plt.title("Averaged Spectrum ("+mole_name+") @"+dir_each)

    plt.xlabel("velocity (km/s)",size=14)
    plt.ylabel("Averaged Spectrum (K)",size=14)
    plt.title(title)
    plt.plot(velocity,spectrum,"-",color="black",lw=1)

    plt.tick_params(axis='both', direction='in')
    plt.xlim(pos - 10, pos+ 10)
    if save:
        plt.savefig(os.path.join('Figures', 'spectrum_'+filename), bbox_inches='tight', dpi=300)

    plt.show()


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
    simbad_name = find_simbad_source_in_file(file_name='names_to_simbad_names.txt', search_word=source_name)
    skycoord_object = get_icrs_coordinates(simbad_name)


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
    levels = np.array([0.2, 0.5, 0.8, 0.95])
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
        plt.savefig(os.path.join('Figures',filename), bbox_inches='tight',dpi=300)
        # plt.savefig(os.path.join('Figures',filename+'_transparent'), bbox_inches='tight', transparent=True)
    plt.show()

    fit_gaussian_2d(image_mom_0,moment_0.wcs)

if __name__ == "__main__":

    source_name = 'IRAS04181+2655S'
    molecule ='HCO+'
    fits_file_name=source_name+'_'+molecule #'V347_Aur_HCO+'
    ### Should slightly modify this so it tries with the given source name OR the name given here.
    ### Even better I should have a list of the SIMBAD name for each of the sources, so I don't need to
    ### input this one manually every time!

    retrieve_and_write_spectral_properties(source_name, molecule)

    ### Step 1 creates a plot of the spectrum
    ### this also estimate the velocity of this component.
    ### you can save parameters to text file. These can be used
    ### later to restrict the velocity range of the moment maps.

    plot_spectrum(source_name, molecule,type='central')



    ### Step 3
    ### Plot the maps
    # skycoord_object = SkyCoord('04 56 57.0 +51 30 50.88', unit=(u.hourangle, u.deg))
    # skycoord_object = get_icrs_coordinates('V347 Aur')
    # plot_moment_zero_map(fits_file_name,source_name=source_name,save=False,sky_cord_object=True)

    ### Step 4
    ### Get an averaged spectrum over 1 beam of the map. You MUST give the SIMBAD-searchable name
    # plot_spectrum(source_name, molecule)

