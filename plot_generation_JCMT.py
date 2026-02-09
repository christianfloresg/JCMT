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
    , calculate_peak_SNR, integrate_flux_over_velocity, fit_gaussian_2d, find_nearest_index
from matplotlib.patches import Arc
import matplotlib.ticker as tkr
from datetime import date,datetime
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.ndimage import rotate
from astropy.stats import sigma_clip
from astropy.stats import mad_std, sigma_clip

import os
from astropy.modeling import models, fitting
from astropy.nddata import NDData
from astropy.visualization import simple_norm
# from photutils.aperture import CircularAperture, aperture_photometry
from photutils.aperture import CircularAperture, EllipticalAperture, aperture_photometry

# from skimage.transform import rotate

today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)

def create_alternating_circle(ax, skycoord_object, aperture_radius, num_segments=20):
    """
    Create a circular patch with alternating dark gray and white segments.

    :param ax: The matplotlib Axes object
    :param skycoord_object: SkyCoord object representing the center of the circle
    :param aperture_radius: Radius of the circle in arcsec
    :param num_segments: Number of segments to divide the circle into
    """
    # Calculate the angle for each segment
    angle_per_segment = 360 / num_segments

    # Define colors for alternating segments
    colors = ['darkgray', 'white']

    # Loop to create each segment of the circle
    for i in range(num_segments):
        # Set the start angle for each segment
        theta_start = i * angle_per_segment
        theta_end = theta_start + angle_per_segment

        # Alternate the color for each segment
        color = colors[i % 2]

        # Create an arc segment
        arc = Arc((skycoord_object.ra.degree, skycoord_object.dec.degree),
                  width=aperture_radius.to(u.deg).value * 2,
                  height=aperture_radius.to(u.deg).value * 2,
                  theta1=theta_start, theta2=theta_end,
                  edgecolor=color, linewidth=2, linestyle='--',
                  transform=ax.get_transform('fk5'))

        # Add the arc segment to the plot
        # return arc
        ax.add_patch(arc)

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

    header = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                # Skip lines starting with '#' or that are empty
                if line.startswith("#") or not line.strip():
                    header.append(line.split())
                    # print(header)
                    continue

                # Split the line into words
                parts = line.split()

                if parts[0] == search_word:
                    # Check if the requested position is within bounds
                    if position < len(parts):
                        print(f"For source {line.split()[0]}, the value of "
                              f"{header[0][position+1]} is {parts[position]} {header[1][position+1]}")
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
    # moment_0 = DataAnalysis(os.path.join('moment_maps',source_name), filename+'_mom0.fits')


    velocity = data_cube.vel
    cube = data_cube.ppv_data#[:,5:-5,5:-5]

    # plt.imshow(np.nansum(cube,axis=0))
    # plt.show()

    averaged_spectrum = np.nanmean(cube, axis=(1,2))

    return averaged_spectrum, velocity


def make_central_spectrum_data(source_name, molecule,noskycoord=False):
    """
    Average spectrum of the central beam.

    """
    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt', search_word=source_name)
    skycoord_object = get_icrs_coordinates(simbad_name)

    filename = source_name+'_'+molecule
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')


    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05 ## This is in arcsec

    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635 ## This is in arcsec

    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    pixel_scale_ra = data_cube.header['CDELT1'] * 3600  # arcseconds per pixel
    pixel_scale_dec = data_cube.header['CDELT2'] * 3600  # arcseconds per pixel
    aperture_radius_pixels = abs(aperture_radius/pixel_scale_ra)



    velocity = data_cube.vel


    if noskycoord:
        print('I am not using the skycoordinates of the object, instead taking the\\ spectrum from the central pixels')
        #### Only use this if there is a problem with WCS and you need to calculate the spectrum at the center.
        ra_center = data_cube.wcs.celestial.all_pix2world(data_cube.nx / 2, data_cube.ny / 2, 0)[0]
        dec_center = data_cube.wcs.celestial.all_pix2world(data_cube.nx / 2, data_cube.ny / 2, 0)[1]
        skycoord_object = SkyCoord(ra=ra_center, dec=dec_center, unit='deg', frame='icrs')
        print(skycoord_object)

    # x_center, y_center = moment_0.wcs.world_to_pixel(skycoord_object) ## This one if 2D cube
    x_center, y_center = data_cube.wcs.celestial.world_to_pixel(skycoord_object) ## This one if 3D cube


    print(f"These are the sky coordinates of your {source_name}: ", skycoord_object)

    # Initialize a list to store the pixel values within the aperture
    center_beam_values = []

    # Iterate over a square region, but filter by distance to make it circular
    for xx in range(int(x_center - aperture_radius_pixels), int(x_center + aperture_radius_pixels) + 1):
        for yy in range(int(y_center - aperture_radius_pixels), int(y_center + aperture_radius_pixels) + 1):
            # Calculate the distance from the center
            distance = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)

            # Check if the distance is within the aperture radius
            if distance <= aperture_radius_pixels:
                # Append the data at this pixel position
                center_beam_values.append(data_cube.ppv_data[:, yy, xx])

    # Convert center_beam_values to a NumPy array for easy manipulation
    center_beam_values = np.array(center_beam_values)

    average_spectrum = np.nanmean(center_beam_values, axis=0)

    return average_spectrum, velocity


def retrieve_and_write_spectral_properties(source_name, molecule, plot=True,noskycoord=False):
    """
    This is a preparatory step where a gaussian is fitted to the spectrum
    so the central wavelength position, the FWHM and other parameters are known
    and then wrote to text
    :param source_name:
    :param molecule:
    :return:
    """

    fits_file_name = source_name+'_'+molecule #'V347_Aur_HCO+'
    vmin,vmax=0.0,0.0 ## define initial values in case the fit does not work

    # try:
    #### First fit the spectrum retrieved from the whole cube
    spectrum_fov, velocity_fov = make_averaged_spectrum_data(source_name, molecule)
    pos_fov, FHWM_fov, sigma_fov = fit_gaussian_to_spectrum(spectrum_fov, velocity_fov,
                                                            velo_range=[-30,30] ,plot=plot,
                                                            source_name=source_name+'_FOV',molecule=molecule)
    # rounded_vel_pos, rounded_FHWM, rounded_sigma = round(pos,3), round(abs(FHWM),3), round(abs(sigma),3)

    pos_fov=6.0
    vmin_fov = pos_fov - 15*abs(sigma_fov)
    vmax_fov = pos_fov + 15*abs(sigma_fov)


    ### Obtain spectral properties from fitting a gaussian to the central spectra
    spectrum, velocity = make_central_spectrum_data(source_name, molecule,noskycoord=noskycoord)

    LB_idx_noise_low, LB_idx_noise_high = find_nearest_index(velocity, -50), find_nearest_index(velocity, vmin - 10)
    UB_idx_noise_low, UB_idx_noise_high = find_nearest_index(velocity, vmax + 10), find_nearest_index(velocity, 50)

    line_noise = (np.nanstd(spectrum[LB_idx_noise_low:LB_idx_noise_high]) +
                  np.nanstd(spectrum[UB_idx_noise_low:UB_idx_noise_high])) / 2.

    integrated_intensity_fov, uncertainty_int_intensity_fov = integrate_flux_over_velocity(velocities=velocity_fov, flux=spectrum_fov,
                                                                  v_min=pos_fov - 3*abs(sigma_fov),
                                                                  v_max=pos_fov + 3*abs(sigma_fov),
                                                                                       rms_noise=line_noise)
    print('Line noise = ', line_noise)

    try:
        pos, FHWM, sigma = fit_gaussian_to_spectrum(spectrum, velocity,
                                                    velo_range=[vmin_fov,vmax_fov], plot=plot,
                                                    source_name=source_name + '_central', molecule=molecule,
                                                    position_guess=pos_fov, sigma_guess=sigma_fov)
                                                    # position_guess = 15, sigma_guess = 1)

        rounded_vel_pos, rounded_FHWM, rounded_sigma = round(pos,3), round(abs(FHWM),3), round(abs(sigma),3)


        # filename = source_name + '_' + molecule  # 'V347_Aur_HCO+'
        # data_cube = DataAnalysis(os.path.join('sdf_and_fits', source_name), filename + '.fits')

        vmin = pos - 3*abs(sigma)
        vmax = pos + 3*abs(sigma)

        ### Calculate the peak emission of the line,
        idx_line_low,idx_line_high = find_nearest_index(velocity,vmin),find_nearest_index(velocity,vmax)

        Tmb = round(np.nanmax(spectrum[idx_line_low:idx_line_high+1]),4)

        ### Calculate  the noise of the line.
        LB_idx_noise_low,LB_idx_noise_high = find_nearest_index(velocity,-50), find_nearest_index(velocity,vmin-10)
        UB_idx_noise_low,UB_idx_noise_high = find_nearest_index(velocity,vmax+10), find_nearest_index(velocity,50)

        line_noise = (np.nanstd(spectrum[LB_idx_noise_low:LB_idx_noise_high])+
                      np.nanstd(spectrum[UB_idx_noise_low:UB_idx_noise_high]))/2.

        line_SNR = round(Tmb/line_noise,2)
        ### Get the integrated intensity from the spectrum
        integrated_intensity_main_beam, uncertainty_integ_intensity_main_beam = \
            integrate_flux_over_velocity(velocities=velocity, flux=spectrum,
                                         v_min=vmin, v_max=vmax,rms_noise=line_noise)

        print('this is integrated flux and uncertainty',integrated_intensity_main_beam,uncertainty_integ_intensity_main_beam)
        #### Get noise & peak SNR from the cube
        peak_signal_in_cube, average_noise_images = calculate_peak_SNR(fits_file_name,source_name=source_name,
                                                                   velo_limits=[vmin, vmax], separate=True)

        image_noise_level = round(average_noise_images,4)
        peak_SNR = round(peak_signal_in_cube/average_noise_images,1)

    except ValueError as err:
        print(f"Parameters for {source_name} and {molecule} was not produced.")
        print(f"An error occurred: {err}")

        Tmb = 0.0
        uncertainty_integ_intensity_main_beam = 0.0
        integrated_intensity_main_beam = 0.0
        integrated_intensity_fov = 0.0
        rounded_vel_pos, rounded_FHWM, rounded_sigma = 0.0,0.0,0.0
        # LB_idx_noise_low,LB_idx_noise_high = find_nearest_index(velocity,-50), find_nearest_index(velocity,vmin-10)
        # UB_idx_noise_low,UB_idx_noise_high = find_nearest_index(velocity,vmax+10), find_nearest_index(velocity,50)
        #
        # line_noise = (np.nanstd(spectrum[LB_idx_noise_low:LB_idx_noise_high])+
        #               np.nanstd(spectrum[UB_idx_noise_low:UB_idx_noise_high]))/2.
        line_SNR = 0.0

        image_noise_level = 0.0
        peak_SNR = 0.0

    # if plot:
    #     print('the integrated intensity is ', integrated_intensity_main_beam)
    #     plt.plot(velocity,spectrum)
    #     plt.axvline(vmin)
    #     plt.axvline(vmax)
    #     plt.show()

    # Prepare the values for writing/updating
    values_to_text = [
        source_name, image_noise_level, peak_SNR, line_noise, Tmb, line_SNR , rounded_vel_pos, rounded_FHWM,
        rounded_sigma, integrated_intensity_main_beam, uncertainty_integ_intensity_main_beam,
                         integrated_intensity_fov, uncertainty_int_intensity_fov, molecule]

    # Call write_or_update_values to save the data
    write_or_update_values(file_name='spectrum_parameters_' + molecule + '.txt', new_values=values_to_text)



def plot_spectrum(source_name, molecule, type='central', save=False, plot=True):
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
        pos = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name, position=6)
        pos = float(pos)
        print('I found this value', pos)
    except:
        pos, FHWM, sigma = fit_gaussian_to_spectrum(spectrum, velocity,velo_range=[-20,30],plot=False)


    plt.figure()
    # plt.title("Averaged Spectrum ("+mole_name+") @"+dir_each)

    plt.xlabel("velocity (km/s)",size=14)
    plt.ylabel("Averaged Spectrum (K)",size=14)
    plt.title(title)
    plt.plot(velocity,spectrum,"-",color="black",lw=1)

    plt.tick_params(axis='both', direction='in')
    plt.xlim(pos - 10, pos+ 10)
    if save:
        plt.savefig(os.path.join('Figures/Spectra/', 'spectrum_'+source_name+'_'+molecule+'_'+type), bbox_inches='tight', dpi=300)

    if plot:
        plt.show()


def get_icrs_coordinates(object_name):
    '''
    get coordinates of astronomical objects by querying to SIMBAD names
    :param object_name:
    :return:
    '''
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
    # coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='fk5')
    # Return the ICRS coordinates in degrees
    return coord


def plot_moment_eight_map(source_name,molecule,use_sky_coord_object=True,save=False,plot=True):
    '''
    Plot the moment eight map of a source and molecule
    :param source_name:
    :param molecule:
    :param use_sky_coord_object:
    :param save:
    :param plot:
    :return:
    '''
    filename=source_name+'_'+molecule #'V347_Aur_HCO+'
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')
    moment_0 = DataAnalysis(os.path.join('moment_maps',source_name), filename+'_mom0.fits')

    print('molecule" ',data_cube.molecule)
    ### Here I can go from sky position to pixel coordinates
    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt', search_word=source_name)
    skycoord_object = get_icrs_coordinates(simbad_name)


    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05
        cmap = sns.color_palette("YlOrBr",as_cmap=True)

    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635
        cmap = sns.color_palette("YlGnBu",as_cmap=True)

    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    try:
        ### I first try to get the velocity centroid from the data saved in text file.
        noise_level = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name,
                                        position=1)
        float_noise_level = float(noise_level)


        moment_eight_noise_array = float_noise_level* np.array([5, 10, 15, 30, 60, 120])

    except:
        moment_eight_noise_array = np.array([0.2, 0.4, 0.8, 0.95])
        print('I can not read the file with velocity and noise levels')*10

    levels = moment_eight_noise_array

    moment_eight = create_moment_eight_map(source_name, molecule)

    plt.figure(figsize=(6, 7))
    fig1 = plt.subplot(projection=moment_0.wcs)

    mom8_im = fig1.imshow(moment_eight, cmap=cmap, origin='lower')#,vmax=0.5)
    cbar = plt.colorbar(mom8_im, fraction=0.048, pad=0.04, label='Peak Intensity (K)',
                        format=tkr.FormatStrFormatter('%1.1f'))
    contour = fig1.contour(moment_eight, levels=levels, colors="black")
    plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')


    offset_coordinates(fig1,skycoord_object)

    if use_sky_coord_object:
        ra_center = skycoord_object.ra.degree
        dec_center = skycoord_object.dec.degree
        print(skycoord_object.to_string('hmsdms'))
        IR_position = fig1.scatter(x=ra_center,y=dec_center, s=200, c='gray', transform=fig1.get_transform('icrs'), marker='x',
                                clip_on=False)


        ra_offset = 50/3600
        dec_offset = 45/3600
        beam_sky_coord_object = SkyCoord(ra=ra_center+ra_offset, dec=dec_center-dec_offset, unit=(u.deg, u.deg), frame='icrs')

        s = SphericalCircle(beam_sky_coord_object, aperture_radius * u.arcsec,
                            edgecolor='white', facecolor='gray',
                            transform=fig1.get_transform('fk5'),linewidth=2,linestyle='-')

        fig1.add_patch(s)

    plt.axis('square')

    if save:
        plt.savefig(os.path.join('Figures/Moment_maps/moment-eight/'+molecule+'/',filename+'_relative_coord'), bbox_inches='tight',dpi=300)
        # plt.savefig(os.path.join('Figures',filename+'_transparent'), bbox_inches='tight', transparent=True)

    if plot:
        plt.show()


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
    lon.set_ticks_position('b')
    lon.set_ticklabel_position('b')
    lon.set_axislabel_position('b')
    lon.set_ticks(spacing=20. * u.arcsec)
    lon.set_ticklabel(size=13)

    lat = overlay['lat']
    lat.set_format_unit(u.arcsec)
    lat.set_ticklabel(size=13)
    lat.set_ticks_position('l')
    lat.set_ticklabel_position('l')
    lat.set_axislabel_position('l')

    lon.set_axislabel(None)
    lat.set_axislabel(None)
    lon.set_auto_axislabel(False)
    lat.set_auto_axislabel(False)

    # lon.set_axislabel(r'$\Delta$RA',size=12)
    # lat.set_axislabel(r'$\Delta$DEC',size=12)

    # lat.set_axislabel(text=' ')
    # lon.set_axislabel(text=' ')



def create_moment_eight_map(source_name,molecule):
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

    filename=source_name+'_'+molecule #'V347_Aur_HCO+'
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')
    # moment_0 = DataAnalysis(os.path.join('moment_maps',source_name), filename+'_mom0.fits')


    try:
        ### I first try to get the velocity centroid from the data saved in text file.

        sigma_vel = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name,
                                        position=8)
        float_sigma_vel = float(sigma_vel)


        veloc = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name,
                                        position=6)

        float_veloc = float(veloc)

    except:
        # moment_eight_noise_array = np.array([0.2, 0.4, 0.8, 0.95])
        print('I can not read the file with velocity and noise levels')*10

    # levels = moment_eight_noise_array

    upper_idx= find_nearest_index(array=data_cube.vel, value=float_veloc-6*float_sigma_vel) # 6 sigma is for consistency with Carney
    lower_idx= find_nearest_index(array=data_cube.vel,value=float_veloc+6*float_sigma_vel)


    moment_eight = np.nanmax(data_cube.ppv_data[upper_idx:lower_idx,:,:],axis=0)

    return moment_eight


def peak_integrated_emission_from_map(source_name, molecule, moment_maps_folder='moment_maps', aperture=None, use_skycoord=True,):
    '''
    Find the peak integrated emission within the moment zero map.
    :param source_name:
    :param molecule:
    :param use_skycoord: True  For some sources obtained in SCAN mode, the coordinates  are wrong and we
    placed the source at the center - for these please set to FALSE!
    :return:
    '''

    filename = source_name+'_'+molecule
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')

    # moment_eight = create_moment_eight_map(source_name, molecule)
    moment_zero = find_moment_zero_map(source_name, molecule, moment_maps_folder)

    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt', search_word=source_name)
    skycoord_object = get_icrs_coordinates(simbad_name)


    if 'HCO+' in data_cube.molecule:
        if aperture is None:
            aperture_radius = 7.05
        else:
            aperture_radius = aperture/2.


    elif data_cube.molecule == 'C18O':
        if aperture is None:
            aperture_radius = 7.635
        else:
            aperture_radius = aperture/2.

    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    pixel_scale_ra = data_cube.header['CDELT1'] * 3600  # arcseconds per pixel
    aperture_radius_pixels = abs(aperture_radius/pixel_scale_ra)


    # x_center, y_center = moment_0.wcs.world_to_pixel(skycoord_object) ## This one if 2D cube
    x_center, y_center = data_cube.wcs.celestial.world_to_pixel(skycoord_object) ## This one if 3D cube


    if use_skycoord==False:
        #### Only use this if there is a problem with WCS and you need to calculate the spectrum at the center.
        ra_center = data_cube.wcs.celestial.all_pix2world(data_cube.nx / 2, data_cube.ny / 2, 0)[0]
        dec_center = data_cube.wcs.celestial.all_pix2world(data_cube.nx / 2, data_cube.ny / 2, 0)[1]
        skycoord_object = SkyCoord(ra=ra_center, dec=dec_center, unit='deg', frame='icrs')
        x_center, y_center = data_cube.wcs.celestial.world_to_pixel(skycoord_object) ## This one if 3D cube


    print ('HERE!!!!')
    print(x_center,y_center)


    print(f"These are the sky coordinates of your {source_name}: ", skycoord_object)

    # Initialize a list to store the pixel values within the aperture
    center_beam_values = []

    # Iterate over a square region, but filter by distance to make it circular
    for xx in range(int(x_center - aperture_radius_pixels), int(x_center + aperture_radius_pixels) + 1):
        for yy in range(int(y_center - aperture_radius_pixels), int(y_center + aperture_radius_pixels) + 1):
            # Calculate the distance from the center
            distance = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)

            # Check if the distance is within the aperture radius
            if distance <= aperture_radius_pixels:
                # Append the data at this pixel position
                center_beam_values.append(moment_zero[yy, xx])

    # Convert center_beam_values to a NumPy array for easy manipulation
    center_beam_values = np.array(center_beam_values)

    peak_integrated_emission = np.nanmax(center_beam_values)

    print('peak integrated emission within central beam: ',peak_integrated_emission, ' K')

    return peak_integrated_emission

def area_and_emission_of_map_above_threshold(source_name,molecule, moment_maps_folder='moment_maps',
                                             aperture=None, n_sigma=1,plot=True,only_whole_area=False):
    '''
    Computes the area of a map of all the emission
    above a given sigma noise level.
    :return: area in arcseconds.
    '''

    filename=source_name+'_'+molecule #'V347_Aur_HCO+'
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')

    print('molecule ',data_cube.molecule)

    if 'HCO+' in data_cube.molecule:
        if aperture is None:
            aperture_radius = 7.05
        else:
            aperture_radius = aperture/2.


    elif data_cube.molecule == 'C18O':
        if aperture is None:
            aperture_radius = 7.635
        else:
            aperture_radius = aperture/2.

    else:
        raise Exception("Sorry, I need to calculate such aperture radius")


    image_mom_0 = find_moment_zero_map(source_name, molecule, moment_maps_folder)
    # image_mom_0 = image_mom_0[20:-17,20:-17]

    try:
        ### I first try to get the velocity centroid from the data saved in text file.
        noise_level = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name,
                                        position=1)

        float_noise_level = float(noise_level)

        sigma_vel = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name,
                                        position=8)

        float_sigma_vel = float(sigma_vel)
        n_sigmas_velocity = 6
        moment_zero_noise = (0.2*n_sigmas_velocity*float_sigma_vel)**0.5*float_noise_level ## 0.2 is the binning in km/s
    except:
        raise Exception("Sorry, the moment zero noise to calculate the area")



    threshold = n_sigma * moment_zero_noise  # Adjust threshold factor if needed
    print('n sigma threshold ',threshold)
    # threshold = 0.13  # Adjusted for Perseus & V347 Aur
    # threshold = 0.065  # Adjusted for Orion
    # threshold = 0.3  # Adjust threshold factor if needed

    print('new threshold ',threshold)

    image_mom_0[image_mom_0 < threshold] = np.nan

    n_pixels_interest = np.count_nonzero(~np.isnan(image_mom_0))
    print('number of no-nan pixels: ',n_pixels_interest)
    print('size of array: ',data_cube.ny*data_cube.nx)

    pixel_area_arcsec2 = data_cube.cdelt_ra*data_cube.cdelt_dec
    area_of_significant_emission = abs(pixel_area_arcsec2*n_pixels_interest)
    print('Total area of  array: ', abs(pixel_area_arcsec2*data_cube.ny*data_cube.nx),' in squared arcsec')
    if only_whole_area:
        return abs(pixel_area_arcsec2*data_cube.ny*data_cube.nx)
    print('Area of significant values: ', area_of_significant_emission,' in squared arcsec')

    '''
    The factor of two is because  we give the radius or half of the FWHM, which is about 15'' in diameter.
    '''
    pix_per_beam = (2*aperture_radius)**2*np.pi / (4*np.log(2)*data_cube.cdelt_ra**2) # pix-per-beam = beam_size_area/pix_area

    print('Pixels per beam: ', pix_per_beam)

    total_emission = np.nansum(image_mom_0/pix_per_beam)

    print('Noise of the moment 1 map:', moment_zero_noise, ' K km/s')
    print('Total integrated emission:', total_emission,' K km/s')

    if plot:
        plt.imshow(image_mom_0,origin='lower')
        plt.show()
    return area_of_significant_emission, total_emission



def evaluate_gauss2d_fit(
    data: np.ndarray,
    fit,                        # astropy.modeling.models.Gaussian2D after fitting
    mask: np.ndarray | None = None,
) -> dict:
    """
    Build model on the data grid, compute residuals + simple quality metrics.
    Returns a dict with model, residuals, and numbers.
    """
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
    model = fit(xx, yy)

    if mask is None:
        mask = ~np.isfinite(data)
    valid = (~mask) & np.isfinite(data) & np.isfinite(model)

    resid = data - model
    resid_rms = float(mad_std(resid[valid]))  # robust RMS

    # degrees of freedom (N - number of free params)
    free_params = [
        p for p in fit.param_names
        if not getattr(getattr(fit, p), "fixed", False)
        and getattr(fit, p).tied is None
    ]
    N = int(valid.sum())
    dof = max(N - len(free_params), 1)

    # simple reduced-chi^2 (assumes homoskedastic noise ~ resid_rms)
    denom = resid_rms if resid_rms > 0 else 1.0
    red_chi2 = float(np.nansum((resid[valid] / denom) ** 2) / dof)

    # convenient extras
    x_fit = float(fit.x_mean.value)
    y_fit = float(fit.y_mean.value)
    sx = float(fit.x_stddev.value)
    sy = float(fit.y_stddev.value)
    theta_deg = float(np.degrees(float(fit.theta.value)))
    fwhm_x = 2.355 * sx
    fwhm_y = 2.355 * sy

    return {
        "model": model,
        "residuals": resid,
        "valid_mask": valid,
        "resid_rms": resid_rms,
        "reduced_chi2": red_chi2,
        "center": (x_fit, y_fit),
        "sigma": (sx, sy),
        "fwhm": (fwhm_x, fwhm_y),
        "theta_deg": theta_deg,
        "N": N,
        "dof": dof,
    }

def plot_fit_diagnostics(
    data: np.ndarray,
    model: np.ndarray,
    residuals: np.ndarray,
    valid_mask: np.ndarray | None = None,
    fit=None,
    wcs=None,                   # optional astropy WCS for RA/Dec axes
):
    """
    2×2 panel: data, model, residual map, residual histogram.
    If WCS is provided, RA/Dec axes are used for the 3 images.
    """
    # common stretch for data/model
    if valid_mask is None:
        valid_mask = np.isfinite(data) & np.isfinite(model)
    vmin, vmax = np.nanpercentile(data[valid_mask], [1, 99])
    norm = simple_norm(data[valid_mask], "linear", min_cut=vmin, max_cut=vmax)
    rmax = np.nanpercentile(np.abs(residuals[valid_mask]), 99)

    subplot_kw = {"projection": wcs} if wcs is not None else {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), subplot_kw=subplot_kw if wcs is not None else None)
    (ax_data, ax_model), (ax_resid, ax_hist) = axes

    im0 = ax_data.imshow(data, origin="lower", norm=norm)
    ax_data.set_title("Data")
    plt.colorbar(im0, ax=ax_data, fraction=0.046, pad=0.04)

    im1 = ax_model.imshow(model, origin="lower", norm=norm)
    ax_model.set_title("Best-fit Model")
    plt.colorbar(im1, ax=ax_model, fraction=0.046, pad=0.04)

    im2 = ax_resid.imshow(residuals, origin="lower", vmin=-rmax, vmax=+rmax)
    ax_resid.set_title("Residuals (data − model)")
    plt.colorbar(im2, ax=ax_resid, fraction=0.046, pad=0.04)

    flat_resid = sigma_clip(residuals[valid_mask], sigma=5, masked=False).ravel()
    ax_hist.hist(flat_resid, bins=50, histtype="step")
    ax_hist.set_title("Residual Histogram")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Count")

    # annotate & optional contours/center
    if fit is not None:
        x_fit, y_fit = float(fit.x_mean.value), float(fit.y_mean.value)
        try:
            levels = np.linspace(np.nanmin(model[valid_mask]), np.nanmax(model[valid_mask]), 6)[1:-1]
            ax_data.contour(model, levels=levels, colors="w", linewidths=0.8)
        except Exception:
            pass
        ax_data.plot(x_fit, y_fit, marker="x", ms=6, mew=1.5, color="w")

    for ax in (ax_data, ax_model, ax_resid):
        ax.set_xlabel("x [pix]" if wcs is None else "RA")
        ax.set_ylabel("y [pix]" if wcs is None else "Dec")

    plt.tight_layout()
    plt.show()

def area_and_emission_via_gaussian(source_name, molecule, moment_maps_folder='moment_maps',aperture=None,
                                   save_fig=False, diagnostics=False):
    """
    Fit a 2D Gaussian to the moment-0 map with the center allowed to move up to ±5 arcsec
    from the near-IR source coordinates. Define Robs from the fitted FWHM_circ (see note below).
    Area = π * Robs^2 (arcsec^2). Total emission is the moment-0 sum within a circular aperture
    of radius Robs, normalized by pixels-per-beam.

    NOTE: If you want Robs to be the half-max radius, set Robs_arcsec = 0.5 * FWHM_circ_arcsec.
    """

    # ---------- helpers ----------
    def _safe_aperture_sum(data2d, x0, y0, r_pix):
        if not np.isfinite(r_pix) or r_pix <= 0:
            return 0.0, 0
        ny, nx = data2d.shape
        if (x0 + r_pix < 0) or (y0 + r_pix < 0) or (x0 - r_pix > nx - 1) or (y0 - r_pix > ny - 1):
            return 0.0, 0
        finite_mask = np.isfinite(data2d)
        data_clean = np.where(finite_mask, data2d, 0.0)
        ap = CircularAperture((x0, y0), r=r_pix)
        m = ap.to_mask(method='exact')
        cutout = m.multiply(data_clean)
        if cutout is None:
            return 0.0, 0
        finite_cut = m.multiply(finite_mask.astype(float))
        n_finite = int(np.count_nonzero(finite_cut > 0))
        return float(np.sum(cutout)), n_finite

    # ---------- load data ----------
    filename = f"{source_name}_{molecule}"
    data_cube = DataAnalysis(os.path.join('sdf_and_fits', source_name), filename + '.fits')
    image_mom_0 = find_moment_zero_map(source_name, molecule, moment_maps_folder)  # 2D array

    # Near-IR center from SIMBAD
    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt',
                                             search_word=source_name)
    skycoord_object = get_icrs_coordinates(simbad_name)
    x0, y0 = data_cube.wcs.celestial.world_to_pixel(skycoord_object)

    # Pixel scales (arcsec/pixel); use absolute values
    pixscale_x = abs(data_cube.cdelt_ra)
    pixscale_y = abs(data_cube.cdelt_dec)
    pixel_area_arcsec2 = pixscale_x * pixscale_y

    if 'HCO+' in data_cube.molecule:
        if aperture is None:
            aperture_radius = 7.05
        else:
            aperture_radius = aperture/2.


    elif data_cube.molecule == 'C18O':
        if aperture is None:
            aperture_radius = 7.635
        else:
            aperture_radius = aperture/2.
    else:
        raise Exception("Sorry, I need to calculate such aperture radius")

    '''
    The factor of two is because  we give the radius or half of the FWHM, which is about 15'' in diameter.
    pi * FWHM^2 / (4 ln2)
    '''
    pix_per_beam = ((2 * aperture_radius) ** 2 * np.pi / (4.0 * np.log(2.0))) / pixel_area_arcsec2

    if not np.isfinite(pix_per_beam) or pix_per_beam <= 0:
        raise ValueError(f"Invalid pix_per_beam: {pix_per_beam}")

    # ---------- prep data for fitting ----------
    data = np.array(image_mom_0, dtype=float)
    base_nan_mask = ~np.isfinite(data)
    if np.all(base_nan_mask):
        return (0.0, 0.0, {"reason": "all_nan_map"}) if diagnostics else (0.0, 0.0)

    # very loose clip or none; you can set sigma=None to disable clipping entirely
    clipped = sigma_clip(data, sigma=120, masked=True)
    clip_mask = getattr(clipped, 'mask', None)
    clip_mask = np.array(clip_mask, dtype=bool) if clip_mask is not None else np.zeros_like(data, dtype=bool)
    fit_mask = base_nan_mask | clip_mask
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]

    # Ensure starting center is inside image
    x0 = float(np.clip(x0, 0, data_cube.nx - 1))
    y0 = float(np.clip(y0, 0, data_cube.ny - 1))

    # ---------- build & run fit (center free but bounded to ±5 arcsec) ----------
    amp0 = np.nanmax(data)*0.5
    sigx0 = 2.0  # px initial guess / the pixel size is 2 arcsec per pixel
    sigy0 = 2.0
    theta0 = 0.0

    g_init = models.Gaussian2D(amplitude=amp0,
                               x_mean=x0, y_mean=y0,
                               x_stddev=sigx0, y_stddev=sigy0,
                               theta=theta0)

    # CHANGED: allow center to move, with bounds of ±5 arcsec (per axis), clipped to image
    max_shift_arcsec = 1.0
    dx_pix = max_shift_arcsec / pixscale_x if pixscale_x > 0 else 0.0
    dy_pix = max_shift_arcsec / pixscale_y if pixscale_y > 0 else 0.0

    x_min = float(max(0, x0 - dx_pix))
    x_max = float(min(data_cube.nx - 1, x0 + dx_pix))
    y_min = float(max(0, y0 - dy_pix))
    y_max = float(min(data_cube.ny - 1, y0 + dy_pix))

    g_init.x_mean.fixed = False
    g_init.y_mean.fixed = False
    g_init.x_mean.min = x_min
    g_init.x_mean.max = x_max
    g_init.y_mean.min = y_min
    g_init.y_mean.max = y_max

    # Your width bounds (you can keep your beam-based mins if you prefer)
    g_init.x_stddev.min = 3.4
    g_init.y_stddev.min = 3.4
    g_init.x_stddev.max = max(10.0, data.shape[1] / 2.0)
    g_init.y_stddev.max = max(10.0, data.shape[0] / 2.0)

    fitter = fitting.LevMarLSQFitter()
    fit = fitter(g_init, xx[~fit_mask], yy[~fit_mask], data[~fit_mask])

    # qa = evaluate_gauss2d_fit(data, fit, mask=fit_mask)
    #
    # # quick numbers for logs
    # print(
    #     f"RMS={qa['resid_rms']:.3g}  "
    #     f"chi2_red≈{qa['reduced_chi2']:.2f}  "
    #     f"center=({qa['center'][0]:.2f}, {qa['center'][1]:.2f})  "
    #     f"FWHM=({qa['fwhm'][0]:.2f}, {qa['fwhm'][1]:.2f}) px"
    # )
    #
    # # plots (use your celestial WCS if available)
    # plot_fit_diagnostics(
    #     data=data,
    #     model=qa["model"],
    #     residuals=qa["residuals"],
    #     valid_mask=qa["valid_mask"],
    #     fit=fit,
    #     wcs=None  # or data_cube.wcs.celestial
    # )

    # Use fitted center from now on
    x_fit = float(fit.x_mean.value)
    y_fit = float(fit.y_mean.value)

    # ---------- FWHM & Robs ----------
    ### Here the standard dev (sigma) of the 2D gaussian is transformed into a FWHM.
    k = 2.0 * np.sqrt(2.0 * np.log(2.0))
    FWHM_x_arcsec = k * float(fit.x_stddev.value) * pixscale_x
    FWHM_y_arcsec = k * float(fit.y_stddev.value) * pixscale_y
    FWHM_circ_arcsec = float(np.sqrt(FWHM_x_arcsec * FWHM_y_arcsec))

    # Robs_arcsec = 0.5 * FWHM_circ_arcsec
    Robs_arcsec = FWHM_circ_arcsec
    # This is effectively the "diameter" of the Gaussian.
    ## This is used based on the definition of the Carney+2016,
    ## where the effective radius is set equal to the FHWM of the Gaussian.

    print('#'*19)
    print('Robs_arcsec : ',Robs_arcsec)
    print('#'*19)

    # Robs_arcsec = 35
    # radius sanity clamp
    fov_x_arcsec = data_cube.nx * pixscale_x
    fov_y_arcsec = data_cube.ny * pixscale_y
    max_reasonable = 0.75 * max(fov_x_arcsec, fov_y_arcsec)
    rob_clamped = False
    if (not np.isfinite(Robs_arcsec)) or (Robs_arcsec <= 0) or (Robs_arcsec > max_reasonable):
        Robs_arcsec = max_reasonable
        rob_clamped = True

    area_arcsec2 = float(np.pi * Robs_arcsec ** 2 )#/ (4.0 * np.log(2.0)))



    # ---------- robust emission inside Robs (centered on the FITTED center) ----------
    if not np.isfinite(pixscale_x) or pixscale_x == 0:
        raise ValueError("Invalid pixel scale (pixscale_x).")
    Robs_pix = max(Robs_arcsec / pixscale_x, 0.0)

    sum_mom0_inside, n_finite = _safe_aperture_sum(image_mom_0, x_fit, y_fit, Robs_pix)
    total_emission = 0.0 if n_finite == 0 else float(sum_mom0_inside / pix_per_beam)

    # ---------- optional plotting ----------

    if save_fig:
        ny, nx = data.shape
        model_img = fit(xx, yy)
        residuals = data - model_img

        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 99)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), subplot_kw={'projection': data_cube.wcs.celestial})
        im0 = axes[0].imshow(data, origin='lower', cmap='inferno',
                             norm=simple_norm(data, 'linear', vmin=vmin, vmax=vmax))
        # mark NIR center and fitted center
        axes[0].plot(x0,   y0,  '+', color='cyan',  ms=10, mew=2, label='NIR center')
        axes[0].plot(x_fit,y_fit,'x', color='white', ms=10, mew=2, label='Fitted center')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].set_title(f"{source_name} {molecule} | Data")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(model_img, origin='lower', cmap='inferno',
                             norm=simple_norm(model_img, 'linear', vmin=vmin, vmax=vmax))
        axes[1].plot(x_fit, y_fit, 'x', color='white', ms=10, mew=2)
        axes[1].set_title("Gaussian Model")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(residuals, origin='lower', cmap='coolwarm')
        axes[2].plot(x_fit, y_fit, 'x', color='k', ms=10, mew=2)
        axes[2].set_title("Residuals")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Overlay FWHM ellipse (semi-axes = FWHM/2) and Robs circle at the FITTED center
        a_pix = (FWHM_x_arcsec / 2.0) / pixscale_x
        b_pix = (FWHM_y_arcsec / 2.0) / pixscale_y
        ell = EllipticalAperture((x_fit, y_fit), a=a_pix, b=b_pix, theta=float(fit.theta.value))
        circ = CircularAperture((x_fit, y_fit), r=Robs_pix)
        for ax in (axes[0], axes[1]):
            # ell.plot(ax=ax, color='lime', lw=1.8, label='FWHM ellipse (half-max)')
            circ.plot(ax=ax, color='deepskyblue', lw=1.6, label='FWHM Gaussian = ' + str(round(Robs_arcsec,1)) + '″')
            ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join('Figures/gaussian_area/',
                                 source_name+'_'+molecule+'_'+'_max_shift_'+str(max_shift_arcsec)+'_'+today+'.png'), bbox_inches='tight',dpi=300)
        # plt.show()

    # ---------- diagnostics ----------
    if diagnostics:
        ny, nx = data.shape
        model_img = fit(xx, yy)
        residuals = data - model_img
        valid = np.isfinite(data) & np.isfinite(model_img)
        res = residuals[valid]
        dat = data[valid]
        sse = float(np.sum(res**2))
        sst = float(np.sum((dat - np.mean(dat))**2)) if np.size(dat) > 1 else np.nan
        r2_like = float(1.0 - (sse / sst)) if (np.isfinite(sst) and sst > 0) else np.nan

        # center shift diagnostics
        dx_fit_arc = (x_fit - x0) * pixscale_x
        dy_fit_arc = (y_fit - y0) * pixscale_y
        shift_arc = float(np.hypot(dx_fit_arc, dy_fit_arc))
        hit_x_bound = bool(np.isclose(x_fit, x_min) or np.isclose(x_fit, x_max))
        hit_y_bound = bool(np.isclose(y_fit, y_min) or np.isclose(y_fit, y_max))

        diag = {
            "nir_center_pix": (float(x0), float(y0)),
            "fitted_center_pix": (float(x_fit), float(y_fit)),
            "center_shift_arcsec": {"dx": float(dx_fit_arc), "dy": float(dy_fit_arc), "dr": shift_arc},
            "center_bounds_pix": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            "center_hit_bound": {"x": hit_x_bound, "y": hit_y_bound},
            "amplitude": float(fit.amplitude.value),
            "sigma_x_pix": float(fit.x_stddev.value),
            "sigma_y_pix": float(fit.y_stddev.value),
            "theta_rad": float(fit.theta.value),
            "FWHM_x_arcsec": float(FWHM_x_arcsec),
            "FWHM_y_arcsec": float(FWHM_y_arcsec),
            "FWHM_circ_arcsec": float(FWHM_circ_arcsec),
            "Robs_arcsec": float(Robs_arcsec),
            "pix_per_beam": float(pix_per_beam),
            "n_finite_in_aperture": int(n_finite),
            "robust_radius_clamped": bool(rob_clamped),
            "r2_like": r2_like,
            "fit_message": getattr(fitter, "fit_info", {}).get("message", None),
            "fit_ierr": getattr(fitter, "fit_info", {}).get("ierr", None),
            "fit_nfev": getattr(fitter, "fit_info", {}).get("nfev", None),
        }
        # print('center shift = ', shift_arc)
        print(diag)
        return area_arcsec2, total_emission#, diag

    return area_arcsec2, total_emission

def find_moment_zero_map(source_name,molecule,moment_maps_folder='moment_maps'):
    '''
    Create moment maps using BTS coode.
    Need to give the data, velocity, and rms levels.
    The moment maps will be computed using a given velocity position
    previously calculated and a velocity dispersion given from gaussian fit.
    A 3 sigma is what we will use for now.
    We adopt a sigma clipping of 1*rms.
    '''
    filename = source_name + '_' + molecule  # 'V347_Aur_HCO+'
    # data_cube = DataAnalysis(os.path.join('sdf_and_fits', source_name), filename + '.fits')
    moment_0 = DataAnalysis(os.path.join(moment_maps_folder, source_name), filename + '_mom0.fits')

    image_mom_0 = moment_0.ppv_data

    image_mom_0 = image_mom_0


    return image_mom_0

def plot_moment_zero_map(source_name,molecule,use_sky_coord_object=False,percentile_outlier=100,save=False,plot=True,rotate=False):
    '''
    Create moment maps using BTS coode.
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

    filename=source_name+'_'+molecule #'V347_Aur_HCO+'
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')
    moment_0 = DataAnalysis(os.path.join('moment_maps',source_name), filename+'_mom0.fits')

    print('molecule: ',data_cube.molecule)
    ### Here I can go from sky position to pixel coordinates
    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt', search_word=source_name)

    skycoord_object = get_icrs_coordinates(simbad_name)

    print('skycoord are: ',skycoord_object)
    print ('wcs', moment_0.wcs.wcs_pix2world(moment_0.nx/2, moment_0.ny/2, 0))

    image_mom_0 = find_moment_zero_map(source_name, molecule)

    # rotated image only for SCAN data, Check MAP_PA, use *-1 value

    # image_mom_0 = np.nan_to_num(image_mom_0, nan=2.8)
    if rotate:
        image_mom_0 = rotate(image_mom_0, angle=rotate, reshape=False)


    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05
        cmap = sns.color_palette("YlOrBr",as_cmap=True)

    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635
        cmap = sns.color_palette("YlGnBu",as_cmap=True)

    elif data_cube.molecule == 'CO':
        aperture_radius = 7.635
        cmap = sns.color_palette("YlGnBu",as_cmap=True)

    else:
        raise Exception("Sorry, I need to calculate such aperture radius")


    # Apply a threshold to mask large values
    if percentile_outlier:
        threshold = np.nanpercentile(image_mom_0.reshape(-1), percentile_outlier)
        percentile_outlier_text = formatted_string = str(percentile_outlier).replace('.', 'p')
        image_mom_0[image_mom_0 > threshold] = np.nan


    try:
        ### I first try to get the velocity centroid from the data saved in text file.
        noise_level = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name,
                                        position=1)

        float_noise_level = float(noise_level)

        sigma_vel = find_word_in_file(file_name='spectrum_parameters_'+molecule+'.txt', search_word=source_name,
                                        position=8)

        float_sigma_vel = float(sigma_vel)

        ### This is the way Carney et al. 2016 defined the noise
        # moment_zero_noise = (0.2*6*float_sigma_vel)**0.5*float_noise_level ## 0.2 is the binning in km/s
        # print('old noise', moment_zero_noise)

        moment_zero_noise = (float_sigma_vel*6/0.2)**0.5*float_noise_level*0.2 ## 0.2 is the binning in km/s

        print('new noise_level ',moment_zero_noise)
        print('number of velocity pixles', float_sigma_vel*6/0.2)

        if 'HCO+' in data_cube.molecule:
            moment_zero_noise_array = moment_zero_noise* np.array([1, 3, 5,10, 20, 40, 60, 80, 100,150])
        elif data_cube.molecule == 'C18O':
            moment_zero_noise_array = moment_zero_noise* np.array([1,3,5,7,9,12,15,20,30,40])
    # print(noise_level)
        # print(sigma_vel)
        # print (moment_zero_noise)

    except:
        print("can't compute the noise for contours, I usea peak intensity intervals")
        moment_zero_noise_array = np.array([0.1,0.3,0.5,0.6, 0.8, 0.95])*np.nanmax(image_mom_0)

    levels = moment_zero_noise_array
    print('levels', levels)
    ## Moment zero
    ## Moment zero

    # plt.figure(figsize=(6, 7))
    # fig1 = plt.subplot(projection=wcs)
    fig1 = plt.subplot(projection=moment_0.wcs)
    ax = fig1.figure
    mom0_im = fig1.imshow(image_mom_0, cmap=cmap, origin='lower',vmin=0.0)#,vmax=0.5)
    # divider = make_axes_locatable(fig1)
    # cax = divider.append_axes(position="top", size="10%", pad=0.05)
    # cbar_ax = fig1.add_axes([0.3, 0.92, 0.4, 0.03])  # 40% width, centered, above plot
    cbar = ax.colorbar(mom0_im, location='top', orientation='horizontal', format='%.1f',fraction=0.047,pad=0.01)# fraction=0.048, pad=0.04)
    cbar.set_label(label='Integrated Intensity ' +r'(K km s$^{-1}$)', size=14)
    cbar.ax.tick_params(labelsize=13, direction='in')

    contour = fig1.contour(image_mom_0, levels=levels, colors="black")
    plt.clabel(contour, inline=True, fontsize=8, fmt='%1.2f')
    plt.tick_params(axis='both', labelsize=24)  # 'both' = x and y

    fig1.set_xlabel('RA',size=12)
    fig1.set_ylabel('DEC',size=12)



    print('These are the sky coordinates of your object: ',skycoord_object)

    # u.hourangle, u.deg

    if use_sky_coord_object:

        offset_coordinates(fig1, skycoord_object)

        ra_center = skycoord_object.ra.degree
        dec_center = skycoord_object.dec.degree
        print(skycoord_object.to_string('hmsdms'))
        IR_position = fig1.scatter(x=ra_center,y=dec_center, s=150, c='gray', transform=fig1.get_transform('icrs'), marker='x',
                                clip_on=False,linewidths=3.0)
        sky_center = SkyCoord(ra=ra_center, dec=dec_center, unit='deg', frame='icrs')

        ### FOr DG Tau
        # ra_offset = 20/3600
        # dec_offset = 80/3600

        ra_offset = 50/3600
        dec_offset = 45/3600
        beam_sky_coord_object = SkyCoord(ra=ra_center+ra_offset, dec=dec_center-dec_offset, unit=(u.deg, u.deg), frame='icrs')
        s = SphericalCircle(beam_sky_coord_object, aperture_radius * u.arcsec,
                            edgecolor='white', facecolor='gray',
                            transform=fig1.get_transform('fk5'),linewidth=2,linestyle='-')

        fig1.add_patch(s)

        x_center, y_center = moment_0.wcs.world_to_pixel(sky_center)  ## This one if 2D cube

        ### For most stars
        pixel_limits_ra = 30.5
        pixel_limits_dec = 30.5

        #### For T Tauri
        pixel_limits_ra = 34.
        pixel_limits_dec = 34.

        fig1.set_xlim(x_center - pixel_limits_ra, x_center + pixel_limits_ra)
        fig1.set_ylim(y_center - pixel_limits_dec, y_center + pixel_limits_dec)


    else:


        ra_center = moment_0.wcs.wcs_pix2world(moment_0.nx/2, moment_0.ny/2, 0)[0]
        dec_center = moment_0.wcs.wcs_pix2world(moment_0.nx/2, moment_0.ny/2, 0)[1]

        sky_center = SkyCoord(ra=ra_center, dec=dec_center, unit='deg', frame='icrs')

        offset_coordinates(fig1, sky_center)

        IR_position = fig1.scatter(x=ra_center,y=dec_center, s=150, c='gray', transform=fig1.get_transform('icrs'), marker='x',
                                clip_on=False,linewidths=3.0)

        ra_offset = 50/3600
        dec_offset = 45/3600
        beam_sky_coord_object = SkyCoord(ra=ra_center+ra_offset, dec=dec_center-dec_offset, unit=(u.deg, u.deg), frame='icrs')
        s = SphericalCircle(beam_sky_coord_object, aperture_radius * u.arcsec,
                            edgecolor='white', facecolor='gray',
                            transform=fig1.get_transform('fk5'),linewidth=2,linestyle='-')

        fig1.add_patch(s)


        ### if SCAN DATA set some smaller limits
        x_center, y_center = moment_0.wcs.world_to_pixel(sky_center)  ## This one if 2D cube
        print('center in pixels', x_center, y_center)
        pixel_limits_ra = 34
        pixel_limits_dec = 34
        fig1.set_xlim(x_center - pixel_limits_ra, x_center + pixel_limits_ra)
        fig1.set_ylim(y_center - pixel_limits_dec, y_center + pixel_limits_dec)


    ### Change the limits
    print('size of cube in pixels')
    print(data_cube.nx,data_cube.ny)

    # x_center, y_center = moment_0.wcs.world_to_pixel(skycoord_object) ## This one if 2D cube
    # pixel_limits_ra = 35
    # pixel_limits_dec = 35
    # fig1.set_xlim(x_center - pixel_limits_ra, x_center + pixel_limits_ra)
    # fig1.set_ylim(y_center - pixel_limits_dec, y_center + pixel_limits_dec)

    # plt.axis('square')
    fig1.tick_params(axis='y', direction='in')

    if save:
        plt.savefig(os.path.join('Figures/Moment_maps/moment-zero/',
                                 filename+'clip_'+percentile_outlier_text+'_coord_offset'+today), bbox_inches='tight',dpi=300)
        # plt.savefig(os.path.join('Figures',filename+'_transparent'), bbox_inches='tight', transparent=True)

    if plot:
        plt.show()



def mass_produce_moment_maps(folder_fits, molecule='C18O'):
    """
    Processes all folders within 'folder_fits' to generate moment maps and spectra
    for specified molecules (default is 'C18O').

    Plot moment-zero map.

    All operations are run in no-plotting mode, saving the maps.

    Args:
        folder_fits (str): Path to the main folder containing subfolders with fits data.
        molecule (str): Name of the molecule to process ('HCO+' or 'C18O').
    """
    folder_list = sorted(next(os.walk(folder_fits))[1])  # List of subfolders
    print("Folders found:", folder_list)

    print(folder_list)
    for sources in folder_list:
        try:
            # Check if the necessary file exists before running the function
            filename = sources + '_' + molecule
            fits_file_path = os.path.join(folder_fits, sources, f"{filename}.fits")
            if not os.path.isfile(fits_file_path):
                print(f"No such file: {fits_file_path}. Skipping this folder.")
                continue  # Move to the next folder if the file doesn't exist

            # Generate the moment-zero map
            plot_moment_zero_map(sources, molecule, save=True, use_sky_coord_object=True, plot=False)

            # Generate the moment-eight map
            # plot_moment_eight_map(sources, molecule, save=True, plot=True)

        except IndexError as err:
            print(f"Map for {sources} was not produced. Check the moment maps.")
            print(f"An error occurred: {err}")

        except Exception as e:
            print(f"An unexpected error occurred with {sources}: {e}")

def mass_produce_spectral_plots(folder_fits, molecule):
    """
    Processes all folders within 'folder_fits' to generate moment maps and spectra
    for specified molecules (default is 'C18O').

    Plot moment-zero map.

    All operations are run in no-plotting mode, saving the maps.

    Args:
        folder_fits (str): Path to the main folder containing subfolders with fits data.
        molecule (str): Name of the molecule to process ('HCO+' or 'C18O').
    """
    folder_list = sorted(next(os.walk(folder_fits))[1])  # List of subfolders
    print("Folders found:", folder_list)

    for sources in folder_list:
        try:
            # Check if the necessary file exists before running the function
            filename = sources + '_' + molecule
            fits_file_path = os.path.join(folder_fits, sources, f"{filename}.fits")
            if not os.path.isfile(fits_file_path):
                print(f"No such file: {fits_file_path}. Skipping this folder.")
                continue  # Move to the next folder if the file doesn't exist

            # Generate the moment-zero map
            plot_spectrum(sources, molecule, type='central', save=True, plot=False)
            plot_spectrum(sources, molecule, type='fov', save=True, plot=False)

        except IndexError as err:
            print(f"Spectral line for {sources} was not produced. Check the plots.")
            print(f"An error occurred: {err}")

        except Exception as e:
            print(f"An unexpected error occurred with {sources}: {e}")

def mass_calculate_spectral_properties(folder_fits, molecule):
    """
    Processes all folders within 'folder_fits' to generate moment maps and spectra
    for specified molecules (default is 'C18O').

    Plot moment-zero map.

    All operations are run in no-plotting mode, saving the maps.

    Args:
        folder_fits (str): Path to the main folder containing subfolders with fits data.
        molecule (str): Name of the molecule to process ('HCO+' or 'C18O').
    """
    folder_list = sorted(next(os.walk(folder_fits))[1])  # List of subfolders
    print("Folders found:", folder_list)

    for sources in folder_list:
        try:
            # Check if the necessary file exists before running the function
            filename = sources + '_' + molecule
            fits_file_path = os.path.join(folder_fits, sources, f"{filename}.fits")
            if not os.path.isfile(fits_file_path):
                print(f"No such file: {fits_file_path}. Skipping this folder.")
                continue  # Move to the next folder if the file doesn't exist

            retrieve_and_write_spectral_properties(sources, molecule,plot=True)

        except IndexError as err:
            print(f"An error occurred: {err}")

        except Exception as e:
            print(f"An unexpected error occurred with {sources}: {e}")

def mass_measurement_from_molecular_lines(source_name, molecule,distance_pc):
    # New input values
    # T_mb_dv_ = 0.7  # Integrated intensity in K km/s
    # beam_area_arcsec2 = 6500  # Extended source area in arcsec^2

    beam_area_arcsec2, T_mb_dv = area_and_emission_of_map_above_threshold(source_name, molecule, n_sigma=1, plot=False)
    beam_area_arcsec2=177 # this is  a 15'' aperture

    print('Integrated intensity in K km/s', T_mb_dv)
    print('Extended source area in arcsec^2', beam_area_arcsec2)

    X_C18O = 1.7e-7
    mass_msun_new = (4.81e-13 *
                     (distance_pc / 140) ** 2 *
                     (1 / X_C18O) *
                     T_mb_dv *
                     beam_area_arcsec2)

    print('envelope mass above 1 sigma', mass_msun_new)

    beam_area_arcsec2, T_mb_dv = area_and_emission_of_map_above_threshold(source_name, molecule, n_sigma=3, plot=True)
    beam_area_arcsec2=177 # this is  a 15'' aperture

    print('Integrated intensity in K km/s', T_mb_dv)
    print('Extended source area in arcsec^2', beam_area_arcsec2)

    X_C18O = 1.7e-7
    mass_msun_new = (4.81e-13 *
                     (distance_pc / 140) ** 2 *
                     (1 / X_C18O) *
                     T_mb_dv *
                     beam_area_arcsec2)

    print('envelope mass above 3 sigma', mass_msun_new)

    return mass_msun_new


if __name__ == "__main__":

    # source_name = 'EC92'
    source_name = 'IRAS05379-0758'
    molecule ='HCO+'
    # molecule ='C18O'
    # distance = 130
    ## Step 0
    # retrieve_and_write_spectral_properties(source_name, molecule, noskycoord=False)

    ### Step 1 creates a plot of the spectrum
    plot_spectrum(source_name, molecule,type='central',save=True)
    # plot_spectrum(source_name, molecule,type='fov',save=False)

    ### Step 3
    ### Plot the maps
    # area_and_emission_of_map_above_threshold(source_name, molecule, n_sigma=1)
    # plot_moment_zero_map(source_name,molecule,save=True,use_sky_coord_object=True,plot=True,percentile_outlier=99.3)
    # plot_moment_eight_map(source_name,molecule,save=False)

    #### Mass produce moment maps
    # mass_calculate_spectral_properties('sdf_and_fits', molecule)
    # mass_produce_moment_maps('sdf_and_fits',molecule)
    # mass_produce_spectral_plots('sdf_and_fits',molecule)

    # peak_integrated_emission_from_map(source_name, molecule)
    # mass_measurement_from_molecular_lines(source_name, molecule,distance_pc=distance)