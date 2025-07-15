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

    lat = overlay['lat']
    lat.set_format_unit(u.arcsec)
    lat.set_ticklabel()
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


def peak_integrated_emission_from_map(source_name, molecule, use_skycoord=True):
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
    moment_zero = create_moment_zero_map(source_name, molecule)

    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt', search_word=source_name)
    skycoord_object = get_icrs_coordinates(simbad_name)


    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05 ## This is in arcsec

    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635 ## This is in arcsec

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

def area_and_emission_of_map_above_threshold(source_name,molecule,n_sigma=1,plot=True,only_whole_area=False):
    '''
    Computes the area of a map of all the emission
    above a given sigma noise level.
    :return: area in arcseconds.
    '''

    filename=source_name+'_'+molecule #'V347_Aur_HCO+'
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')

    print('molecule ',data_cube.molecule)

    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05
        # cmap = sns.color_palette("YlOrBr",as_cmap=True)

    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635
        # cmap = sns.color_palette("YlGnBu",as_cmap=True)

    else:
        raise Exception("Sorry, I need to calculate such aperture radius")


    image_mom_0 = create_moment_zero_map(source_name, molecule)
    image_mom_0 = image_mom_0[3:-3,3:-3]

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


def create_moment_zero_map(source_name,molecule):
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
    moment_0 = DataAnalysis(os.path.join('moment_maps', source_name), filename + '_mom0.fits')

    image_mom_0 = moment_0.ppv_data

    image_mom_0 = image_mom_0


    return image_mom_0

def plot_moment_zero_map(source_name,molecule,use_sky_coord_object=False,percentile_outlier=100,save=False,plot=True):
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

    image_mom_0 = create_moment_zero_map(source_name, molecule)

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

        moment_zero_noise = (0.2*6*float_sigma_vel)**0.5*float_noise_level ## 0.2 is the binning in km/s

        moment_zero_noise_array = moment_zero_noise* np.array([1, 3, 5,10,20,50,80,100,150])
        # print(noise_level)
        # print(sigma_vel)
        # print (moment_zero_noise)
    except:
        print("can't compute the noise for contours, I usea peak intensity intervals")
        moment_zero_noise_array = np.array([0.5,0.6, 0.8, 0.95])*np.nanmax(image_mom_0)

    levels = moment_zero_noise_array

    ## Moment zero
    ## Moment zero

    # plt.figure(figsize=(6, 7))
    # fig1 = plt.subplot(projection=wcs)
    fig1 = plt.subplot(projection=moment_0.wcs)
    mom0_im = fig1.imshow(image_mom_0, cmap=cmap, origin='lower')#,vmax=0.5)
    # divider = make_axes_locatable(fig1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mom0_im, fraction=0.048, pad=0.04, format='%.1f')
    cbar.set_label(label='Integrated Intensity ' +r'(K km s$^{-1}$)', size=14)

    contour = fig1.contour(image_mom_0, levels=levels, colors="black")
    plt.clabel(contour, inline=True, fontsize=8, fmt='%1.2f')

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
        pixel_limits_ra = 35
        pixel_limits_dec = 35
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

    if save:
        plt.savefig(os.path.join('Figures/Moment_maps/moment-zero/',
                                 filename+'clip_'+percentile_outlier_text+'_coord_offset'), bbox_inches='tight',dpi=300)
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

    # source_name = 'IRS5'
    source_name = 'DG-Tau'
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
    # plot_moment_zero_map(source_name,molecule,save=True,use_sky_coord_object=True,plot=True,percentile_outlier=100.0)
    # plot_moment_eight_map(source_name,molecule,save=False)

    #### Mass produce moment maps
    # mass_calculate_spectral_properties('sdf_and_fits', molecule)
    # mass_produce_moment_maps('sdf_and_fits',molecule)
    # mass_produce_spectral_plots('sdf_and_fits',molecule)

    # peak_integrated_emission_from_map(source_name, molecule)
    # mass_measurement_from_molecular_lines(source_name, molecule,distance_pc=distance)