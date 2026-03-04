import os
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from Process_JCMT_data import DataAnalysis
from data_cube_analysis import fit_gaussian_to_spectrum, write_or_update_values\
    , calculate_peak_SNR, integrate_flux_over_velocity, fit_gaussian_2d, find_nearest_index

def get_spectrum_data(source_name, molecule, type='central', **kwargs):
    """
    Returns (spectrum, velocity, title_piece) for one molecule and one spectrum type.
    kwargs get passed to the underlying makers (e.g., noskycoord for central).
    """
    if type.lower() == 'fov':
        spectrum, velocity = make_averaged_spectrum_data(source_name, molecule)
        title_piece = "FOV averaged spectrum"
    else:
        # allow noskycoord etc.
        spectrum, velocity = make_central_spectrum_data(source_name, molecule, **kwargs)
        title_piece = "Central beam spectrum"
    return spectrum, velocity, title_piece

def get_velocity_centroid(source_name, molecule, spectrum, velocity, velo_range=(-20, 30)):
    """
    Try reading centroid from file; otherwise fit.
    Returns pos (float).
    """
    try:
        pos = find_word_in_file(
            file_name='spectrum_parameters_' + molecule + '.txt',
            search_word=source_name,
            position=6
        )
        pos = float(pos)
        print(f'[{molecule}] found centroid in file: {pos}')
    except Exception:
        pos, FWHM, sigma = fit_gaussian_to_spectrum(
            spectrum, velocity, velo_range=list(velo_range), plot=False
        )
        print(f'[{molecule}] fit centroid: {pos}')
    return pos

def plot_spectrum_two_molecules(
    source_name,
    molecules=("HCO+", "C18O"),
    type="central",
    save=False,
    plot=True,
    align_velocity_to=None,   # None, or one of the molecules, e.g. "HCO+"
    xlim_mode="union",        # "union" or "center_on_first" or "center_on_each"
    window_kms=10,
    **kwargs
):
    """
    Plot spectra of ONE source for TWO molecules on the same figure.

    Parameters
    ----------
    molecules : tuple/list of 2 strings
    type : 'central' or 'fov'
    align_velocity_to : if set to a molecule name, shift the other molecule's velocity axis
                        so that both centroids line up (useful for visual comparison).
    xlim_mode :
        - 'union': xlim covers both lines (based on centroids ± window_kms)
        - 'center_on_first': xlim centered on first molecule's centroid
        - 'center_on_each': no global xlim change (lets matplotlib autoscale)
    window_kms : half-window around centroid(s) if xlim_mode uses it.
    kwargs : passed through to make_central_spectrum_data (e.g. noskycoord=True)
    """
    if len(molecules) != 2:
        raise ValueError("molecules must be a 2-element iterable, e.g. ('HCO+','C18O')")

    mol1, mol2 = molecules

    s1, v1, title_piece = get_spectrum_data(source_name, mol1, type=type, **kwargs)
    s2, v2, _          = get_spectrum_data(source_name, mol2, type=type, **kwargs)

    pos1 = get_velocity_centroid(source_name, mol1, s1, v1)
    pos2 = get_velocity_centroid(source_name, mol2, s2, v2)

    # Optionally align centroids by shifting velocity axis of the "other" molecule
    if align_velocity_to is not None:
        if align_velocity_to == mol1:
            v2_plot = v2 - pos2 + pos1
            v1_plot = v1
            pos2_plot = pos1
            pos1_plot = pos1
        elif align_velocity_to == mol2:
            v1_plot = v1 - pos1 + pos2
            v2_plot = v2
            pos1_plot = pos2
            pos2_plot = pos2
        else:
            raise ValueError(f"align_velocity_to must be None, '{mol1}', or '{mol2}'")
    else:
        v1_plot, v2_plot = v1, v2
        pos1_plot, pos2_plot = pos1, pos2

    plt.figure()
    plt.xlabel("velocity (km/s)", size=22)
    plt.ylabel("Intensity (K)", size=22)
    plt.title(f"{source_name}",size=28)# for {mol1} and {mol2}")

    plt.plot(v1_plot, s1, "-", lw=3, label=mol1,color='C1')
    plt.plot(v2_plot, s2, "-", lw=3, label=mol2,color='C0')

    # Mark centroids (after any alignment)
    plt.axvline(pos1_plot, ls="--", lw=1,color='gray')
    plt.axvline(pos2_plot, ls="--", lw=1,color='gray')
    plt.ylim(-0.5,9.0)
    plt.tick_params(axis='both', direction='in',labelsize=20)
    plt.legend(fontsize=16)

    # x-limits handling
    if xlim_mode == "union":
        lo = min(pos1_plot - window_kms, pos2_plot - window_kms)
        hi = max(pos1_plot + window_kms, pos2_plot + window_kms)
        plt.xlim(lo, hi)
    elif xlim_mode == "center_on_first":
        plt.xlim(pos1_plot - window_kms, pos1_plot + window_kms)
    elif xlim_mode == "center_on_each":
        pass  # autoscale
    else:
        raise ValueError("xlim_mode must be 'union', 'center_on_first', or 'center_on_each'")

    if save:
        out = f"spectrum_{source_name}_{mol1}_{mol2}_{type}.png"
        plt.savefig(os.path.join("Figures/Spectra/both_molec/", out), bbox_inches="tight", dpi=300)

    if plot:
        plt.show()

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

import os

def mass_produce_spectral_plots(folder_fits, molecule=None, molecules=None):
    """
    Mass-produce spectral plots from sdf_and_fits/<source>/<source>_<molecule>.fits.

    Use either:
      - molecule="C18O"  (single molecule mode)
      - molecules=("HCO+","C18O") (two-molecule overlay mode)

    folder_fits: the parent folder containing <source> subfolders (e.g. "sdf_and_fits")
    """
    folder_list = sorted(next(os.walk(folder_fits))[1])
    print("Folders found:", folder_list)

    for src in folder_list:

        if src =='IRAS05256+3049' or src =='IRAS04369+2539' or src == 'IRAS03220+3035':
            print('I will skip ',src)
            continue

        try:
            # --- single molecule mode ---
            if molecule is not None:
                fn = f"{src}_{molecule}.fits"
                p = os.path.join(folder_fits, src, fn)

                if not os.path.isfile(p):
                    print(f"[{src}] missing {molecule}: {p} (skipping)")
                else:
                    plot_spectrum(src, molecule, type='central', save=True, plot=False)
                    plot_spectrum(src, molecule, type='fov',     save=True, plot=False)

            # --- two molecule overlay mode ---
            if molecules is not None:
                mol1, mol2 = molecules
                p1 = os.path.join(folder_fits, src, f"{src}_{mol1}.fits")
                p2 = os.path.join(folder_fits, src, f"{src}_{mol2}.fits")

                if not os.path.isfile(p1) or not os.path.isfile(p2):
                    missing = []
                    if not os.path.isfile(p1): missing.append(p1)
                    if not os.path.isfile(p2): missing.append(p2)
                    print(f"[{src}] missing one/both for overlay; skipping:\n  " + "\n  ".join(missing))
                else:
                    plot_spectrum_two_molecules(src, molecules=(mol1, mol2), type='central', save=True, plot=False)
                    # plot_spectrum_two_molecules(src, molecules=(mol1, mol2), type='fov',     save=True, plot=False)

        except IndexError as err:
            print(f"[{src}] spectral line not produced. Check plots. Error: {err}")

        except Exception as e:
            print(f"[{src}] unexpected error: {e}")

if __name__ == "__main__":

    source_name = 'T-Tauri'
    # source_name = 'Elia33'
    # molecule ='HCO+'
    molecule ='C18O'
    # distance = 130
    mass_produce_spectral_plots("sdf_and_fits", molecules=("HCO+", "C18O"))
