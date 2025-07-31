from plot_generation_JCMT import *
import numbers
import math
from scipy import stats
from dust_extinction.parameter_averages import CCM89, F99, M14
from dust_extinction.grain_models import WD01
import astropy.units as u
import difflib
import os
from datetime import date,datetime
today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)

def are_names_approximate(name1, name2, threshold=0.8):
    """
    Compare two names and return True if they are approximately the same,
    based on a similarity threshold.

    Args:
        name1 (str): The first name to compare.
        name2 (str): The second name to compare.
        threshold (float): The similarity threshold (default is 0.8).

    Returns:
        bool: True if the names are approximately the same, False otherwise.
    """
    # Calculate similarity ratio using difflib's SequenceMatcher
    similarity = difflib.SequenceMatcher(None, name1, name2).ratio()

    # Return True if similarity exceeds the threshold
    return similarity >= threshold



def extinction_corrected_flux(wavelength, uncorrected_flux, Av_mag):#, Rv=3.1):
    '''
    We assume simply that the K-band magnitude is ~10% of the visual magnitude
    The final equation is the definition of extinction A=-2.5np.ln(F/Fo) (ln = natural log)
    flux is given in ergs/s/cm^2/wavelength
    '''

    unit_wavelength = wavelength*u.um
    # unit_flux = flux * u.erg / (u.cm ** 2 * u.AA * u.s)
    unit_flux = uncorrected_flux * u.erg / (u.cm ** 2 * u.AA * u.s)

    # astropy.visualization.quantity_support()

    Ebv = Av_mag/5.5#Rv
    # ext = F99(Rv=Rv)
    ext= WD01('MWRV55')
    corrected_spectrum = unit_flux/ext.extinguish(unit_wavelength, Ebv=Ebv)

    # plt.semilogy(wavelength,uncorrected_flux)
    # plt.semilogy(wavelength,corrected_spectrum,label='de-redden')
    # plt.legend()
    # plt.show()
    return corrected_spectrum.value

def magnitude_to_flux(mag_kband,wavelength_microns):
    '''
    simple magnitude to flux conversion to check
    for consistency in the measure spectral flux
    f_r is the reference flux (e.g., from vega)
    m_r is the reference magnitude (e.g., from vega)
    '''


    if 2.0<wavelength_microns<2.5:

        f_r = 3.961e-11 # This is the vega zero point flux in ergs/s/cm^2/A
        m_k = 0.129 # This is the K-band maginute of Vega

        return f_r * 10 ** ((m_k - mag_kband) / 2.5)

    elif 20<wavelength_microns<25:
        f_r = 7.17*2.99792458E-05/(wavelength_microns)**2
        m_k = 0. # This is the 24 maginute of Vega

        return f_r * 10 ** ((m_k - mag_kband) / 2.5)


def jy_to_cgs_factor(wavelength_microns):
    '''
    :param wavelength_microns:
    :return:
    '''
    return 2.99792458E-05/(1e4*wavelength_microns)**2



def cgs_to_jy_factor(wavelength_microns):
    '''
    :param wavelength_microns:
    :return:
    '''
    return (1e4*wavelength_microns)**2/2.99792458E-05


def calculate_spectral_index(two_micron_flux, two_micron_wave, twenty_micron_flux,twenty_micron_wave,
                             flux_scale='cgs_micron'):
    '''
    Using the two and twenty-ish micron fluxes, compute the infrared spectral indices
    :param two_microon_mag:
    :param twenty_micron_mag:
    :return: slope
    '''

    if flux_scale=='jansky':
        near_ir_flux = np.log10(two_micron_flux*jy_to_cgs_factor(two_micron_wave)*two_micron_wave) #
        mid_ir_flux = np.log10(twenty_micron_flux*jy_to_cgs_factor(twenty_micron_wave)*twenty_micron_wave) #

    elif flux_sacle=='cgs_micron':
        near_ir_flux = np.log10(two_micron_flux*two_micron_wave) #
        mid_ir_flux = np.log10(twenty_micron_flux*twenty_micron_wave) #

    alpha = (mid_ir_flux - near_ir_flux)/ (np.log10(twenty_micron_wave) - np.log10(two_micron_wave))

    return alpha

def read_parameters(filename):
    '''
    Read parameters from file. The first column is a name
    the rest of the columns are numerical values
    :param filename: name of the file
    :return:
    '''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line.startswith('#'):
                num_columns = len(line.split())
                continue

    print('number of cols', num_columns)
    dtype = [('col1', 'U20')] + [(f'col{i+2}', 'f8') for i in range(num_columns - 1)]

    our_data = np.genfromtxt(filename, dtype=dtype,
                     delimiter=None, encoding='utf-8', comments='#')

    star_name = our_data['col1']

    two_mass_flux, two_mass_uncert = our_data['col2'], our_data['col3']
    mid_ir_flux, mid_ir_uncert = our_data['col4'],our_data['col5']
    wavelength_mid_ir = our_data['col6']
    Av_literature = our_data['col7']
    alpha_literature = our_data['col8']
    Tbol, Lbol = our_data['col9'],our_data['col10']
    Av_connelley = our_data['col11']


    return two_mass_flux, two_mass_uncert, mid_ir_flux, mid_ir_uncert, wavelength_mid_ir, Av_literature,\
           alpha_literature, Tbol, Lbol, Av_connelley, star_name


def plot_gravity_vs_spectral_index(filename, color_map='viridis', molecule='HCO+',save=False):
    """
    Plots gravity vs. spectral index with c_factor as the color of the data points.

    Args:
        gravity (array-like): Array of gravity values for the x-axis.
        spectral_index (array-like): Array of spectral index values for the y-axis.
        c_factor (array-like): Array of c_factor values for coloring the points.
        color_map (str): Colormap to use for the points (default: 'viridis').
    """

    Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
    vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name=    read_parameters(filename)

    if molecule=='HCO+':
        # Check that all input arrays have the same length
        if len(logg_values) != len(ir_index_values) or len(logg_values) != len(HCO_data):
            raise ValueError("All input arrays must have the same length.")
        is_numeric = [not math.isnan(x) for x in HCO_data]
        is_nan = [math.isnan(x) for x in HCO_data]

        molecular_data=HCO_data
        min_val= -2
    elif molecule=='C18O':
        # Check that all input arrays have the same length
        if len(logg_values) != len(ir_index_values) or len(logg_values) != len(C18O_data):
            raise ValueError("All input arrays must have the same length.")

        is_numeric = [not math.isnan(x) for x in C18O_data]
        is_nan = [math.isnan(x) for x in C18O_data]
        molecular_data=C18O_data
        min_val= -1

    # Create the scatter plot
    plt.figure(figsize=(8, 6))

    # Use c_factor as colors
    scatter = plt.scatter(logg_values[is_numeric], ir_index_values[is_numeric], c=molecular_data[is_numeric],
                          cmap=color_map, edgecolor='k', s=100,vmin=min_val)
    # Add colorbar
    cbar = plt.colorbar(scatter)

    if molecule=='HCO+':
        cbar.set_label('HCO+ Concentration Factor', fontsize=12)
    elif molecule=='C18O':
        cbar.set_label('C18O Concentration Factor', fontsize=12)

    # Use black for non-numerical c_factor
    plt.scatter(logg_values[is_nan], ir_index_values[is_nan], color='red', edgecolor='k', s=100)

    # Add labels and title
    plt.xlabel('Gravity', fontsize=14)
    plt.ylabel('Spectral Index', fontsize=14)
    plt.title('Gravity vs Spectral Index', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule+'_concentration_factor_3_parameters.png'),
                    bbox_inches='tight', dpi=300)
    plt.show()


def save_spectral_indices_to_file(save_filename):

    def save_to_file(save_filename, new_values):

        formatted_entry = (
            f"{new_values[0]:<30}"  # Source name in 20 bytes
            f"{new_values[1]:<30}"  # 
            f"{new_values[2]:<30}"  #
            f"{new_values[3]:<30}"  #
            f"{new_values[4]:<30}"  #
            f"{new_values[5]:<30}"  #
        )
        # Read the file if it exists, otherwise start with a header
        try:
            with open(save_filename, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            # If the file doesn't exist, start with a formatted header line
            header = (
                f"{'## SourceMame':<30}"
                f"{'spectral index':<30}"
                f"{'Av literature':<30}"
                f"{'corr. spectral index lit':<30}"
                f"{'Av Connelley':<30}"
                f"{'corr. spectral index Con':<30}\n"

            )

            lines = [header]

        first_value = new_values[0]
        found = False

        # Check if the first value is already in the file and update the line if it exists
        for i, line in enumerate(lines):
            # Check if this line starts with the source name
            if line.startswith(f"{first_value:<30}"):
                lines[i] = formatted_entry + "\n"
                found = True
                break

        # If the source name was not found, append the new formatted entry
        if not found:
            lines.append(formatted_entry + "\n")

        # Write the updated content back to the file
        with open(save_filename, 'w') as file:
            file.writelines(lines)

    two_mass_flux, two_mass_uncert, mid_ir_flux, mid_ir_uncert, wavelength_mid_ir, Av_literature, \
    alpha_literature, Tbol, Lbol, Av_connelley, star_name  = read_parameters('text_files/SED_parameters.txt')


    for ii in range(len(star_name)):

        spectral_indices = calculate_spectral_index(two_micron_flux=two_mass_flux[ii] * 1e-3, two_micron_wave=2.17,
                                                    twenty_micron_flux=mid_ir_flux[ii] * 1e-3,
                                                    twenty_micron_wave=wavelength_mid_ir[ii], flux_scale='jansky')
        # print(type(Av_connelley[ii]),Av_connelley[ii])

        if isinstance( Av_connelley[ii], np.float64 ) and not np.isnan(Av_connelley[ii]):
            print('using connelley',Av_connelley[ii])
            connelley_near_ir_corrected_flux = extinction_corrected_flux(wavelength=2.17,
                                                               uncorrected_flux=two_mass_flux[ii], Av_mag=Av_connelley[ii])

            connelley_twenty_micron_corrected_flux = extinction_corrected_flux(wavelength=wavelength_mid_ir[ii],
                                                                     uncorrected_flux=mid_ir_flux[ii],  Av_mag=Av_connelley[ii])

        else:
            connelley_near_ir_corrected_flux = extinction_corrected_flux(wavelength=2.17,
                                                               uncorrected_flux=two_mass_flux[ii], Av_mag=0)
            connelley_twenty_micron_corrected_flux = extinction_corrected_flux(wavelength=wavelength_mid_ir[ii],
                                                                     uncorrected_flux=mid_ir_flux[ii],  Av_mag=0)


        if isinstance( Av_literature[ii], np.float64 )  and not np.isnan(Av_literature[ii]):
            print('using literature')
            literature_near_ir_corrected_flux = extinction_corrected_flux(wavelength=2.17,
                                                               uncorrected_flux=two_mass_flux[ii], Av_mag=Av_literature[ii])
            literature_twenty_micron_corrected_flux = extinction_corrected_flux(wavelength=wavelength_mid_ir[ii],
                                                                     uncorrected_flux=mid_ir_flux[ii],  Av_mag=Av_literature[ii])
        else:
            literature_near_ir_corrected_flux = extinction_corrected_flux(wavelength=2.17,
                                                               uncorrected_flux=two_mass_flux[ii], Av_mag=0)
            literature_twenty_micron_corrected_flux = extinction_corrected_flux(wavelength=wavelength_mid_ir[ii],
                                                                     uncorrected_flux=mid_ir_flux[ii],  Av_mag=0)

        connelley_corrected_spectral_indices = calculate_spectral_index(two_micron_flux=connelley_near_ir_corrected_flux * 1e-3, two_micron_wave=2.17,
                                                    twenty_micron_flux=connelley_twenty_micron_corrected_flux * 1e-3,
                                                    twenty_micron_wave=wavelength_mid_ir[ii], flux_scale='jansky')

        literature_corrected_spectral_indices = calculate_spectral_index(two_micron_flux=literature_near_ir_corrected_flux * 1e-3, two_micron_wave=2.17,
                                                    twenty_micron_flux=literature_twenty_micron_corrected_flux * 1e-3,
                                                    twenty_micron_wave=wavelength_mid_ir[ii], flux_scale='jansky')

        save_to_file(os.path.join('text_files',save_filename),
                     [star_name[ii],round(spectral_indices,3),Av_literature[ii],
                      round(literature_corrected_spectral_indices,3),Av_connelley[ii]
                     , round(connelley_corrected_spectral_indices, 3)])


if __name__ == "__main__":

    source_name = 'V347_Aur'
    # molecule ='HCO+'
    molecule ='C18O'

    # m=magnitude_to_flux(mag_kband=5, wavelength_microns=2.16)
    # n=cgs_to_jy_factor(wavelength_microns=2.16)
    # print(m*n)
    save_spectral_indices_to_file(save_filename='my_spectral_indices_'+today+'.txt')

    # a=calculate_spectral_index(two_micron_flux=5000, two_micron_wave=2.16, twenty_micron_flux=63300, twenty_micron_wave=22.1,
    #                          flux_scale='jansky')
    # print('spectral index',a)
    # wavelength= 2.17
    # flux = 250e-3*jy_to_cgs_factor(wavelength_microns=wavelength)
    # print (cgs_to_jy_factor(wavelength_microns=wavelength)*flux)
    #
    # corrected_flux = extinction_corrected_flux(wavelength=wavelength, uncorrected_flux=flux, Av_mag=7.9)
    # print('corrected flux_erg = ',corrected_flux)
    # print('corrected flux_Jy = ',corrected_flux*cgs_to_jy_factor(wavelength_microns=wavelength)*1e3)
