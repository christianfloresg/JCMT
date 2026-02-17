import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# import plot_alma_images as Data
import matplotlib.colors as colors
from astropy import units as u
import astropy.constants as constants
from data_cube_analysis import fit_gaussian_to_spectrum
from plot_generation_JCMT import make_averaged_spectrum_data, area_and_emission_of_map_above_threshold, make_central_spectrum_data
from data_cube_analysis import fit_gaussian_to_spectrum, gauss, find_nearest_index

from datetime import date,datetime

today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)

"""
Measure mass of the streamers using the equation
in Zapata et al. (2014). Also check Lopez-Vazquez et al. (2020). The mass is measured assuming LTE
and optically thin emission. The equations are taken from La fisica del medio 
Interestellar de Estelella & Anglada (1996).
"""
GG = constants.G.cgs.value
Msun = constants.M_sun.cgs.value  # .value
k_B = constants.k_B.cgs.value  # .value
h_p = constants.h.cgs.value
pc = constants.pc.cgs.value  # .value
cc = constants.c.cgs.value  # .value
m_H2 = 2 * constants.m_p.cgs.value  # .value
X_H2_to_CO = 1.e4
au_to_cm = 1.496e+13
years_to_seconds = 3.154e+7
Jy_to_cgs = u.Jy.cgs
Hz = u.Hz.cgs
Kelvin = u.K.cgs
steradian = u.sr.cgs


class MolecularPhysicalParameters:
    '''
    This class is used to compute physical parameters of a cloud of gas
    it can be used also for envelopes or streamers.
    In general when asked for brightness temperature, please take peak brightness temperature
    i.e., the peak of the emission at a given parcel of gas or mom8?
    '''

    def __init__(self, molecule_name, J, Be, nu,A_coeff):
        '''
        J is the upper level state of the transition
        Be must be given in Hertz - In splatalog is given in MHz
        nu is the rest frequency of the molecular emission in (Hz)
        The A Einstein Coefficient must be given in s^-1 - In splatalog is given as log10
        '''
        self.J = J
        self.Be =Be
        self.nu = nu
        self.A_coeff = A_coeff
        self.Be = Be # This is the molecular rotational constant

        print('you are calculating parameters for ')
        print(molecule_name+' transition ' + str(J) + '-> ' +str(J-1))


    def intensity(self, temp):
        '''
        Intensity Jv(T) expressed in units of temperature
        in the RJ approximation
        '''
        numerator = h_p / k_B * self.nu
        denominator = np.exp( h_p / k_B * self.nu / temp) - 1

        return numerator/denominator

    def excitation_temperature(self,brightness_temp):
        '''
        This excitation temperature is obtained assuming that the measured brightness temperature
        comes from optically thick emission and we are subtracting a background temperature of 2.7 K
        '''
        T_bg = self.intensity(2.7) # Notice that this is the T_bg for the optically thick molecule
        hv_k = h_p / k_B * self.nu
        denominator = np.log( 1+ hv_k/(brightness_temp+T_bg) )
        return hv_k/denominator

    def brightness_temperature(self, intensity):
        '''
        Intensity must be given in cgs
        Make sure the intensity is given divided by 
        pixel area e.g., by (field_size_in_arc*radian_to_arcsec)**2

        '''
        hv_k = h_p * nu / k_B
        denominator = np.log1p(2 * h_p * nu ** 3 / (intensity * cc ** 2))
        return hv_k / denominator

    def optical_depth(self,brightness_temp,T_ex):
        '''
        measurement of optical depth of a line
        assuming that the excitation temperature of the gas is a good reference
        for the temperature of this molecule too.
        '''
        T_bg = self.intensity(2.7) # This T_bg is for a molecule that is not necessarily optically thick

        return -np.log( 1 - brightness_temp/( self.intensity(T_ex) - T_bg) )

    def column_density(self,brightness_temp, Tex, dv_km=0.2):
        '''
        There are two flavors to calculate column density
        1) for un-resolved observations
        2) when a resolved map is given

        -Tex is the excitation temperature of the molecule in (K) obtained from an
        optically thick isotopologue, e.g., use the function defined in this Class

        1) for un-resolved observations
            -brightness temperature is the peak brightness temperature of the spectrum in (K)
            -Delta_v is approx the FWHM size in km/s of the temperature spectrum
            (which is then transformed to cm)
            This first approximation assumes that the area of the integral of optical depth
            can be calculated from the width of the spectrum multiplied by the peak (in optical depth space)

        2) when a resolved map is given
            -The integral of the optical depth profile over the velocity range must be provided
            -This is not an approximation, instead
        return: column density in cm**(-2)
        '''

        Delta_v = dv_km*1e5 # from km/s to cm/s
        optical_depth = self.optical_depth(brightness_temp,Tex)

        # This should literally be the integral of the optical depth calculated from the spectrum vs. velocity
        if type(brightness_temp)==float or type(brightness_temp)==int:
            integrated_optical_depth = optical_depth * Delta_v
            # integrated_optical_depth = 1.3 * Delta_v
            print('A float was given')
        else:
            integrated_optical_depth = np.sum(optical_depth)*Delta_v
            print('An array was given')
            # plt.plot(optical_depth)
            # plt.show()
        # integrated_optical_depth = 330

        N_n= integrated_optical_depth * 8 * np.pi*self.nu**3
        N_d =  cc**3 * self.A_coeff * (np.exp(h_p / k_B * self.nu/ Tex) -1 )
        Nj=N_n/N_d

        Z_part=Tex/self.Be * (h_p/k_B)**(-1) # This is an approximation of the partition function
        N_total = Nj/(2*self.J+1) * np.exp( (h_p/k_B)*self.Be * self.J*(self.J+1)/Tex ) * Z_part

        return N_total

    def h2_column_density(self,X_H2_to_molec,Ntotal):
        '''
        This is the column density of H2 based on the column density of the molecule
        e.g., for 13CO, X_H2_to_mole is 5e5 and must be multiplied to Ntot to get
        H2 column density

        X_H2_to_molec are the fractional abundances of H2 to the molecule used
        Ntotal is the total column density of the molecule of reference
        '''
        return X_H2_to_molec*Ntotal


    def h2_gas_mass(self,X_H2_to_molec,Ntotal,distance_pc,area_arcsec2):
        '''
        Here we calculate the gas mass based on the molecular weight of H2
        the H2 column density calculated in h2_column_density(), the distance
        to the source and the area of the region.

        m_H2 is the mass of molecular hydrogen
        '''
        h2_column = self.h2_column_density(X_H2_to_molec,Ntotal)

        # area = np.pi*(distance*radius*au_to_cm)**2
        area_cm2 =  area_arcsec2 * distance_pc**2 * au_to_cm**2 #(distance* 2*radius*au_to_cm)**2


        return m_H2*h2_column*area_cm2/Msun

    def Av_mag_from_H2_col_dens(self,H2_coldens):
        '''
        We can calculate the amount of extinction (more or less empiraically) from
        the column density of molecular hydrogen. Generall there is 1 mag of visual
        extinction per every 10^21 H2_molec/cm^2
        H2_coldens is given in cm^-2
        return: visual extinction in magnitudes
        '''
        return H2_coldens/(0.94e21)


def obtain_parameters_from_spectrum(source_name, molecule,type='FOV'):

    spectrum, velocity = make_averaged_spectrum_data(source_name, molecule)
    if type=='central':
        spectrum, velocity = make_central_spectrum_data(source_name, molecule, noskycoord=False)

    pos_fov, FHWM_fov, sigma_fov = fit_gaussian_to_spectrum(spectrum, velocity,
                                                            velo_range=[-30,30] ,plot=False,
                                                            source_name=source_name+'_FOV',molecule=molecule)

    broad_lower_idx= find_nearest_index(array=velocity,value=pos_fov+5*sigma_fov)
    broad_upper_idx= find_nearest_index(array=velocity, value=pos_fov-5*sigma_fov)


    shortened_vel=velocity[broad_lower_idx:broad_upper_idx]
    shortened_flux=spectrum[broad_lower_idx:broad_upper_idx]

    if broad_upper_idx<broad_lower_idx:
        shortened_vel = velocity[broad_upper_idx:broad_lower_idx]
        shortened_flux = spectrum[broad_upper_idx:broad_lower_idx]


    # plt.plot(shortened_vel, gauss(velocity_fov, H=0, A=1, x0=pos_fov, sigma=sigma_fov), '--r', label='fit')
    # plt.plot(shortened_vel, shortened_flux)
    # plt.title(source_name + ' ' + type)
    # plt.show()

    return shortened_flux, shortened_vel


def save_c18o_envelope_parameters_to_file(folder_fits, molecule, save_filename):

    def save_to_file(save_filename, new_values):

        formatted_entry = (
            f"{new_values[0]:<30}"  # Source name in 20 bytes
            f"{new_values[1]:<20}"  # 
            f"{new_values[2]:<30}"  #
            f"{new_values[3]:<30}"  #
            f"{new_values[4]:<30}"  #
            f"{new_values[5]:<30}"  #
            f"{new_values[6]:<30}"  #
            f"{new_values[7]:<30}"  #
            f"{new_values[8]:<30}"  #
            f"{new_values[9]:<30}"  #
            f"{new_values[10]:<30}"  #
            f"{new_values[11]:<30}"  #
        )
        # Read the file if it exists, otherwise start with a header
        try:
            with open(save_filename, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            # If the file doesn't exist, start with a formatted header line
            header = (
                f"{'## SourceMame':<30}"
                f"{'source_distance':<20}"
                f"{'C18O_col_dens_fov':<30}"
                f"{'H2_col_dens_fov':<30}"
                f"{'area_arcsec2':<30}"
                f"{'visual_mag_fov':<30}"
                f"{'H2_mass_fov':<30}"
                f"{'C18O_col_dens_center':<30}"
                f"{'H2_col_dens_center':<30}"
                f"{'beam_area':<30}"
                f"{'visual_mag_center':<30}"
                f"{'H2_mass_center':<30}\n"
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

    folder_list = sorted(next(os.walk(folder_fits))[1])  # List of subfolders
    print("Folders found:", folder_list)

    new_array = [source_name] + [0] * 11

    for sources in folder_list:
        # Check if the necessary file exists before running the function
        filename = sources + '_' + molecule
        fits_file_path = os.path.join(folder_fits, sources, f"{filename}.fits")
        if not os.path.isfile(fits_file_path):
            print(f"No such file: {fits_file_path}. Skipping this folder.")
            continue  # Move to the next folder if the file doesn't exist

        print('This is the length of the array',len(new_array), new_array)
        try:

            ### Here is some information about the source
            source_distance_pc = find_distance_of_stars(file_name='text_files/source_distances.txt', search_word=sources)
            source_distance_pc = float(source_distance_pc)

            CO_18_molec = MolecularPhysicalParameters(molecule_name='C18O', J=3, Be=Be_18, nu=nu_18, A_coeff=A_coeff_18)

            #### Here we use the whole size of the array.
            spectrum_fov, velocity_fov = obtain_parameters_from_spectrum(sources, molecule,type='FOV')

            area_of_array = area_and_emission_of_map_above_threshold(sources, molecule, n_sigma=1, plot=False, only_whole_area=True)
            column_density_fov = CO_18_molec.column_density(brightness_temp=spectrum_fov, Tex=Tex, dv_km=0.2)
            ### FOR the integration it is better to consider the integral of the fitted function!
            ### Or at least define a smaller region around the line to do the integration.
            print('C18O column density FOV')
            print(column_density_fov)

            h2_col_dens_fov = CO_18_molec.h2_column_density(X_C18O_to_H2, column_density_fov)
            print('H2 column density')
            print(h2_col_dens_fov)
            visual_extinction_fov = CO_18_molec.Av_mag_from_H2_col_dens(h2_col_dens_fov)
            print('mag of visual extinction = ',visual_extinction_fov)

            h2_mass_fov = CO_18_molec.h2_gas_mass(X_C18O_to_H2, column_density_fov, distance_pc=source_distance_pc,
                                              area_arcsec2=area_of_array)
            print('H2 mass ', h2_mass_fov)

            values_fov = [sources, source_distance_pc, column_density_fov, h2_col_dens_fov,
                          area_of_array,  visual_extinction_fov, h2_mass_fov]
            new_array = values_fov + new_array[len(values_fov):]

            print('This is the length of the array fov', len(new_array), new_array)


        except IndexError as err:
            print(f"An error occurred: {err}")

        except Exception as e:
            print(f"An unexpected error occurred with {sources}: {e}")


        try:
            #### Here we use only the central beam
            spectrum_center, velocity_center = obtain_parameters_from_spectrum(sources, molecule, type='central')

            aperture_radius_c18o = 7.635
            one_beam_area = (2*aperture_radius_c18o)**2*np.pi / (4*np.log(2))

            column_density_central = CO_18_molec.column_density(brightness_temp=spectrum_center, Tex=Tex, dv_km=0.2)

            print('C18O column density FOV')
            print(column_density_central)

            h2_col_dens_center = CO_18_molec.h2_column_density(X_C18O_to_H2, column_density_central)
            print('H2 column density')
            print(h2_col_dens_center)

            visual_extinction_center = CO_18_molec.Av_mag_from_H2_col_dens(h2_col_dens_center)
            print('mag of visual extinction = ',visual_extinction_center)

            h2_mass_center = CO_18_molec.h2_gas_mass(X_C18O_to_H2, column_density_central, distance_pc=source_distance_pc,
                                              area_arcsec2=one_beam_area)
            print('H2 mass ', h2_mass_center)

            values_central = [column_density_central,
                          h2_col_dens_center,one_beam_area,visual_extinction_center,h2_mass_center]

            fov_length_values = len(new_array) - len(values_central)
            new_array =  new_array[:fov_length_values] + values_central

            print('This is the length of the array central', len(new_array), new_array)

        except IndexError as err:
            print(f"An error occurred: {err}")

        except Exception as e:
            print(f"An unexpected error occurred with {sources}: {e}")

        save_to_file(save_filename,new_array)



def find_distance_of_stars(file_name, search_word):
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
                    print("you are trying to find the distnace for "+parts[0]+' which is ' + parts[1])
                    return parts[1] if len(parts) > 1 else ''

        # If no match is found
        print("This source is not in the list.")

    except FileNotFoundError:
        print(f"Error: The file {file_name} does not exist.")

    return None

if __name__ == "__main__":

    A_coeff_18 = 10**(-5.6632)
    Be_18 = 54891.42e6
    nu_18 = 219560.3568e6
    Tex = 20
    X_C18O_to_H2=3.5e6
    # source_distance=132
    source_name='OphIRS63'
    molecule='C18O'


    save_c18o_envelope_parameters_to_file(folder_fits='sdf_and_fits',
                                          molecule=molecule,
                                          save_filename='envelope_mass_' + today + '_' + hour_now + '_' + molecule + '.txt')

