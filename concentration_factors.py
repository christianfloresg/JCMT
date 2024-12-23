from plot_generation_JCMT import *
import numbers
import math
from scipy import stats

def calculate_concentration_factor(source_name, molecule, n_sigma=3):
    '''
    Calculate concentration factors from molecular data as defined
    in van Kempen 2009 and Carney 2016
    :param source_name:
    :param molecule:
    :param n_sigma:
    :return:
    '''
    filename=source_name+'_'+molecule #'V347_Aur_HCO+'
    data_cube = DataAnalysis(os.path.join('sdf_and_fits',source_name), filename+'.fits')

    print('molecule ',data_cube.molecule)

    if 'HCO+' in data_cube.molecule:
        aperture_radius = 7.05

    elif data_cube.molecule == 'C18O':
        aperture_radius = 7.635

    beam_size = np.pi/(4*np.log(2)) * (2 * aperture_radius)**2
    peak_temp = peak_temperature_from_map(source_name, molecule)
    area , integrated_emission =area_and_emission_of_map_above_threshold(source_name, molecule, n_sigma,plot=False)
    concentration = 1 - beam_size / area * integrated_emission / peak_temp
    rounded_concentration = round(concentration,3)
    print(f"For {source_name}, the concentration factor is {rounded_concentration}")
    return concentration

def save_concentration_factors_to_file(folder_fits, molecule, save_filename):

    def save_to_file(save_filename, new_values):

        formatted_entry = (
            f"{new_values[0]:<20}"  # Source name in 20 bytes
            f"{new_values[1]:<20}"  # 
            f"{new_values[2]:<20}"  #
        )
        # Read the file if it exists, otherwise start with a header
        try:
            with open(save_filename, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            # If the file doesn't exist, start with a formatted header line
            header = (
                f"{'## SourceMame':<20}"
                f"{'c-factor_1sigma':<20}"
                f"{'c-factor_3sigma':<20}\n"
            )

            lines = [header]

        first_value = new_values[0]
        found = False

        # Check if the first value is already in the file and update the line if it exists
        for i, line in enumerate(lines):
            # Check if this line starts with the source name
            if line.startswith(f"{first_value:<20}"):
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

    for sources in folder_list:
        try:
            # Check if the necessary file exists before running the function
            filename = sources + '_' + molecule
            fits_file_path = os.path.join(folder_fits, sources, f"{filename}.fits")
            if not os.path.isfile(fits_file_path):
                print(f"No such file: {fits_file_path}. Skipping this folder.")
                continue  # Move to the next folder if the file doesn't exist

            concentration_1sigma = round(calculate_concentration_factor(sources, molecule, n_sigma=1),4)
            concentration_3sigma = round(calculate_concentration_factor(sources, molecule, n_sigma=3),4)

            new_values = [sources, concentration_1sigma, concentration_3sigma]

            save_to_file(save_filename,new_values)

        except IndexError as err:
            print(f"An error occurred: {err}")

        except Exception as e:
            print(f"An unexpected error occurred with {sources}: {e}")

def read_parameters(filename):
    our_data = np.genfromtxt(filename, skip_header=1, skip_footer=0, )
    # print Obs_data
    Temp_values = our_data[:, 1]
    temp_uncertainty = our_data[:, 2], our_data[:, 3]
    logg_values = our_data[:, 4]
    logg_uncertainty = our_data[:, 5], our_data[:, 6]

    vsini_values = our_data[:, 16]
    vsini_uncertainty = our_data[:, 17], our_data[:, 18]

    bfield_values = our_data[:, 13]
    bfield_uncertainty = our_data[:, 14], our_data[:, 15]

    ir_index_values = our_data[:, 19]

    HCO_data = our_data[:, 20]
    C18O_data = our_data[:, 21]

    # file_with_name = np.genfromtxt(filename, skip_header=1, skip_footer=0, dtype='string')
    # star_name = file_with_name[:, 0]
    star_name = our_data[:, 0]

    return Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
           vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name


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
        plt.savefig(os.path.join('Figures/concentration/', molecule+'_concentration_factor_3_parameters.png'), bbox_inches='tight', dpi=300)
    plt.show()


def plot_parameters(filename,molecule,save=False):
    Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
    vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name=    read_parameters(filename)

    if molecule=='HCO+':
        molecule_data = HCO_data
    elif molecule=='C18O':
        molecule_data =  C18O_data

    plt.figure(figsize=(8, 6))

    # plt.scatter(ir_index_values,molecule_data, edgecolor='k', s=100,c='C0')
    # plt.xlabel('Spectral Index', fontsize=14)
    # plt.ylabel('Concentration factor', fontsize=14)


    plt.scatter(logg_values,molecule_data, edgecolor='k', s=100,c='C1')
    plt.xlabel('Gravity', fontsize=14)
    plt.ylabel('Concentration factor', fontsize=14)

    # plt.title('Gravity vs Spectral Index', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    # plt.ylim(-2,1)

    # Show the plot
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule+'_concentration_factor_2_parameters_gravity.png'), bbox_inches='tight', dpi=300)
    plt.show()

def make_histograms(filename, parameter='gravity', molecule='HCO+',save=False):

    Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
    vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name=    read_parameters(filename)

    logg_values =logg_values/100.
    if molecule=='HCO+':
        # Check that all input arrays have the same length
        if len(logg_values) != len(ir_index_values) or len(logg_values) != len(HCO_data):
            raise ValueError("All input arrays must have the same length.")
        is_numeric = [not math.isnan(x) for x in HCO_data]
        is_nan = [math.isnan(x) for x in HCO_data]

        # molecular_data=HCO_data
        # min_val= -2

    elif molecule=='C18O':
        # Check that all input arrays have the same length
        if len(logg_values) != len(ir_index_values) or len(logg_values) != len(C18O_data):
            raise ValueError("All input arrays must have the same length.")

        is_numeric = [not math.isnan(x) for x in C18O_data]
        is_nan = [math.isnan(x) for x in C18O_data]
        # molecular_data=C18O_data
        # min_val= -1

    if parameter.lower() == 'gravity':
        my_parameter = logg_values
        bins = np.arange(2.8,4.0,0.2)
        x_label =' Gravity'
        x_leg,y_leg = 2.85, 7
    elif parameter.lower() == 'ir_index':
        my_parameter =ir_index_values
        bins = np.arange(-1.0,1.5,0.25)
        x_label ='Spectral index'
        x_leg,y_leg = 0.4, 5

    else:
        print('wrong parameter')

    res= stats.ttest_ind(my_parameter[is_numeric], my_parameter[is_nan], equal_var=False)
    print(res)

    mean_val_1 = round(np.nanmean(my_parameter[is_numeric]),3)
    std_val_1 = round(np.nanstd(my_parameter[is_numeric]),3)

    mean_val_2 = round(np.nanmean(my_parameter[is_nan]),3)
    std_val_2 = round(np.nanstd(my_parameter[is_nan]),3)


    plt.hist(my_parameter[is_numeric],bins,alpha=0.7, label='detections', edgecolor="black")
    plt.text(x=x_leg,y=y_leg,s=r'$\mu$ = '+str(mean_val_1) +r' $\sigma$ = '+str(std_val_1),size=12,color='C0',weight=600)
    plt.text(x=x_leg,y=y_leg-1,s=r'$\mu$ = '+str(mean_val_2) +r' $\sigma$ = '+str(std_val_2),size=12,color='C1',weight=600)
    plt.hist(my_parameter[is_nan],bins,alpha=0.7, label='non-detections', edgecolor="black")
    plt.xlabel(x_label,size=14)
    plt.legend(loc='upper left')
    plt.title(molecule)
    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule + '_histogram_'+parameter+'.png'),
                    bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":

    source_name = 'V347_Aur'
    # molecule ='HCO+'
    molecule ='C18O'

    # calculate_concentration_factor(source_name, molecule, n_sigma=1)
    # save_concentration_factors_to_file(folder_fits='sdf_and_fits', molecule=molecule,save_filename='concentrations_'+molecule+'.txt')
    # plot_parameters(filename='text_files/Class_I_for-JCMT-plots.txt',molecule=molecule,save=True)

    # plot_gravity_vs_spectral_index(filename='text_files/Class_I_for-JCMT-plots.txt', color_map='viridis',
    #                                molecule=molecule,save=True)

    make_histograms(filename='text_files/Class_I_for-JCMT-plots.txt', parameter='ir_index',
                                   molecule=molecule,save=True)