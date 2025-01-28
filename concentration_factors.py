from plot_generation_JCMT import *
import numbers
import math
from scipy import stats
import astropy.units as u
import difflib


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
    area , integrated_emission = area_and_emission_of_map_above_threshold(source_name, molecule, n_sigma,plot=False)
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
    # our_data = np.genfromtxt(filename, skip_header=1, skip_footer=0, )

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line.startswith('#'):
                num_columns = len(line.split())
                continue

        # header = f.readline()#.strip()  # Read the first line
        # num_columns = len(header.split())  # Count the number of columns based on delimiter
    print('number of cols', num_columns)
    dtype = [('col1', 'U20')] + [(f'col{i+2}', 'f8') for i in range(num_columns - 1)]

    our_data = np.genfromtxt(filename, dtype=dtype,
                     delimiter=None, encoding='utf-8', comments='#')

    # Access specific columns
    # first_column = our_data['col1']  # String column
    # remaining_columns = [our_data[f'col{i+2}'] for i in range(num_columns - 1)]  # Numerical columns


    star_name = our_data['col1']

    # print Obs_data
    Temp_values = our_data['col2']
    temp_uncertainty = our_data['col3'], our_data['col4']
    logg_values = our_data['col5']
    logg_uncertainty = our_data['col6'], our_data['col7']

    vsini_values = our_data['col17']
    vsini_uncertainty = our_data['col18'], our_data['col19']

    bfield_values = our_data['col14']
    bfield_uncertainty = our_data['col15'], our_data['col16']

    ir_index_values = our_data['col20']

    HCO_data = our_data['col21']
    C18O_data = our_data['col22']


    return Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
           vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name


def read_map_parameters(filename):

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line.startswith('#'):
                num_columns = len(line.split())
                continue

        # header = f.readline()#.strip()  # Read the first line
        # num_columns = len(header.split())  # Count the number of columns based on delimiter
    print('number of cols', num_columns)
    dtype = [('col1', 'U20')] + [(f'col{i+2}', 'f8') for i in range(num_columns - 1)]


    our_data = np.genfromtxt(filename, dtype=dtype,
                     delimiter=None, encoding='utf-8', comments='#')

    star_name = our_data['col1']#our_data[:, 0]
    T_mb = our_data['col5']#our_data[:, 4]
    S_peak = our_data['col10']#our_data[:, 9]
    S_area = our_data['col11']#our_data[:, 10]

    return T_mb,S_peak,S_area, star_name


def read_ir_index_parameters(filename):

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line.startswith('#'):
                num_columns = len(line.split())
                continue

        # header = f.readline()#.strip()  # Read the first line
        # num_columns = len(header.split())  # Count the number of columns based on delimiter
    print('number of cols', num_columns)
    dtype = [('col1', 'U20')] + [(f'col{i+2}', 'f8') for i in range(num_columns - 1)]


    our_data = np.genfromtxt(filename, dtype=dtype,
                     delimiter=None, encoding='utf-8', comments='#')

    star_name = our_data['col1']#our_data[:, 0]
    ir_index = our_data['col2']#our_data[:, 4]
    corrected_ir_index = our_data['col3']#our_data[:, 9]


    return ir_index, corrected_ir_index, star_name

def plot_spectral_vs_map_parameters(spectrum_file,map_file,molecule='HCO+',save=False):
    '''
    Plot parameters related to the star against parameters related to the maps
    :param spectrum_file:
    :param map_file:
    :param molecule:
    :param save:
    :return:
    '''

    Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
    vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name =  read_parameters(spectrum_file)

    T_mb, S_peak, S_area,star_name_map = read_map_parameters(map_file)

    for ii in range(len(star_name)):
        for jj in range(len(star_name_map)):
            if are_names_approximate(star_name[ii], star_name_map[jj], threshold=0.9):
                print(star_name[ii], star_name_map[jj])
                # plt.scatter(logg_values[ii]/100., S_peak[jj], color='red', edgecolor='k', s=100)
                plt.scatter(logg_values[ii]/100., S_area[jj], color='orange', edgecolor='k', s=100)
                # plt.scatter(logg_values[ii]/100., T_mb[jj], color='blue', edgecolor='k', s=100)

    # Add labels and title
    plt.xlabel('Gravity', fontsize=14)
    # plt.ylabel('Integrated Intensity Main Beam (K km/s)', fontsize=14)
    plt.ylabel('Integrated Intensity FOV (K km/s)', fontsize=14)
    # plt.ylabel('Peak Temperature (K)', fontsize=14)

    plt.title(molecule, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join('Figures/stellar_vs_gas/', molecule+'_gravity_vs_integrated_intensity_FOV.png'), bbox_inches='tight', dpi=300)
    plt.show()


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


    plt.scatter(HCO_data,C18O_data, edgecolor='k', s=100,c='C1')
    plt.xlabel('HCO+ concentration', fontsize=14)
    plt.ylabel('C18O concentration', fontsize=14)

    # plt.title('Gravity vs Spectral Index', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    one_to_one = np.linspace(0,1,10)
    plt.plot(one_to_one,one_to_one,'k--',label='')
    plt.ylim(-0.2,1)
    plt.xlim(0,1)
    #
    # Show the plot
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule+'_comparing_both_concentrations.png'), bbox_inches='tight', dpi=300)
    plt.show()

def make_histogram_several_files(filename, map_file, ir_file, molecule='HCO+',save=False):

    Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
    vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name=    read_parameters(filename)

    logg_values =logg_values/100.

    T_mb, S_peak, S_area,star_name_map = read_map_parameters(map_file)

    # ir_index, ir_corrected_values, star_name_map = read_ir_index_parameters(ir_file)

    protostars_parameter =[]
    not_protostar_parameter=[]
    counted_source = []

    for ii in range(len(star_name)):
        for jj in range(len(star_name_map)):
            if are_names_approximate(star_name[ii], star_name_map[jj], threshold=0.9):
                counted_source.append(ii)
                if HCO_data[ii]>0.2 and not math.isnan(HCO_data[ii]) and S_peak[jj]>0.35:
                # if not math.isnan(HCO_data[ii]) and S_peak[jj] > 0.35:

                    protostars_parameter.append(logg_values[ii])
                    # print (ir_index[jj])
                    # protostars_parameter.append(ir_corrected_values[jj])
                    # protostars_parameter.append(ir_index[jj])

                    print('Detections',star_name[ii])
                    # print()

                    # print(star_name[ii])

                else:
                    not_protostar_parameter.append(logg_values[ii])
                    # not_protostar_parameter.append(ir_corrected_values[jj])
                    # not_protostar_parameter.append(ir_index[jj])

                    print('NOOOOO detections',star_name[ii])
                    # print()

        if ii not in counted_source:
            not_protostar_parameter.append(logg_values[ii])
            # not_protostar_parameter.append(ir_corrected_values[jj])
            # not_protostar_parameter.append(ir_index[jj])

    res= stats.ttest_ind(protostars_parameter, not_protostar_parameter, equal_var=False)
    print(res)
    x_label =' Gravity'
    bins = np.arange(2.8, 4.0, 0.2)

    # x_label ='ir index'
    # bins = np.arange(-1.25, 1.5, 0.25)

    x_leg, y_leg = 2.8, 6
    # x_leg, y_leg = 0.4, 5

    mean_val_1 = round(np.nanmean(protostars_parameter), 3)
    std_val_1 = round(np.nanstd(protostars_parameter), 3)

    mean_val_2 = round(np.nanmean(not_protostar_parameter), 3)
    std_val_2 = round(np.nanstd(not_protostar_parameter), 3)

    plt.text(x=x_leg, y=y_leg, s=r'$\mu$ = ' + str(mean_val_1) + r' $\sigma$ = ' + str(std_val_1), size=12, color='C0',
             weight=600)
    plt.text(x=x_leg, y=y_leg - 1, s=r'$\mu$ = ' + str(mean_val_2) + r' $\sigma$ = ' + str(std_val_2), size=12,
             color='C1', weight=600)


    # plt.hist(not_protostar_parameter,bins,alpha=0.7, label='non-detections', histtype='step',edgecolor="C1",lw=2)
    plt.hist(not_protostar_parameter,bins,alpha=0.5, label='non-detections', edgecolor="black")

    # plt.hist(protostars_parameter,bins,alpha=0.7, label='detections', edgecolor="C0", histtype='step',lw=2)
    plt.hist(protostars_parameter,bins,alpha=0.5, label='detections', edgecolor="black")


    plt.xlabel(x_label,size=14)
    plt.legend(loc='upper left')
    plt.title(molecule)

    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule + '_histogram_C02_and_emission_abov0p35.png'),
                    bbox_inches='tight', dpi=300)
    plt.show()


def make_histograms(filename, parameter='gravity', molecule='HCO+',save=False):

    Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
    vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name=    read_parameters(filename)

    logg_values =logg_values/100.


    if molecule=='HCO+':
        # Check that all input arrays have the same length
        if len(logg_values) != len(ir_index_values) or len(logg_values) != len(HCO_data):
            raise ValueError("All input arrays must have the same length.")
        is_numeric = [not math.isnan(x) and x>0.4 for x in HCO_data]
        is_nan = [math.isnan(x) or x<0.4 for x in HCO_data]
        # print()
        # molecular_data=HCO_data
        # min_val= -2

    elif molecule=='C18O':
        # Check that all input arrays have the same length
        if len(logg_values) != len(ir_index_values) or len(logg_values) != len(C18O_data):
            raise ValueError("All input arrays must have the same length.")

        is_numeric = [not math.isnan(x) and x>0.4 for x in C18O_data]
        is_nan = [math.isnan(x) or x<0.4 for x in C18O_data]
        # is_numeric = [not math.isnan(x) for x in C18O_data]
        # is_nan = [math.isnan(x) for x in C18O_data]
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

    plt.text(x=x_leg,y=y_leg,s=r'$\mu$ = '+str(mean_val_1) +r' $\sigma$ = '+str(std_val_1),size=12,color='C0',weight=600)
    plt.text(x=x_leg,y=y_leg-1,s=r'$\mu$ = '+str(mean_val_2) +r' $\sigma$ = '+str(std_val_2),size=12,color='C1',weight=600)

    plt.hist(my_parameter[is_numeric],bins,alpha=0.7, label='detections', edgecolor="black")
    plt.hist(my_parameter[is_nan],bins,alpha=0.7, label='non-detections', edgecolor="black")
    plt.xlabel(x_label,size=14)
    plt.legend(loc='upper left')
    plt.title(molecule)
    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule + '_histogram_C04'+parameter+'.png'),
                    bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":

    source_name = 'V347_Aur'
    molecule ='HCO+'
    # molecule ='C18O'

    # calculate_concentration_factor(source_name, molecule, n_sigma=1)
    # save_concentration_factors_to_file(folder_fits='sdf_and_fits', molecule=molecule,save_filename='concentrations_'+molecule+'.txt')
    # plot_parameters(filename='text_files/Class_I_for-JCMT-plots.txt',molecule=molecule,save=True)

    # plot_gravity_vs_spectral_index(filename='text_files/Class_I_for-JCMT-plots.txt', color_map='viridis',
    #                                molecule=molecule,save=True)

    # make_histograms(filename='text_files/Class_I_for-JCMT-plots.txt', parameter='ir_index',
    #                                molecule=molecule,save=True)

    # make_histograms(filename='text_files/Class_I_for-JCMT-plots-with_names-corrected.txt', parameter='gravity',
    #                                molecule=molecule,save=False)

    make_histogram_several_files(filename='text_files/Class_I_for-JCMT-plots-with_names-corrected.txt'
                                 , map_file='spectrum_parameters_HCO+.txt',ir_file='text_files/my_spectral_indices_new.txt',
                                 molecule='HCO+', save=True)

    # plot_spectral_vs_map_parameters(spectrum_file='text_files/Class_I_for-JCMT-plots-with_names-corrected.txt',
    #                                 map_file='spectrum_parameters_'+molecule+'.txt',molecule=molecule,save=True)