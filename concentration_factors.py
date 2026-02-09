from plot_generation_JCMT import *
import numbers
import math
from scipy import stats
import astropy.units as u
import difflib
from datetime import date,datetime
today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)


"""
to use run
conda activate /Users/christianflores/anaconda3/envs/astropy

"""

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

def calculate_concentration_factor(source_name, molecule,
                                   n_sigma=3, use_skycoord=True,
                                   gaussian_fit=False,plot=False,
                                   moment_maps_folder='moment_maps',
                                   aperture=None):
    '''
    Calculate concentration factors from molecular data as defined
    in van Kempen 2009 and Carney 2016
    :param source_name:
    :param molecule:
    :param n_sigma:
    :param use_skycoord: True  For some sources obtained in SCAN mode, the coordinates  are wrong and we
    placed the source at the center - for these please set to FALSE!
    :return:
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

    beam_size = np.pi/(4*np.log(2)) * (2 * aperture_radius)**2
    peak_integrated_emission = peak_integrated_emission_from_map(source_name, molecule,
                                                                 use_skycoord=use_skycoord,
                                                                 moment_maps_folder= moment_maps_folder) ### This one requires object information.

    if gaussian_fit:
        area, integrated_emission = area_and_emission_via_gaussian(source_name, molecule,
                                                                   save_fig=plot, diagnostics=True,
                                                                   moment_maps_folder= moment_maps_folder)

    else:
        area , integrated_emission = area_and_emission_of_map_above_threshold(source_name, molecule, n_sigma, plot=plot,
                                                                              moment_maps_folder= moment_maps_folder)


    concentration = 1 - beam_size / area * integrated_emission / peak_integrated_emission
    rounded_concentration = round(concentration,3)
    print('beam_size',beam_size)
    print('peak_integrated_emission',peak_integrated_emission)
    print('area, integrated_emission',area, integrated_emission)
    print(f"For {source_name}, the concentration factor is {rounded_concentration}")
    return concentration


def save_concentration_factors_to_file(folder_fits, molecule, save_filename):

    def save_to_file(save_filename, new_values):

        formatted_entry = (
            f"{new_values[0]:<20}"  # Source name in 20 bytes
            f"{new_values[1]:<20}"  # 
            f"{new_values[2]:<20}"  #
            f"{new_values[3]:<20}"  #
            f"{new_values[4]:<20}"  #

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
                f"{'c-factor_2sigma':<20}"
                f"{'c-factor_3sigma':<20}"
                f"{'c-factor_gaussian':<20}\n"
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

            concentration_1sigma = round(calculate_concentration_factor(sources, molecule,
                                                                        n_sigma=1, plot=False, gaussian_fit=False),4)
            concentration_2sigma = round(calculate_concentration_factor(sources, molecule,
                                                                        n_sigma=2, plot=False, gaussian_fit=False),4)
            concentration_3sigma = round(calculate_concentration_factor(sources, molecule,
                                                                        n_sigma=3, plot=False, gaussian_fit=False),4)
            concentration_gaussian = round(calculate_concentration_factor(sources, molecule,
                                                                          plot=True, gaussian_fit=True),4)

            new_values = [sources, concentration_1sigma,concentration_2sigma, concentration_3sigma,concentration_gaussian]

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

def read_envelope_mass_parameters(filename):

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
    H2_coldense_fov = our_data['col4']#our_data[:, 4]
    H2_coldense_central = our_data['col9']#our_data[:, 9]

    return star_name, H2_coldense_fov, H2_coldense_central


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
    S_peak_uncert = our_data['col11']#our_data[:, 10]

    return T_mb,S_peak,S_peak_uncert, star_name


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
    corrected_ir_index = our_data['col4']#our_data[:, 9]


    return ir_index, corrected_ir_index, star_name

def read_c_factor_parameters(filename):
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
    c_factor_1_sigma = our_data['col2']#our_data[:, 4]
    c_factor_2_sigma = our_data['col3']#our_data[:, 9]
    c_factor_3_sigma = our_data['col4']#our_data[:, 9]
    try:
        c_factor_gaussian = our_data['col5']#our_data[:, 9]
        return star_name, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma, c_factor_gaussian
    except:
        return star_name, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma


def plot_spectral_vs_map_parameters(spectrum_file,spectral_map_file, color_map='gist_rainbow',molecule='HCO+',sigma_threshold=2,save=False):
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

    T_mb, S_beam, S_beam_uncert, star_spectral_map = read_map_parameters(spectral_map_file)

    S_beam = np.array(S_beam)

    if molecule=='HCO+':
        star_name_map, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma,\
        c_factor_gaussian = read_c_factor_parameters('text_files/concentrations_2025-08-12_22_HCO+.txt')

    elif molecule=='C18O':
        star_name_map, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma, \
        c_factor_gaussian = read_c_factor_parameters('text_files/concentrations_2025-08-12_23_C18O.txt')

    else:
        ValueError('only two values are possible C18O or HCO+')


    if sigma_threshold ==1:
        c_factor_n_sigma = c_factor_1_sigma
    elif sigma_threshold==2:
        c_factor_n_sigma = c_factor_2_sigma
    elif sigma_threshold==3:
        c_factor_n_sigma = c_factor_3_sigma
    else:
        c_factor_n_sigma = c_factor_gaussian
        # raise ValueError("Number must be 1, 2, or 3 sigmas.")

    save_name, save_temp, save_logg, save_logg_uncert_h,\
    save_logg_uncert_l, save_IR_index, save_this_c_factor, save_S_beam, save_S_beam_uncert= [],[],[],[],[],[],[],[],[]

    #### get the spectral parameters associated to the sources of the main molecule
    for ii in range(len(star_name)):
        c_factor_aux=0.0
        for jj in range(len(star_name_map)):
            if are_names_approximate(star_name[ii], star_name_map[jj], threshold=0.9):
                # print(star_name[ii], star_name_map[jj], c_factor_n_sigma[jj],logg_values[ii])
                # print(ii)

                if math.isnan(c_factor_n_sigma[jj]) or c_factor_n_sigma[jj] == float('-inf'):
                    c_factor_aux = 0.0
                else:
                    c_factor_aux = c_factor_n_sigma[jj]

                continue
        save_name = np.append(save_name, star_name[ii])
        save_temp = np.append(save_temp, Temp_values[ii])
        save_logg = np.append(save_logg, logg_values[ii])
        save_logg_uncert_h = np.append(save_logg_uncert_h, logg_uncertainty[0][ii])
        save_logg_uncert_l = np.append(save_logg_uncert_l, logg_uncertainty[1][ii])

        save_this_c_factor = np.append(save_this_c_factor, c_factor_aux)
        save_IR_index = np.append(save_IR_index,ir_index_values[ii])


    #### get the integrated intensities associated with the sources
    for ii in range(len(save_name)):

        if save_this_c_factor[ii] == 0.0:
            save_S_beam = np.append(save_S_beam,0)
            print(ii, save_name[ii])

        else:
            for jj in range(len(star_spectral_map)):

                if are_names_approximate(save_name[ii], star_spectral_map[jj], threshold=0.9):
                    print(ii,save_name[ii],star_spectral_map[jj], S_beam[jj])

                    save_S_beam = np.append(save_S_beam,S_beam[jj])
                    save_S_beam_uncert = np.append(S_beam_uncert,S_beam_uncert[jj])
                    continue


    # Separate the points lower than X
    save_temp = np.array(save_temp)
    save_logg = np.array(save_logg)
    # save_logg_uncert = np.array(save_logg_uncert)
    save_IR_index = np.array(save_IR_index)
    save_this_c_factor = np.array(save_this_c_factor)

    mask_low = save_this_c_factor < -0.2
    mask_good = ~mask_low

    print(save_logg[mask_low])
    # # Scatter for 'good' c_factors

    # print(save_logg_uncert_l,save_logg_uncert_h)

    plt.figure(figsize=(7, 6))


    # if use_other_c_cbar:
    scatter = plt.scatter(save_logg[mask_good]/1.e2, save_this_c_factor[mask_good],
                          c=save_S_beam[mask_good], cmap=color_map, edgecolor='k', s=150, vmin=0.0,vmax=2)

    plt.scatter(save_logg[mask_low] / 1.e2, np.full_like(save_logg[mask_low], -0.2),
                c='red', edgecolor='k', s=150, marker='v')


    # Add error bars on top of the scatter plot
    _, caps, bars = plt.errorbar(x=save_logg[mask_good]/1.e2, y=save_this_c_factor[mask_good],
                                          xerr=[save_logg_uncert_l[mask_good]/1.e2,save_logg_uncert_h[mask_good]/1.e2],
                                fmt='none', ecolor='gray', capsize=4, alpha=0.5, zorder=1)

    _, caps, bars = plt.errorbar(x=save_logg[mask_low]/1.e2, y=np.full_like(save_logg[mask_low], -0.2),
                                          xerr=[save_logg_uncert_l[mask_low]/1.e2,save_logg_uncert_h[mask_low]/1.e2],
                                fmt='none', ecolor='gray', capsize=4, alpha=0.5, zorder=1)


    # Add labels and title
    plt.xlabel('log(g)', fontsize=18)
    # plt.ylabel('Integrated Intensity Main Beam (K km/s)', fontsize=14)
    plt.ylabel('C-factor ' + molecule, fontsize=18)
    # plt.ylabel('Peak Temperature (K)', fontsize=14)
    cbar = plt.colorbar(scatter,cmap='gist_rainbow', location='top', orientation='horizontal', format='%.1f',fraction=0.1,pad=0.01)# fraction=0.048, pad=0.04)

    cbar.set_label(label=molecule + ' Integrated Intensity (K km/s) ', size=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    # cbar = plt.colorbar(scatter,cmap='gist_rainbow')


    plt.ylim(-0.25,1.0)
    plt.xlim(2.65,4.15)

    # plt.title(molecule, fontsize=16)
    # plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join('Figures/concentration/',
                                 molecule+'_gravity_vs_c_and_integrated_intensity_'+today+'.png'),
                    bbox_inches='tight', dpi=300)
    plt.show()


def plot_stellar_params_and_coldense(spectrum_file,map_file,molecule='HCO+',save=False, color_map='gist_rainbow'):
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

    star_name_map, H2_coldense_fov, H2_coldense_central = read_envelope_mass_parameters(map_file)

    save_temp,save_logg,save_H2_coldense_central=[],[],[]

    for ii in range(len(star_name)):
        for jj in range(len(star_name_map)):
            if are_names_approximate(star_name[ii], star_name_map[jj], threshold=0.9):
                print(star_name[ii], star_name_map[jj],H2_coldense_central[jj])
                # plt.scatter(logg_values[ii]/100., S_peak[jj], color='red', edgecolor='k', s=100)
                # plt.scatter(logg_values[ii]/100., S_area[jj], color='orange', edgecolor='k', s=100)
                # plt.scatter(logg_values[ii]/100., T_mb[jj], color='blue', edgecolor='k', s=100)
                save_temp.append(Temp_values[ii])
                save_logg.append(logg_values[ii])
                save_H2_coldense_central.append(H2_coldense_central[jj])

    # scatter = plt.scatter(save_temp, save_logg, c=np.log10(np.asarray(save_H2_coldense_central)),
    #                       cmap=color_map, edgecolor='k', s=130,vmin=21)

    scatter = plt.scatter(save_temp, save_logg, c=np.log10(np.asarray(save_H2_coldense_central)),
                          cmap=color_map, edgecolor='k', s=130,vmin=20.5)
    # Add labels and title
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('logg', fontsize=14)
    cbar = plt.colorbar(scatter,cmap='gist_rainbow')
    plt.xlim(4300,2900)
    plt.ylim(400,270)

    # plt.title(molecule, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule+'_gravity_vs_integrated_intensity_FOV.png'), bbox_inches='tight', dpi=300)
    plt.show()


def plot_stellar_params_and_c_factors(spectrum_file,sigma_threshold='2',molecule='HCO+',save=False, use_other_c_cbar=False,color_map='gist_rainbow'):
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

    if molecule=='HCO+':
        # star_name_map, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma = read_c_factor_parameters('concentrations_2025-06-02_10_HCO+.txt')
        star_name_map, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma,\
        c_factor_gaussian = read_c_factor_parameters('text_files/concentrations_2025-08-12_22_HCO+.txt')
        star_name_map_other, c_factor_1_sigma_other, c_factor_2_sigma_other, c_factor_3_sigma_other,\
        c_factor_gaussian_other = read_c_factor_parameters('text_files/concentrations_2025-08-12_23_C18O.txt')
        other_molecule = 'C18O'

    elif molecule=='C18O':
        star_name_map, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma, \
        c_factor_gaussian = read_c_factor_parameters('text_files/concentrations_2025-08-12_23_C18O.txt')
        star_name_map_other, c_factor_1_sigma_other, c_factor_2_sigma_other, c_factor_3_sigma_other,\
        c_factor_gaussian_other = read_c_factor_parameters('text_files/concentrations_2025-08-12_22_HCO+.txt')
        other_molecule = 'HCO+'
    else:
        ValueError('only two values are possible C18O or HCO+')


    if sigma_threshold ==1:
        c_factor_n_sigma = c_factor_1_sigma
        c_factor_n_sigma_other = c_factor_1_sigma_other
    elif sigma_threshold==2:
        c_factor_n_sigma = c_factor_2_sigma
        c_factor_n_sigma_other = c_factor_2_sigma_other
    elif sigma_threshold==3:
        c_factor_n_sigma = c_factor_3_sigma
        c_factor_n_sigma_other = c_factor_2_sigma_other
    else:
        c_factor_n_sigma = c_factor_gaussian
        # raise ValueError("Number must be 1, 2, or 3 sigmas.")
        c_factor_n_sigma_other = c_factor_gaussian_other

    save_name, save_temp, save_logg, save_logg_uncert_h,save_logg_uncert_l, save_IR_index, save_this_c_factor, save_other_c_factor=[],[],[],[],[],[],[],[]

    #### get the spectral parameters associated to the sources of the main molecule
    for ii in range(len(star_name)):
        c_factor_aux=0.0
        for jj in range(len(star_name_map)):
            if are_names_approximate(star_name[ii], star_name_map[jj], threshold=0.9):
                print(star_name[ii], star_name_map[jj], c_factor_n_sigma[jj],logg_values[ii])
                if math.isnan(c_factor_n_sigma[jj]) or c_factor_n_sigma[jj] == float('-inf'):
                    c_factor_aux = 0.0
                else:
                    c_factor_aux = c_factor_n_sigma[jj]

                continue
        save_name = np.append(save_name, star_name[ii])
        save_temp = np.append(save_temp, Temp_values[ii])
        save_logg = np.append(save_logg, logg_values[ii])
        save_logg_uncert_h = np.append(save_logg_uncert_h, logg_uncertainty[0][ii])
        save_logg_uncert_l = np.append(save_logg_uncert_l, logg_uncertainty[1][ii])

        save_this_c_factor = np.append(save_this_c_factor, c_factor_aux)
        save_IR_index = np.append(save_IR_index,ir_index_values[ii])



    #### get the spectral parameters associated to the sources of the main molecule
    for ii in range(len(save_name)):
        c_factor_aux=0.0
        for jj in range(len(star_name_map_other)):
            if are_names_approximate(save_name[ii], star_name_map_other[jj], threshold=0.9):
                print(save_name[ii], star_name_map_other[jj], c_factor_n_sigma_other[jj])
                if math.isnan(c_factor_n_sigma_other[jj]) or c_factor_n_sigma_other[jj] == float('-inf'):
                    c_factor_aux = 0.0
                else:
                    c_factor_aux = c_factor_n_sigma_other[jj]

                # print(c_factor_aux,type(c_factor_aux))
                continue

        save_other_c_factor = np.append(save_other_c_factor, c_factor_aux)

    # Separate the points lower than X
    save_temp = np.array(save_temp)
    save_logg = np.array(save_logg)
    # save_logg_uncert = np.array(save_logg_uncert)
    save_IR_index = np.array(save_IR_index)
    save_this_c_factor = np.array(save_this_c_factor)
    save_other_c_factor = np.array(save_other_c_factor)


    mask_low = save_this_c_factor < -0.2
    mask_good = ~mask_low

    # # Scatter for 'good' c_factors

    print(save_logg_uncert_l,save_logg_uncert_h)

    if use_other_c_cbar:
        scatter = plt.scatter(save_logg[mask_good]/1.e2, save_this_c_factor[mask_good],
                              c=save_other_c_factor[mask_good], cmap=color_map, edgecolor='k', s=100, vmin=0.2)

        plt.scatter(save_logg[mask_low] / 1.e2, np.full_like(save_logg[mask_low], -0.2),
                    c=save_other_c_factor[mask_low], cmap=color_map, edgecolor='k', s=100, marker='v', vmin=0.0)


    # Add error bars on top of the scatter plot
        _, caps, bars = plt.errorbar(x=save_logg[mask_good]/1.e2, y=save_this_c_factor[mask_good],
                                              xerr=[save_logg_uncert_l[mask_good]/1.e2,save_logg_uncert_h[mask_good]/1.e2],
                                    fmt='none', ecolor='gray', capsize=4, alpha=0.5, zorder=1)

        _, caps, bars = plt.errorbar(x=save_logg[mask_low]/1.e2, y=np.full_like(save_logg[mask_low], -0.2),
                                              xerr=[save_logg_uncert_l[mask_low]/1.e2,save_logg_uncert_h[mask_low]/1.e2],
                                    fmt='none', ecolor='gray', capsize=4, alpha=0.5, zorder=1)

    else:
        scatter = plt.scatter(save_logg[mask_good]/1.e2, save_this_c_factor[mask_good],
                          c=save_this_c_factor[mask_good], cmap=color_map, edgecolor='k', s=100, vmin=0.0)

        # scatter = plt.scatter(save_temp, save_logg/1.e2,
        #                   c=save_this_c_factor, cmap=color_map, edgecolor='k', s=100, vmin=0.0)
        # plt.xlim(4300,2900)
        # plt.ylim(4.00,2.70)
    # plt.xlim(2.65,4.15)


    # # Scatter for 'low' c_factors as down arrows at c_factor=-0.3
    # Use marker='v' for down arrow. We'll plot at y=-0.3, but color by the *real* c_factor value
    # plt.scatter(save_logg[mask_low]/1.e2, np.full_like(save_logg[mask_low], -0.2),
    #             c='red', edgecolor='k', s=100, marker='v')


    # Scatter for 'good' c_factors
    # scatter = plt.scatter(save_temp,save_logg/1.e2,
    #                       c=save_this_c_factor, cmap=color_map, edgecolor='k', s=100, vmin=0.1,vmax=0.8)
    # print(save_IR_index,)

    #####################################################################
    #### plot the names of the sources with C>0.2
    # mask_name = save_this_c_factor > 0.2
    #
    # for ii in range(len(save_name[mask_name])):
    #     plt.text(x=save_logg[mask_name][ii]/1.e2, y=save_this_c_factor[mask_name][ii], s=save_name[mask_name][ii], size=12,
    #              color='k')

    # Add labels and title

    # plt.xlabel('Temperature (K)', fontsize=14)
    # plt.ylabel('logg', fontsize=14)
    # cbar = plt.colorbar(scatter,cmap='gist_rainbow')
    # plt.xlim(4300,2900)
    # plt.ylim(4.00,2.70)

    # Add labels and title
    # plt.ylabel('HCO+', fontsize=14)
    plt.xlabel('logg', fontsize=14)
    # cbar = plt.colorbar(scatter,cmap='gist_rainbow')
    # plt.ylim(-0.3,1.0)

    # Add labels and title
    plt.ylabel(molecule, fontsize=14)
    # plt.ylabel('HCO+', fontsize=14)

    # plt.xlabel('Corrected alpha index', fontsize=14)
    # plt.ylim(-0.3,1.0)

    # plt.title(str(sigma_threshold)+' sigma threshold')
    # scatter = plt.scatter(save_IR_index,save_this_c_factor,
    #                       c=save_this_c_factor, cmap=color_map, edgecolor='k', s=100, vmin=0.1,vmax=0.8)

    # plt.scatter(save_IR_index[mask_low], np.full_like(save_IR_index[mask_low], -0.2),
    #             c='red', edgecolor='k', s=100, marker='v')

    # plt.axvline(x=-0.3,linestyle='--')
    # plt.axvline(x=0.3,linestyle='--')

    cbar = plt.colorbar(scatter,cmap='gist_rainbow')

    cbar.set_label(label='C-factor ' + molecule, size=14)

    if use_other_c_cbar:
        cbar.set_label(label='C-factor ' + other_molecule, size=14)


    # plt.title(molecule, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join('Figures/concentration/', 'logg_vs_'+molecule+str(sigma_threshold)+'_'+today+'.png')
                    , bbox_inches='tight', dpi=300)
    plt.show()


def fit_polinomials(HCO_1, C18O_1):

    HCO_1 = np.array(HCO_1)
    C18O_1 = np.array(C18O_1)

    # Mask to include only positive HCO+ values
    mask_1 = (HCO_1 > 0) #& (C18O_1 > 0) #HCO_1 > 0

    # Fit first-order polynomials (linear fits)
    fit_1 = np.polyfit(HCO_1[mask_1], C18O_1[mask_1], 1)

    print('the slope is ', fit_1[0])

    # Generate x-values for plotting lines
    x_vals = np.linspace(-0.5, 1.0, 100)
    y_fit_1 = np.polyval(fit_1, x_vals)


    x = HCO_1[mask_1]
    y = C18O_1[mask_1]
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)

    # Calculate R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(r_squared, coeffs)

    return x_vals,y_fit_1

def plot_parameters(save=False):
    # Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
    # vsini_values, vsini_uncertainty, ir_index_values, HCO_data, C18O_data, star_name=    read_parameters(filename)
    #
    # if molecule=='HCO+':
    #     molecule_data = HCO_data
    # elif molecule=='C18O':
    #     molecule_data =  C18O_data
    #
    # plt.figure(figsize=(8, 6))
    #
    # # plt.scatter(ir_index_values,molecule_data, edgecolor='k', s=100,c='C0')
    # # plt.xlabel('Spectral Index', fontsize=14)
    # # plt.ylabel('Concentration factor', fontsize=14)
    #
    #
    # plt.scatter(HCO_data,C18O_data, edgecolor='k', s=100,c='C1')
    # plt.xlabel('HCO+ concentration', fontsize=14)
    # plt.ylabel('C18O concentration', fontsize=14)
    #
    # # plt.title('Gravity vs Spectral Index', fontsize=16)
    # plt.grid(True, linestyle='--', alpha=0.6)
    #
    # one_to_one = np.linspace(0,1,10)
    # plt.plot(one_to_one,one_to_one,'k--',label='')
    # plt.ylim(-0.2,1)
    # plt.xlim(0,1)
    # #
    # # Show the plot
    # plt.tight_layout()

    # Define the data from the LaTeX table
    HCO_1sigma = [0.306, 0.1869, 0.6736, 0.3170, 0.4456, 0.8003, 0.1554, 0.7607, 0.4236, 0.4451, 0.1693, 0.6633,
                  0.7373, 0.4367, 0.4544, 0.4273, 0.9506, 0.6309, -0.5635, -1.7506]
    HCO_2sigma = [-0.00, -0.0807, 0.6611, 0.3078, 0.4397, 0.7773, 0.0877, 0.7412, 0.3349, 0.2210, 0.0819, 0.5586,
                  0.6998, 0.2461, 0.4544, 0.2086, 0.9376, 0.5687, -0.5898, -1.9718]
    HCO_3sigma = [np.nan, -0.3475, 0.6427, 0.3034, 0.4352, 0.7457, -0.0781, 0.7281, 0.2279, 0.1009, -0.0570, 0.3600,
                  0.6447, -0.0090, 0.4540, -0.0120, 0.9240, 0.4420, -0.6380, -2.2140]

    C18O_1sigma = [0.4871, 0.1642, 0.6130, 0.3260, 0.4933, 0.6358, 0.3307, 0.2969, 0.4446, 0.5044, 0.2839, 0.6789,
                   0.5641, -0.0707, 0.3645, 0.6057, 0.8673, 0.5360, 0.1770, 0.0907]
    C18O_2sigma = [0.3944, 0.1131, 0.6010, 0.3260, 0.4933, 0.6319, 0.3015, 0.2900, 0.3688, 0.4364, 0.2564, 0.6455,
                   0.5255, -0.4915, 0.3645, 0.5744, 0.8489, 0.3839, 0.1770, 0.0907]
    C18O_3sigma = [0.256, 0.0231, 0.5860, 0.3260, 0.4933, 0.6184, 0.2852, 0.2622, 0.2663, 0.3309, 0.1968, 0.6045,
                   0.4784, -0.8937, 0.3643, 0.5121, 0.8217, 0.1481, 0.1760, 0.0907]


    # Create the scatter plots
    plt.figure(figsize=(8, 7))
    fig1 = plt.subplot()
    fig1.scatter(HCO_1sigma, C18O_1sigma, color='#e6550d', label=r'1$\sigma$', s=200, edgecolor='k')
    fig1.scatter(HCO_2sigma, C18O_2sigma, color='#fd8d3c', label=r'2$\sigma$',s=200,marker='p',ec='k')
    fig1.scatter(HCO_3sigma, C18O_3sigma, color='#feedde', label=r'3$\sigma$',s=200,marker='d',ec='k')

    fig1.set_xlabel('HCO$^+$ Concentration Factor', fontsize=14)
    fig1.set_ylabel('C$^{18}$O Concentration Factor', fontsize=14)
    # plt.title('Comparison of HCO$^+$ vs. C$^{18}$O Concentration Factors')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    # fig1.set_ylim(-0.54,1)
    # fig1.set_xlim(-0.0,1)
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=14)

    one_to_one = np.linspace(-1.,1,10)
    fig1.plot(one_to_one,one_to_one,color='gray',linestyle=(0, (5, 10)))

    ### Simple polynomial fit
    # x_vals, y_fit = fit_polinomials(HCO_1sigma, C18O_1sigma)
    # fig1.plot(x_vals, y_fit, color='#e6550d', linestyle='--', label=r'1$\sigma$ fit')

    fig1.set_axisbelow(True)
    fig1.grid(True,zorder=3)

    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule+'_comparing_both_concentrations'+today+'.png'), bbox_inches='tight', dpi=300)
    plt.show()


def make_histogram_several_files(filename, map_file, ir_file, parameter='gravity',sigma_threshold=2, molecule='HCO+',save=False):

    Temp_values, temp_uncertainty, logg_values, logg_uncertainty, bfield_values, bfield_uncertainty, \
    vsini_values, vsini_uncertainty, old_ir_index_values, HCO_data, C18O_data, star_name =  read_parameters(filename)

    ir_index_values, ir_corrected_values, star_name_ir_index = read_ir_index_parameters(ir_file)


    logg_values = logg_values/1.e2
    T_mb, S_beam, S_beam_uncert, star_spectral_map = read_map_parameters(map_file)

    S_beam = np.array(S_beam)


    if molecule=='HCO+':
        # star_name_map, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma = read_c_factor_parameters('concentrations_2025-06-02_10_HCO+.txt')
        star_name_map, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma,\
        c_factor_gaussian = read_c_factor_parameters('text_files/concentrations_5arsrc_max_new_rad_2025-10-31_12_HCO+.txt')
        star_name_map_other, c_factor_1_sigma_other, c_factor_2_sigma_other, c_factor_3_sigma_other,\
        c_factor_gaussian_other = read_c_factor_parameters('text_files/concentrations_5arsrc_max_new_rad_2025-10-31_13_C18O.txt')
        other_molecule = 'C18O'

    elif molecule=='C18O':
        star_name_map, c_factor_1_sigma, c_factor_2_sigma, c_factor_3_sigma, \
        c_factor_gaussian = read_c_factor_parameters('text_files/concentrations_5arsrc_max_new_rad_2025-10-31_13_C18O.txt')
        star_name_map_other, c_factor_1_sigma_other, c_factor_2_sigma_other, c_factor_3_sigma_other,\
        c_factor_gaussian_other = read_c_factor_parameters('text_files/concentrations_5arsrc_max_new_rad_2025-10-31_12_HCO+.txt')
        other_molecule = 'HCO+'
    else:
        ValueError('only two values are possible C18O or HCO+')


    if sigma_threshold ==1:
        c_factor_n_sigma = c_factor_1_sigma
        c_factor_n_sigma_other = c_factor_1_sigma_other
    elif sigma_threshold==2:
        c_factor_n_sigma = c_factor_2_sigma
        c_factor_n_sigma_other = c_factor_2_sigma_other
    elif sigma_threshold==3:
        c_factor_n_sigma = c_factor_3_sigma
        c_factor_n_sigma_other = c_factor_2_sigma_other
    else:
        c_factor_n_sigma = c_factor_gaussian
        # raise ValueError("Number must be 1, 2, or 3 sigmas.")
        c_factor_n_sigma_other = c_factor_gaussian_other


    save_name, save_temp, save_logg, save_logg_uncert_h,\
    save_logg_uncert_l, save_corrected_IR_index, save_IR_index, save_this_c_factor, save_S_beam, save_S_beam_uncert, save_other_c_factor= [],[],[],[],[],[],[],[],[],[],[]

    #### get the spectral parameters associated to the sources of the main molecule
    for ii in range(len(star_name)):
        c_factor_aux=0.0
        for jj in range(len(star_name_map)):
            if are_names_approximate(star_name[ii], star_name_map[jj], threshold=0.9):
                # print(star_name[ii], star_name_map[jj], c_factor_n_sigma[jj],logg_values[ii])
                # print(ii)

                if math.isnan(c_factor_n_sigma[jj]) or c_factor_n_sigma[jj] == float('-inf'):
                    c_factor_aux = 0.0
                else:
                    c_factor_aux = c_factor_n_sigma[jj]

                continue
        save_name = np.append(save_name, star_name[ii])
        save_temp = np.append(save_temp, Temp_values[ii])
        save_logg = np.append(save_logg, logg_values[ii])
        save_logg_uncert_h = np.append(save_logg_uncert_h, logg_uncertainty[0][ii])
        save_logg_uncert_l = np.append(save_logg_uncert_l, logg_uncertainty[1][ii])

        save_this_c_factor = np.append(save_this_c_factor, c_factor_aux)
        # save_IR_index = np.append(save_IR_index,ir_index_values[ii])
        # save_corrected_IR_index = np.append(save_corrected_IR_index,ir_corrected_values)

    for ii in range(len(save_name)):
        c_factor_aux=0.0
        for jj in range(len(star_name_map_other)):
            if are_names_approximate(save_name[ii], star_name_map_other[jj], threshold=0.9):
                if math.isnan(c_factor_n_sigma_other[jj]) or c_factor_n_sigma_other[jj] == float('-inf'):
                    c_factor_aux = 0.0
                else:
                    c_factor_aux = c_factor_n_sigma_other[jj]

                # print(c_factor_aux,type(c_factor_aux))
                continue

        save_other_c_factor = np.append(save_other_c_factor, c_factor_aux)


    #### get the integrated intensities associated with the sources
    for ii in range(len(save_name)):

        if save_this_c_factor[ii] == 0.0:
            save_S_beam = np.append(save_S_beam,0)
            print(ii, save_name[ii])

        else:
            for jj in range(len(star_spectral_map)):

                if are_names_approximate(save_name[ii], star_spectral_map[jj], threshold=0.9):

                    save_S_beam = np.append(save_S_beam,S_beam[jj])
                    save_S_beam_uncert = np.append(S_beam_uncert,S_beam_uncert[jj])
                    continue


    protostars_parameter =[]
    not_protostar_parameter=[]
    # protostars_parameter_uncertainty =[]
    # not_protostars_parameter_uncertainty =[]

    if parameter.lower() == 'gravity':
        for ii in range(len(save_name)):
            if save_this_c_factor[ii]>0.2 and save_S_beam[ii]>0.3 and save_other_c_factor[ii]>0.2:
                protostars_parameter.append(save_logg[ii])
            else:
                not_protostar_parameter.append(save_logg[ii])

    elif parameter.lower() == 'temperature':
        for ii in range(len(save_name)):
            if save_this_c_factor[ii]>0.2 and save_S_beam[ii]>0.3 and save_other_c_factor[ii]>0.2:
                protostars_parameter.append(save_temp[ii])
            else:
                not_protostar_parameter.append(save_temp[ii])

        print('n_sources envelope:',len(protostars_parameter))
        print('n_source non_envelope:',len(not_protostar_parameter))

        print('min and max envelope:', np.nanmin(protostars_parameter), np.nanmax(protostars_parameter))
        print('min and max non_envelope:', np.nanmin(ot_protostar_parameter), np.nanmax(ot_protostar_parameter))

    save_name_ir_index, save_this_c_factor_ir_index, save_other_c_factor_ir_index,\
    save_S_beam_ir_index,save_IR_index, save_corrected_IR_index , save_logg_ir_index, save_temp_ir_index = [], [], [], [], [],[],[],[]

    #### get the spectral parameters associated to the spectral indices
    counter = 0
    for ii in range(len(star_name_ir_index)):
        for jj in range(len(save_name)):
            if are_names_approximate(star_name_ir_index[ii], save_name[jj], threshold=0.9):
                counter=counter+1

                # print(counter, star_name_ir_index[ii], save_name[jj],ir_corrected_values[ii],save_this_c_factor[jj] )
                # print(counter)

                save_this_c_factor_ir_index = np.append(save_this_c_factor_ir_index,save_this_c_factor[jj])
                save_other_c_factor_ir_index = np.append(save_other_c_factor_ir_index,save_other_c_factor[jj])
                save_S_beam_ir_index = np.append(save_S_beam_ir_index,save_S_beam[jj])
                save_logg_ir_index = np.append(save_logg_ir_index, save_logg[jj])
                save_temp_ir_index = np.append(save_temp_ir_index,save_temp[jj])
                save_IR_index = np.append(save_IR_index, ir_index_values[ii])
                save_corrected_IR_index = np.append(save_corrected_IR_index, ir_corrected_values[ii])
                save_name_ir_index = np.append(save_name_ir_index, star_name_ir_index[ii])




    if parameter.lower() == 'ir_index':
        for ii in range(len(save_IR_index)):
            if save_this_c_factor_ir_index[ii]>0.2 and save_S_beam_ir_index[ii]>0.3 and save_other_c_factor_ir_index[ii]>0.2:
                protostars_parameter.append(save_IR_index[ii])
                print('envelope ',save_name_ir_index[ii] , save_IR_index[ii] , round(save_logg_ir_index[ii],2), save_temp_ir_index[ii])

            else:
                not_protostar_parameter.append(save_IR_index[ii])
                print('non-envelope ',save_name_ir_index[ii] ,save_IR_index[ii] , round(save_logg_ir_index[ii],2), save_temp_ir_index[ii])

        print('n_sources envelope:', len(protostars_parameter))
        print('n_source non_envelope:', len(not_protostar_parameter))

        print('min and max envelope:', np.nanmin(protostars_parameter), np.nanmax(protostars_parameter))
        print('min and max non_envelope:', np.nanmin(not_protostar_parameter), np.nanmax(not_protostar_parameter))

    elif parameter.lower() == 'corrected_ir_index':
        for ii in range(len(save_IR_index)):
            if save_this_c_factor_ir_index[ii]>0.2 and save_S_beam_ir_index[ii]>0.3 and save_other_c_factor_ir_index[ii]>0.2:
                protostars_parameter.append(save_corrected_IR_index[ii])
                print('envelope ',save_name_ir_index[ii] ,save_corrected_IR_index[ii])
            else:
                not_protostar_parameter.append(save_corrected_IR_index[ii])


        print('n_sources envelope:', len(protostars_parameter))
        print('n_source non_envelope:', len(not_protostar_parameter))

        print('min and max envelope:', np.nanmin(protostars_parameter), np.nanmax(protostars_parameter))
        print('min and max non_envelope:', np.nanmin(not_protostar_parameter), np.nanmax(not_protostar_parameter))

    res_t_test = stats.ttest_ind(protostars_parameter, not_protostar_parameter, equal_var=False)
    res_ks_test = stats.kstest(protostars_parameter, not_protostar_parameter)

    print('T test \n'*5)
    print(res_t_test)

    print('KS test \n'*5)
    print(res_ks_test)

    mean_val_1 = round(np.nanmean(protostars_parameter), 2)
    std_val_1 = round(np.nanstd(protostars_parameter), 2)

    mean_val_2 = round(np.nanmean(not_protostar_parameter), 2)
    std_val_2 = round(np.nanstd(not_protostar_parameter), 2)



    print(len(protostars_parameter))
    print(len(not_protostar_parameter))

    # plt.title(molecule)
    if parameter.lower() == 'gravity':
        x_label ='log(g)'
        bins = np.arange(2.7, 4.1, 0.2)
        x_leg, y_leg = 2.75, 6.5

        print('mean gravity envelope', mean_val_1)
        print('std error mean gravity envelope', std_val_1/(len(protostars_parameter))**0.5)

        print('mean gravity non-envelope', mean_val_2)
        print('std error mean gravity non-envelope', std_val_2/(len(not_protostar_parameter))**0.5)

        print('mean, median, and std uncertainty of the gravities',
              np.nanmean(logg_uncertainty),np.nanmedian(logg_uncertainty),np.nanstd(np.nanmean(logg_uncertainty)))

        # print('mean, median of envelope', np.nanmean(protostars_parameter_uncertainty), np.nanmedian(protostars_parameter_uncertainty))
        # print('mean, median of non-envelope', np.nanmean(not_protostars_parameter_uncertainty), np.nanmedian(not_protostars_parameter_uncertainty))

        # plt.legend(loc='upper left')

    elif parameter.lower() == 'temperature':
        x_label ='Temperature'
        bins = np.arange(3000, 4400, 100)
        x_leg, y_leg = 3000, 5

        print('mean temperature envelope', mean_val_1)
        print('std error mean temperature envelope', std_val_1/(len(protostars_parameter))**0.5)

        print('mean temperature non-envelope', mean_val_2)
        print('std error mean temperature non-envelope', std_val_2/(len(not_protostar_parameter))**0.5)

        print('mean, median, and std uncertainty of the temperature',
              np.nanmean(logg_uncertainty), np.nanmedian(logg_uncertainty), np.nanstd(np.nanmean(logg_uncertainty)))

    elif parameter.lower() == 'ir_index':
        x_label ='IR index'
        bins = np.arange(-1.3, 1.3, 0.25)
        x_leg, y_leg = 0.1, 4
        # plt.legend(loc='upper right')
        print('meanir_index envelope', mean_val_1)
        print('std error mean gravity envelope', std_val_1/(len(protostars_parameter))**0.5)

        print('mean ir_index non-envelope', mean_val_2)
        print('std error mean gravity non-envelope', std_val_2/(len(not_protostar_parameter))**0.5)

        print('mean, median, and std uncertainty of the ir_index',
              np.nanmean(logg_uncertainty), np.nanmedian(logg_uncertainty), np.nanstd(np.nanmean(logg_uncertainty)))

    elif parameter.lower() == 'corrected_ir_index':
        x_label ='Corrected IR index'
        bins = np.arange(-1.3, 1.3, 0.25)
        x_leg, y_leg = 0.3, 4
        # plt.legend(loc='upper right')
        print('mean Corrected IR index envelope', mean_val_1)
        print('std error mean Corrected IR index envelope', std_val_1/(len(protostars_parameter))**0.5)

        print('mean Corrected IR index non-envelope', mean_val_2)
        print('std error mean Corrected IR index non-envelope', std_val_2/(len(not_protostar_parameter))**0.5)

        print('mean, median, and std uncertainty of the Corrected IR index',
              np.nanmean(logg_uncertainty), np.nanmedian(logg_uncertainty), np.nanstd(np.nanmean(logg_uncertainty)))

    # plt.hist(not_protostar_parameter,bins,alpha=0.7, label='non-detections', histtype='step',edgecolor="C1",lw=2)
    plt.hist(not_protostar_parameter,bins,alpha=0.5, label='dissipated envelope', edgecolor="black")

    # plt.hist(protostars_parameter,bins,alpha=0.7, label='detections', edgecolor="C0", histtype='step',lw=2)
    plt.hist(protostars_parameter,bins,alpha=0.5, label='detected envelope', edgecolor="black")

    plt.xlabel(x_label,size=14)

    plt.text(x=x_leg, y=y_leg, s=r'$\mu$ = ' + str(mean_val_1) + r' $\sigma$ = ' + str(std_val_1), size=12, color='C1',
             weight=600)
    plt.text(x=x_leg, y=y_leg - 0.5, s=r'$\mu$ = ' + str(mean_val_2) + r' $\sigma$ = ' + str(std_val_2), size=12,
             color='C0', weight=600)

    plt.legend()
    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule + 'alpha_corrected_ir_index_Mike_Hist_'+ today +'.png'),
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
        is_numeric = [not math.isnan(x) and x>0.2 for x in HCO_data]
        is_nan = [math.isnan(x) or x<0.2 for x in HCO_data]
        # print()
        # molecular_data=HCO_data
        # min_val= -2

    elif molecule=='C18O':
        # Check that all input arrays have the same length
        if len(logg_values) != len(ir_index_values) or len(logg_values) != len(C18O_data):
            raise ValueError("All input arrays must have the same length.")

        is_numeric = [not math.isnan(x) and x>0.2 for x in C18O_data]
        is_nan = [math.isnan(x) or x<0.2 for x in C18O_data]
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
    elif parameter.lower() == 'hco':
        my_parameter = HCO_data
        bins = np.arange(-1.9,1.0,0.1)
        x_label ='HCO+ concentration'
        x_leg,y_leg = 0.4, 5
    elif parameter.lower() == 'c18o':
        my_parameter = C18O_data
        bins = np.arange(-1.0,1.0,0.1)
        x_label ='C18O concentration'
        x_leg,y_leg = 0.4, 5
    else:
        print('wrong parameter')

    res= stats.ttest_ind(my_parameter[is_numeric], my_parameter[is_nan], equal_var=False)
    print(res)

    mean_val_1 = round(np.nanmean(my_parameter[is_numeric]),3)
    std_val_1 = round(np.nanstd(my_parameter[is_numeric]),3)

    mean_val_2 = round(np.nanmean(my_parameter[is_nan]),3)
    std_val_2 = round(np.nanstd(my_parameter[is_nan]),3)

    # plt.text(x=x_leg,y=y_leg,s=r'$\mu$ = '+str(mean_val_1) +r' $\sigma$ = '+str(std_val_1),size=12,color='C0',weight=600)
    # plt.text(x=x_leg,y=y_leg-1,s=r'$\mu$ = '+str(mean_val_2) +r' $\sigma$ = '+str(std_val_2),size=12,color='C1',weight=600)

    # plt.hist(my_parameter[is_numeric],bins,alpha=0.7, label='detections', edgecolor="black" , color='C0')
    # plt.hist(my_parameter[is_nan],bins,alpha=0.7, label='non-detections', edgecolor="black", color='C0')
    plt.hist(my_parameter,bins,alpha=0.7, label='non-detections', edgecolor="black", color='C0')

    plt.xlabel(x_label,size=14)
    # plt.legend(loc='upper left')
    plt.title(molecule)
    if save:
        plt.savefig(os.path.join('Figures/concentration/', molecule + '_histogram_C04'+parameter+'.png'),
                    bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":

    source_name = 'GV_Tau'
    molecule ='HCO+'
    # molecule ='C18O'

    moment_maps_folder = 'moment_maps'
    aperture=None

    calculate_concentration_factor(source_name, molecule,
                                   n_sigma=1, gaussian_fit=True,
                                   plot=True, use_skycoord=True,
                                   moment_maps_folder=moment_maps_folder,
                                   aperture=aperture)

    # save_concentration_factors_to_file(folder_fits='sdf_and_fits',
    #                                    molecule=molecule,
    #                                    save_filename='concentrations_1arcsec_max_mov_'+ today + '_' + hour_now + '_' + molecule+'.txt')


    # plot_parameters(save=True)

    # plot_stellar_params_and_c_factors(spectrum_file='text_files/Class_I_for-JCMT-plots.txt',
    #                                   sigma_threshold=2,
    #                                   # sigma_threshold='gaussian',
    #                                   molecule=molecule, save=False, color_map='gist_rainbow', use_other_c_cbar=True)

    # plot_stellar_params_and_coldense(spectrum_file='text_files/Class_I_for-JCMT-plots-with_names-corrected.txt',
    #                                 map_file='text_files/envelope_mass_2025-05-21_22_C18O.txt',molecule=molecule,save=False)

    # make_histograms(filename='text_files/Class_I_for-JCMT-plots.txt', parameter='hco',
    #                                molecule=molecule,save=False)

    # make_histograms(filename='text_files/Class_I_for-JCMT-plots-with_names-corrected.txt', parameter='gravity',
    #                                molecule=molecule,save=False)

    # make_histogram_several_files(filename='text_files/Class_I_for-JCMT-plots.txt'
    #                              , map_file='spectrum_parameters_HCO+.txt',ir_file='text_files/my_spectral_indices_2025-07-27.txt',
    #                              parameter='gravity',molecule=molecule, save=False) #ir_index

    # plot_spectral_vs_map_parameters(spectrum_file='text_files/Class_I_for-JCMT-plots.txt',
    #                                 spectral_map_file='spectrum_parameters_'+molecule+'.txt',
    #                                 sigma_threshold=2,molecule=molecule, color_map='gist_rainbow',save=True)