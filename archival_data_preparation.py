import os
import subprocess
import numpy as np

def find_files_in_folder(path_to_folder):
    for x in os.listdir(path_to_folder):
        if x.endswith(".sdf"):
            # Prints only text file present in My Folder
            return x

def create_shell_script(path_to_folder):

    star_name= path_to_folder.split('/')[-2]
    molec_line= path_to_folder.split('/')[-1]

    sdf_file = star_name+"_"+molec_line+".sdf"
    tmb_file = star_name+"_"+molec_line+"_Tmb"+".sdf"
    resampled_file = star_name+"_"+molec_line+"Tmb_resampled"+".sdf"
    final_fits_file = star_name+"_"+molec_line+".fits"

    file = open(os.path.join(path_to_folder,"myfile.sh"), "w")
    file.write("#!/usr/bin/env bash \n")
    file.write("export STARLINK_DIR=/Users/christianflores/Programs/star-2023A \n")
    file.write("source $STARLINK_DIR/etc/profile \n")
    file.write("convert \n")

    fits_file = find_files_in_folder(path_to_folder)

    file.write("fits2ndf "+fits_file+" "+ sdf_file +"\n")

    file.write("kappa \n")
    file.write("wcsattrib ndf="+sdf_file+" mode=set name=system\(3\) newval=vrad \n")
    file.write("wcsattrib ndf="+sdf_file+" mode=set name=StdofRest newval=LSRK\n")

    file.write("cdiv in="+sdf_file+" scalar=0.63 out="+tmb_file +"\n")
    file.write('sqorst in='+tmb_file+' out='+resampled_file+' factors="[4,4,1] conserve"' +"\n")


    file.write("ndf2fits in="+resampled_file+" out="+final_fits_file +"\n")

    file.close()

def archival_data_preparation(path_to_folder):
    subprocess.check_call(['bash',os.path.join(path_to_folder,"myfile.sh")], cwd=path_to_folder)


def print_path(name_sdf,name_star,molecule):

    if molecule=='C18O':
        print("export ORAC_DATA_IN=.")
        name_Tmb = name_star+"_C18O_Tmb"
        print("cdiv in = "+ name_sdf + " scalar = 0.63 out = "+name_Tmb)

        name_resmapled = name_star+"_C18O_Tmb_resampled"
        print("sqorst in = "+name_Tmb+" out = "+name_resmapled +" factors = [4, 4, 1] conserve ")
        fits_name = name_star

        print("ndf2fits in = " + name_resmapled +".sdf" + " out = "+fits_name+"_C18O.fits")

    elif molecule=='HCO+':
        print("export ORAC_DATA_IN=.")
        name_Tmb = name_star+"_HCO+_Tmb"
        print("cdiv in = "+ name_sdf + " scalar = 0.63 out = "+name_Tmb)

        name_resmapled = name_star+"_HCO+_Tmb_resampled"
        print("sqorst in = "+name_Tmb+" out = "+name_resmapled +" factors = [4, 4, 1] conserve ")
        fits_name = name_star

        print("ndf2fits in = " + name_resmapled +".sdf" + " out = "+fits_name+"_HCO+.fits")



def compute_concentration_factor(B,R_obs,S_int,S_peak):
    """
    Concentration factor as derived in Carney  et al. 2016
    :param B:
    :param R_obs:
    :param S_int:
    :param S_peak:
    :return:
    """

    return 1 - (1.13*B**2*S_int)/(np.pi*R_obs**2*S_peak)


if __name__ == "__main__":
    path_to_folder = '/Users/christianflores/Documents/work/Astronomy_data/JCMT/Archival/IRAS04181+2655/HCO+'

    create_shell_script(path_to_folder)
    # archival_data_preparation(path_to_folder)
    # print(compute_concentration_factor(B=15,R_obs=25,S_int=14.1,S_peak=4.7))
    # print(compute_concentration_factor(B=15,R_obs=45,S_int=160,S_peak=17.6))