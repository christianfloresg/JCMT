import os
import subprocess
import numpy as np


def find_files_in_folder(path_to_folder):
    for x in os.listdir(path_to_folder):
        if x.endswith(".fits"):
            # Prints only text file present in My Folder
            return x

def create_shell_script_data_reduction(path_to_folder):

    file = open(os.path.join(path_to_folder,"myfile.sh"), "w")
    file.write("#!/usr/bin/env bash \n")
    file.write("export STARLINK_DIR=/Users/christianflores/Programs/star-2023A \n")
    file.write("source $STARLINK_DIR/etc/profile \n")
    file.write("oracdr_acsis \n")
    file.write("export ORAC_DATA_IN=. \n")
    file.write("ls "+path_to_folder+"*.sdf > mylist \n")
    file.write("oracdr -loop file -file mylist -onegroup \n")
    file.close()

def archival_data_preparation(path_to_folder):
    subprocess.check_call(['bash',os.path.join(path_to_folder,"myfile.sh")], cwd=path_to_folder)


def print_path(name_sdf,name_star,molecule):

    if molecule=='C18O':
        print("export ORAC_DATA_IN=.")
        name_Tmb = name_star+"_C18O_Tmb"
        print("cdiv in = "+ name_sdf + " scalar = 0.63 out = "+name_Tmb)

        name_resmapled = name_star+"_C18O_Tmb_resampled"
        print("sqorst in = "+name_Tmb+" out = "+name_resmapled +" factors = [6, 6, 1] conserve ")
        fits_name = name_star

        print("ndf2fits in = " + name_resmapled +".sdf" + " out = "+fits_name+"_C18O.fits")

    elif molecule=='HCO+':
        print("export ORAC_DATA_IN=.")
        name_Tmb = name_star+"_HCO+_Tmb"
        print("cdiv in = "+ name_sdf + " scalar = 0.63 out = "+name_Tmb)

        name_resmapled = name_star+"_HCO+_Tmb_resampled"
        print("sqorst in = "+name_Tmb+" out = "+name_resmapled +" factors = [6, 6, 1] conserve ")
        fits_name = name_star

        print("ndf2fits in = " + name_resmapled +".sdf" + " out = "+fits_name+"_HCO+.fits")

if __name__ == "__main__":
    print_path(name_sdf="ga20231125_72_1_reduced001.sdf",name_star="IRAS04591-0856",molecule="HCO+")
    # print_path(name_sdf="ga20231125_74_1_reduced001.sdf",name_star="IRAS05379-0758",molecule="C18O")

    # path_to_folder = '/Users/christianflores/Documents/work/Astronomy_data/JCMT/M22BH10B/IRAS04591-0856/HCO+/'

    # create_shell_script_data_reduction(path_to_folder)
    # archival_data_preparation(path_to_folder)