#!/usr/bin/env bash 
export STARLINK_DIR=/Users/christianflores/Programs/star-2023A 
source $STARLINK_DIR/etc/profile  
cd /Users/christianflores/Documents/work/Astronomy_data/JCMT/M24AH15A/DG-Tau/HCO+/reduced/
kappa 
INPUTNAME="ga20120728_48_1_0p20bin001" 
SOURCE="DG-Tau" 
MOLEC="HCO+" 
RESA="_resampled" 
SOURCENAME=$SOURCE"_"$MOLEC 
RESAMP=$SOURCE"_"$MOLEC$RESA 
convert 
cdiv in=$INPUTNAME.sdf scalar=0.63 out=$SOURCENAME.sdf 
setunits $SOURCENAME.sdf units="K km/s " 
ndf2fits in=$SOURCENAME.sdf out=$SOURCENAME"_original".fits 
sqorst in=$SOURCENAME.sdf out=$RESAMP.sdf factors="[4,4,1]" 
ndf2fits in=$RESAMP.sdf out=$RESAMP.fits 
mkdir /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/DG-Tau
mv $RESAMP.sdf /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/DG-Tau
mv $RESAMP.fits /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/DG-Tau 
cd /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/DG-Tau 
mv $RESAMP.sdf $SOURCENAME.sdf 
mv $RESAMP.fits $SOURCENAME.fits 
