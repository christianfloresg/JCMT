#!/usr/bin/env bash 
export STARLINK_DIR=/Users/christianflores/Programs/star-2023A 
source $STARLINK_DIR/etc/profile  
cd /Users/christianflores/Documents/work/Astronomy_data/JCMT/M23AP008/IRS5/HCO+/reduced/
kappa 
INPUTNAME="ga20230703_53_1_0p20bin001" 
SOURCE="IRS5" 
MOLEC="HCO+" 
RESA="_resampled" 
SOURCENAME=$SOURCE"_"$MOLEC 
RESAMP=$SOURCE"_"$MOLEC$RESA 
cdiv in=$INPUTNAME.sdf scalar=0.63 out=$SOURCENAME.sdf 
setunits $SOURCENAME.sdf units="K km/s " 
sqorst in=$SOURCENAME.sdf out=$RESAMP.sdf factors="[6,6,1] conserve" 
convert 
ndf2fits in=$RESAMP.sdf out=$RESAMP.fits 
mkdir /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/IRS5
mv $RESAMP.sdf /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/IRS5
mv $RESAMP.fits /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/IRS5 
cd /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/IRS5 
mv $RESAMP.sdf $SOURCENAME.sdf 
mv $RESAMP.fits $SOURCENAME.fits 
