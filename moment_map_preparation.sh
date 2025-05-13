#!/usr/bin/env bash 
export STARLINK_DIR=/Users/christianflores/Programs/star-2023A 
source $STARLINK_DIR/etc/profile  
cd /Users/christianflores/Documents/work/Astronomy_data/JCMT/M25AP032/DoAr43/C18O/reduced/
kappa 
INPUTNAME="ga20250502_63_1_0p20bin001" 
SOURCE="DoAr43" 
MOLEC="C18O" 
RESA="_resampled" 
SOURCENAME=$SOURCE"_"$MOLEC 
RESAMP=$SOURCE"_"$MOLEC$RESA 
convert 
cdiv in=$INPUTNAME.sdf scalar=0.63 out=$SOURCENAME.sdf 
setunits $SOURCENAME.sdf units="K km/s " 
ndf2fits in=$SOURCENAME.sdf out=$SOURCENAME"_original".fits 
sqorst in=$SOURCENAME.sdf out=$RESAMP.sdf factors="[4,4,1]" 
ndf2fits in=$RESAMP.sdf out=$RESAMP.fits 
mkdir /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/DoAr43
mv $RESAMP.sdf /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/DoAr43
mv $RESAMP.fits /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/DoAr43 
cd /Users/christianflores/Documents/GitHub/JCMT/sdf_and_fits/DoAr43 
mv $RESAMP.sdf $SOURCENAME.sdf 
mv $RESAMP.fits $SOURCENAME.fits 
