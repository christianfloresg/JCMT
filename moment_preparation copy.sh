#!/usr/bin/env bash

kappa

INPUTNAME="ga20220820_15_1_0p20bin001_new"

SOURCE="V347_Aur"
#MOLEC="C18O"
MOLEC="HCO+"
RESA="_resamp"
SOURCENAME=$SOURCE$MOLEC
RESAMP=$SOURCE$MOLEC$RESA

cdiv in=$INPUTNAME.sdf scalar=0.63 out=$SOURCENAME.sdf

setunits $SOURCENAME.sdf units=\"K km/s \"

sqorst in=$SOURCENAME.sdf out=$RESAMP.sdf factors="[6,6,1] conserve"

convert
ndf2fits in=$RESAMP.sdf out=$RESAMP.fits