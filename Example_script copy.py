#### A script to show how to use each of the 4 functions in BTS.
import BTS
import numpy as np


#### Example to show how to fit a single spectrum

# Read in the relevant parameter file

if __name__ == "__main__":
    # Read in the relevant parameter file
    param = BTS.read_parameters("./Fit_cube.param")

    # Run the function to make the moments using the moment-masking technique
    BTS.make_moments(param)
