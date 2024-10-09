import numpy as np
from scipy.optimize import curve_fit

# Define a 2D Gaussian function
def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    (x, y) = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g.ravel()

# Function to fit the Gaussian and calculate FWHM
def fit_gaussian_2d(image):
    # Create x and y indices
    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    x, y = np.meshgrid(x, y)

    # Initial guess for the parameters: amplitude, x0, y0, sigma_x, sigma_y, theta, offset
    initial_guess = (np.max(image), image.shape[1]//2, image.shape[0]//2, 1, 1, 0, np.min(image))

    # Fit the Gaussian
    popt, _ = curve_fit(gaussian_2d, (x, y), image.ravel(), p0=initial_guess)

    # Extract the fit parameters
    amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt

    # Calculate FWHM for both x and y directions
    fwhm_x = 2.355 * sigma_x
    fwhm_y = 2.355 * sigma_y

    return fwhm_x, fwhm_y, popt  # Return the FWHM values and the fit parameters

# Example usage
if __name__ == "__main__":
    # Create a 2D Gaussian image for testing
    x = np.linspace(0, 50, 50)
    y = np.linspace(0, 50, 50)
    x, y = np.meshgrid(x, y)
    test_image = gaussian_2d((x, y), 3, 25, 25, 5, 7, 0, 10).reshape(50, 50)

    # Fit the Gaussian to the image
    fwhm_x, fwhm_y, params = fit_gaussian_2d(test_image)
    print(f"FWHM in x: {fwhm_x}")
    print(f"FWHM in y: {fwhm_y}")
