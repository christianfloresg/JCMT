import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel

def fwhm_to_sigma_pix(fwhm_arcsec, pixscale_arcsec):
    """Convert FWHM (arcsec) to Gaussian sigma (pixels)."""
    sigma_arcsec = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma_arcsec / pixscale_arcsec

def kernel_fwhm_arcsec(theta_nat_arcsec, theta_targ_arcsec):
    """Gaussian kernel FWHM (arcsec) needed to go from native to target beam."""
    if theta_targ_arcsec <= theta_nat_arcsec:
        raise ValueError("Target beam must be larger than native beam (no deconvolution).")
    return np.sqrt(theta_targ_arcsec**2 - theta_nat_arcsec**2)

def convolve_cube_spatial_fft(
    cube,
    theta_nat_arcsec,
    theta_targ_arcsec,
    pixscale_arcsec,
    nan_treatment="interpolate"
):
    """
    Convolve a spectral cube spatially (per channel) from native to target beam.
    cube: numpy array with shape (nv, ny, nx) OR (ny, nx, nv) (handled below)
    Returns convolved cube (same shape as input) and kernel FWHM (arcsec).
    """
    # Ensure cube is (nv, ny, nx)
    orig_shape = cube.shape
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {orig_shape}")

    # Heuristic: if last axis is much smaller than first, might be (ny,nx,nv)
    # Safer is to assume FITS ordering (nv, ny, nx) after reading typical radio cubes.
    # We'll only transpose if it looks like (ny,nx,nv).
    transposed = False
    if orig_shape[0] == orig_shape[1] and orig_shape[2] != orig_shape[0]:
        # ambiguous; do nothing
        pass
    if orig_shape[0] > 16 and orig_shape[2] <= 16:
        # likely (ny, nx, nv) with small nv
        cube = np.moveaxis(cube, -1, 0)
        transposed = True

    theta_ker = kernel_fwhm_arcsec(theta_nat_arcsec, theta_targ_arcsec)
    sigma_pix = fwhm_to_sigma_pix(theta_ker, pixscale_arcsec)
    kernel = Gaussian2DKernel(x_stddev=sigma_pix)

    nv, ny, nx = cube.shape
    out = np.empty_like(cube, dtype=np.float32)

    # Convolve each channel spatially
    for i in range(nv):
        out[i] = convolve_fft(
            cube[i].astype(np.float32),
            kernel,
            allow_huge=True,
            nan_treatment=nan_treatment,
            normalize_kernel=True
        )

    # Restore original axis order if we transposed
    if transposed:
        out = np.moveaxis(out, 0, -1)

    return out, theta_ker

def convolve_datacube_fits(infile, outfile, theta_targ_arcsec, overwrite=True):
    """
    Read a FITS spectral cube, convolve spatially to target beam, and write new FITS.
    Updates beam keywords in header (BMAJ/BMIN/BPA) and HISTORY.
    """
    with fits.open(infile) as hdul:
        hdu = hdul[0]
        data = hdu.data
        header = hdu.header.copy()

    # Drop leading singleton axes if present (common: (1, nv, ny, nx))
    while data.ndim > 3 and data.shape[0] == 1:
        data = data[0]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube after squeezing, got shape {data.shape}")

    # Pixel scale from header (degrees/pixel -> arcsec/pixel)
    if "CDELT1" not in header:
        raise KeyError("No CDELT1 found in header; cannot infer pixel scale.")
    pixscale_arcsec = abs(header["CDELT1"]) * 3600.0

    # Native beam: prefer BMAJ/BMIN if present; else error
    # if "BMAJ" not in header:
    #     raise KeyError("No BMAJ found in header; cannot infer native beam.")
    # bmaj_arcsec = header["BMAJ"] * 3600.0
    # bmin_arcsec = header.get("BMIN", header["BMAJ"]) * 3600.0

    bmaj_arcsec = 14
    bmin_arcsec = 14

    # For C-factor use you may adopt circularized beam; for convolution assume circular kernel
    theta_nat_arcsec = np.sqrt(bmaj_arcsec * bmin_arcsec)

    data_conv, theta_ker = convolve_cube_spatial_fft(
        data,
        theta_nat_arcsec=theta_nat_arcsec,
        theta_targ_arcsec=theta_targ_arcsec,
        pixscale_arcsec=pixscale_arcsec
    )

    # Update header to reflect new beam
    header["BMAJ"] = theta_targ_arcsec / 3600.0
    header["BMIN"] = theta_targ_arcsec / 3600.0
    header["BPA"] = 0.0

    header.add_history(
        f"Spatially convolved cube from "
        f"{theta_nat_arcsec:.3f}\" (circ. from BMAJ/BMIN) to "
        f"{theta_targ_arcsec:.3f}\" using Gaussian kernel FWHM={theta_ker:.3f}\""
    )

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fits.writeto(outfile, data_conv.astype(np.float32), header, overwrite=overwrite)

    return theta_ker

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example: match to 43 arcsec beam
    distance = 400
    theta_target = 14.0 * 200/130. # arcsec
    theta_nat_arcsec = 14.0
    molecule = "HCO+"

    infile = "sdf_and_fits/GV_Tau/GV_Tau_"+molecule+".fits"
    outfile = "sdf_and_fits_degraded/GV_Tau_"+molecule+"_"+str(round(theta_target,0))+"_arcsec.fits"

    theta_kernel = convolve_datacube_fits(infile, outfile, theta_target)
    print(f"Wrote {outfile}")
    print(f"Applied spatial convolution kernel FWHM = {theta_kernel:.2f}\"")

