import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from plot_generation_JCMT import find_simbad_source_in_file, get_icrs_coordinates
from astropy.constants import k_B, h, c, m_p

def angular_radius_arcsec(r_au: float, distance_pc: float) -> float:
    """
    Small-angle: 1 au at 1 pc = 1 arcsec, so theta(") = r_au / d_pc
    """
    return r_au / distance_pc

def pixel_scale_arcsec_per_pix(header) -> float:
    """
    Estimate pixel scale from WCS keywords.
    Works for typical celestial axes where CDELT is deg/pix.
    """
    if "CDELT1" in header:
        return abs(header["CDELT1"]) * 3600.0
    # Fallback: CD matrix
    if "CD1_1" in header:
        return abs(header["CD1_1"]) * 3600.0
    raise KeyError("Cannot determine pixel scale (need CDELT1 or CD1_1).")

def aperture_mask(ny, nx, x0, y0, r_pix):
    """
    Boolean mask for circular aperture in pixel coordinates.
    x0,y0 in pixel coords (0-based), r_pix in pixels.
    """
    yy, xx = np.ogrid[:ny, :nx]
    return (xx - x0)**2 + (yy - y0)**2 <= r_pix**2


def fixed_physical_aperture_integrate_mom0(
    fits_path: str,
    distance_pc: float,
    radius_au: float = 6000.0,
    center_ra_dec: tuple | None = None,   # (ra_deg, dec_deg)
    center_pix: tuple | None = None,      # (x_pix, y_pix) in 0-based pixel coordinates
    ext: int = 0,
    nan_policy: str = "ignore",           # "ignore" or "zero"
):
    """
    Integrate a moment-0 map (K km/s) within a fixed physical radius (au),
    consistently across distances.

    Returns a dict with:
      - theta_arcsec: aperture radius in arcsec
      - r_pix: aperture radius in pixels
      - sum_Kkms_arcsec2: sum of (K km/s)*arcsec^2 within aperture
      - mean_Kkms: area-weighted mean moment0 within aperture (K km/s)
      - n_pix: number of pixels in aperture
      - center_pix: (x0, y0)
    """
    with fits.open(fits_path) as hdul:
        hdu = hdul[ext]
        data = np.array(hdu.data, dtype=float)
        header = hdu.header.copy()

    # Handle possible extra singleton dimensions (e.g. (1, ny, nx))
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D moment-0 map; got shape {data.shape}")

    ny, nx = data.shape
    wcs = WCS(header)

    # Pixel scale
    pixscale = pixel_scale_arcsec_per_pix(header)  # arcsec/pix
    pix_area_arcsec2 = pixscale**2

    # Aperture angular radius
    theta_arcsec = angular_radius_arcsec(radius_au, distance_pc)
    r_pix = theta_arcsec / pixscale

    # Center determination
    if center_pix is None and center_ra_dec is None:
        # default: peak pixel (robust-ish for moment0 products)
        # (you can change this to any other center definition)
        if nan_policy == "ignore":
            idx = np.nanargmax(data)
        else:
            idx = np.argmax(np.nan_to_num(data, nan=0.0))
        y0, x0 = np.unravel_index(idx, data.shape)
    elif center_pix is not None:
        x0, y0 = center_pix
        x0, y0 = float(x0), float(y0)
    else:
        ra_deg, dec_deg = center_ra_dec
        sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        # world_to_pixel returns (x, y) in 0-based coordinates
        x0, y0 = wcs.world_to_pixel(sc)
        x0, y0 = float(x0), float(y0)

    # Build aperture mask
    mask = aperture_mask(ny, nx, x0, y0, r_pix)

    vals = data[mask]
    if nan_policy == "ignore":
        vals = vals[~np.isnan(vals)]
    elif nan_policy == "zero":
        vals = np.nan_to_num(vals, nan=0.0)
    else:
        raise ValueError("nan_policy must be 'ignore' or 'zero'")

    n_pix = vals.size
    if n_pix == 0:
        raise RuntimeError("Aperture contains zero valid pixels (check center/radius/WCS).")

    # Sum over solid angle: (K km/s) * arcsec^2
    sum_Kkms_arcsec2 = float(np.sum(vals) * pix_area_arcsec2)

    # Mean intensity in aperture (K km/s): sum / area
    area_arcsec2 = float(n_pix * pix_area_arcsec2)
    mean_Kkms = float(sum_Kkms_arcsec2 / area_arcsec2)

    return {
        "theta_arcsec": float(theta_arcsec),
        "r_pix": float(r_pix),
        "sum_Kkms_arcsec2": sum_Kkms_arcsec2,
        "mean_Kkms": mean_Kkms,
        "n_pix": int(n_pix),
        "center_pix": (float(x0), float(y0)),
        "pixscale_arcsec": float(pixscale),
        "area_arcsec2": area_arcsec2,
    }

def upper_limits_from_cube_rms(
    sigma_chan_K,
    dv_kms,
    DeltaV_kms,
    beam_fwhm_arcsec,
    distance_pc,
    R_au=6000.0,
    nsigma=3.0
):
    """
    Compute upper limits for integrated intensity in a fixed physical aperture,
    using only cube RMS and assumed integration width.

    Returns:
      sigma_W (1-beam) in K km/s,
      N_beam,
      W_mean_UL (nsigma) in K km/s,
      W_sum_UL  (nsigma) in K km/s * beam  (i.e. sum over independent beams)
    """
    # aperture radius (arcsec)
    theta_arcsec = R_au / distance_pc
    A_ap = np.pi * theta_arcsec**2  # arcsec^2

    # beam area (arcsec^2)
    A_beam = 1.133 * beam_fwhm_arcsec**2

    # independent beams in aperture
    N_beam = max(A_ap / A_beam, 1.0)

    # 1-sigma integrated intensity per beam (K km/s)
    sigma_W = sigma_chan_K * np.sqrt(dv_kms * DeltaV_kms)

    # upper limits
    W_mean_UL = nsigma * sigma_W / np.sqrt(N_beam)
    W_sum_UL  = nsigma * sigma_W * np.sqrt(N_beam)

    return {
        "theta_arcsec": theta_arcsec,
        "N_beam": N_beam,
        "sigma_W_Kkms_per_beam": sigma_W,
        "W_mean_UL_Kkms": W_mean_UL,
        "W_sum_UL_Kkms_beamsum": W_sum_UL
    }

# ----------------------------
# C18O(3-2) LTE, optically thin column density
# ----------------------------


def c18o32_Ntot_from_W(W_Kkms, Tex_K,
                      nu_GHz=329.330,         # GHz  (C18O 3-2) :contentReference[oaicite:1]{index=1}
                      A_ul=2.16e-6,           # s^-1 (typical value; override if desired)
                      Eu_over_k=31.6,         # K    (C18O 3-2) :contentReference[oaicite:2]{index=2}
                      gu=7):
    """
    Convert W = ∫Tmb dv (K km/s) to Ntot(C18O) (cm^-2) assuming LTE & optically thin.

    Uses:
      Ntot = (8π k ν^2)/(h c^3 Aul) * (Q(Tex)/gu) * exp(Eu/kTex) / (1 - exp(-hν/kTex)) * W

    For a linear rotor, Q(T) ≈ kT/(hB), with B from Eu/k = (hB/k)*J(J+1), J=3.
    """
    W = np.array(W_Kkms, dtype=float)

    # Constants and units
    nu = (nu_GHz * 1e9) * u.Hz
    Aul = A_ul / u.s
    Tex = Tex_K * u.K
    Eu = Eu_over_k * k_B

    # Rotational constant B via Eu = hB J(J+1), for J=3
    J = 3
    B = (Eu / (h * (J*(J+1))))  # in Hz

    # Partition function approximation for linear rotor
    Q = (k_B * Tex / (h * B)).decompose().value  # dimensionless

    # Prefactor
    pref = (8*np.pi * k_B * nu**2) / (h * c**3 * Aul)
    pref = pref.decompose().value  # in SI: (K^-1 * (m/s)^-1) etc, handled below

    # The Planck correction term
    x = (h * nu / (k_B * Tex)).decompose().value
    planck_corr = 1.0 / (1.0 - np.exp(-x))

    # Convert W from K km/s to K m/s for SI consistency
    W_SI = W * 1e3  # km/s -> m/s

    # Ntot in m^-2 then to cm^-2
    N_m2 = pref * (Q/gu) * np.exp(Eu_over_k / Tex_K) * planck_corr * W_SI
    N_cm2 = N_m2 * 1e-4

    return N_cm2

# ----------------------------
# Mass within a fixed physical aperture
# ----------------------------

def mass_within_aperture_from_mom0(
    fits_path,
    distance_pc,
    radius_au=6000.0,
    Tex_list=(10,20,30,40),
    X_c18o=1.7e-7,      # C18O/H2 abundance (user-set; varies in reality!)
    mu_gas=2.8,         # mean molecular weight per H2 including He
    center_ra_dec=None, # (ra_deg, dec_deg) OR None -> use peak pixel
    ext=0
):
    with fits.open(fits_path) as hdul:
        hdu = hdul[ext]
        Wmap = np.array(hdu.data, dtype=float)
        header = hdu.header.copy()

    # Squeeze singleton dims
    while Wmap.ndim > 2 and Wmap.shape[0] == 1:
        Wmap = Wmap[0]
    if Wmap.ndim != 2:
        raise ValueError(f"Expected 2D moment-0 map; got {Wmap.shape}")

    ny, nx = Wmap.shape
    wcs = WCS(header)

    # Pixel scale and pixel physical area
    pixscale_arcsec = pixel_scale_arcsec_per_pix(header)
    pixscale_rad = (pixscale_arcsec * u.arcsec).to(u.rad).value
    d_cm = (distance_pc * u.pc).to(u.cm).value
    A_pix_cm2 = (d_cm**2) * (pixscale_rad**2)

    # Aperture radius in pixels
    theta_arcsec = angular_radius_arcsec(radius_au, distance_pc)
    r_pix = theta_arcsec / pixscale_arcsec

    # Center
    if center_ra_dec is None:
        # peak pixel center (simple default)
        idx = np.nanargmax(Wmap)
        y0, x0 = np.unravel_index(idx, Wmap.shape)
        x0, y0 = float(x0), float(y0)
    else:
        ra_deg, dec_deg = center_ra_dec
        sc = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
        x0, y0 = wcs.world_to_pixel(sc)
        x0, y0 = float(x0), float(y0)

    mask = aperture_mask(ny, nx, x0, y0, r_pix)
    Wvals = Wmap[mask]
    Wvals = Wvals[~np.isnan(Wvals)]
    if Wvals.size == 0:
        raise RuntimeError("No valid pixels in aperture.")

    results = {}
    for Tex in Tex_list:
        N_c18o = c18o32_Ntot_from_W(Wvals, Tex_K=Tex)  # cm^-2 per pixel
        # Total H2 molecules in aperture: sum( N(H2) * A_pix )
        N_h2 = N_c18o / X_c18o
        M_g = mu_gas * m_p.to(u.g).value * np.sum(N_h2 * A_pix_cm2)
        M_msun = (M_g * u.g).to(u.Msun).value
        results[Tex] = M_msun

    return {
        "Mgas_Msun_by_Tex": results,
        "aperture_radius_arcsec": theta_arcsec,
        "aperture_radius_pix": r_pix,
        "n_pix": int(Wvals.size),
        "center_pix": (x0, y0),
        "X_c18o_used": X_c18o
    }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":

    source_name = 'UYAur'
    molecule = "C18O"
    mom0_fits = "moment_maps/"+source_name+"/"+source_name+"_"+molecule+"_mom0.fits"
    d_pc = 130.0

    simbad_name = find_simbad_source_in_file(file_name='text_files/names_to_simbad_names.txt', search_word=source_name)
    skycoord_object = get_icrs_coordinates(simbad_name)

    print(skycoord_object.ra, skycoord_object.dec)

    # Option B: specify a center in sky coords (deg)
    # out = fixed_physical_aperture_integrate_mom0(
    #     mom0_fits, distance_pc=d_pc, radius_au=6000, center_ra_dec=(skycoord_object.ra.deg,skycoord_object.dec.deg)
    # )

    # print("Aperture radius:", out["theta_arcsec"], "arcsec  (", out["r_pix"], "pix )")
    # print("Sum within aperture:", out["sum_Kkms_arcsec2"], "K km/s arcsec^2")
    # print("Mean within aperture:", out["mean_Kkms"], "K km/s")
    # print("Area:", out["area_arcsec2"], "arcsec^2")
    # print("Center (x,y):", out["center_pix"])

    out = upper_limits_from_cube_rms(
        sigma_chan_K=0.09,  # K per channel
        dv_kms=0.2,  # km/s per channel
        DeltaV_kms=1.0,  # assumed integration window
        beam_fwhm_arcsec=15,  # arcsec
        distance_pc=130,
        R_au=6000,
        nsigma=3
    )
    for k, v in out.items():
        print(k, v)

    ####
    #### Mass calculation
    ####


    # out = mass_within_aperture_from_mom0(
    #     fits_path=mom0_fits,
    #     distance_pc=d_pc,
    #     radius_au=6000.0,
    #     Tex_list=(10,20,30,40),
    #     X_c18o=2.8e-7,          # adjust for your region if needed
    #     center_ra_dec=None      # or (ra_deg, dec_deg)
    # )
    # print("Aperture radius (arcsec):", out["aperture_radius_arcsec"])
    # print("Masses (Msun) by Tex:", out["Mgas_Msun_by_Tex"])
    # print("Abundance used X(C18O)=", out["X_c18o_used"])