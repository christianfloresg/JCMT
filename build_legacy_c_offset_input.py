#!/usr/bin/env python3
"""Build the editable input table for the legacy-C/peak-offset analysis.

The published/legacy c-factor_gaussian values are read without alteration.
Peak offsets are measured independently from the moment-zero maps.  Candidate
peaks are local maxima above 3 sigma within 60 arcsec; the comparison footprint
has radius one half-beam, and the nearest significant candidate is selected.
The global maximum and every candidate are retained for audit and manual edits.

The Gaussian fit performed here is used only to draw the legacy aperture in
the diagnostic figures.  Its center is fixed exactly at the supplied YSO
coordinate; it does not replace the legacy concentration values.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import maximum_filter
from scipy.optimize import least_squares


BEAM = {"HCO+": 14.1, "C18O": 15.27}

# source, display name, map source, ICRS position
SOURCES = [
    ("IRAS03260+3111", "CON 3B", "IRAS03260+3111", "03:29:10.40 +31:21:59.3"),
    ("IRAS03301+3111", "IRAS 03301+3111", "IRAS03301+3111", "03:33:12.84 +31:21:24.1"),
    ("IRAS04108+2803", "IRAS 04108+2803 B", "IRAS04108+2803", "04:13:54.72 +28:11:32.9"),
    ("IRAS04113+2758S", "MHO 2", "IRAS04113+2758S", "04:14:26.40 +28:05:59.6"),
    ("FPTau", "FP Tau", "FPTau", "04:14:47.30 +26:46:26.4"),
    ("IRAS04181+2655M", "IRAS 04181+2654 B", "IRAS04181+2655M", "04:21:10.38 +27:01:37.3"),
    ("IRAS04181+2655S", "IRAS 04181+2654", "IRAS04181+2655S", "04:21:11.46 +27:01:09.4"),
    ("T-Tauri", "T Tauri North", "T-Tauri", "04:21:59.43 +19:32:06.4"),
    ("DG-Tau", "DG Tau", "DG-Tau", "04:27:04.69 +26:06:16.1"),
    ("GV_Tau", "GV Tau", "GV_Tau", "04:29:23.73 +24:33:00.2"),
    ("HK-Tau", "HK Tau", "HK-Tau", "04:31:50.57 +24:24:17.8"),
    ("Haro6-13", "Haro 6-13", "Haro6-13", "04:32:15.41 +24:28:59.6"),
    ("IRAS04295+2251", "IRAS 04295+2251", "IRAS04295+2251", "04:32:32.05 +22:57:26.7"),
    ("Haro6-28", "Haro 6-28", "Haro6-28", "04:35:56.85 +22:54:36.1"),
    ("Haro6-33", "Haro 6-33", "Haro6-33", "04:41:38.82 +25:56:26.7"),
    ("UYAur", "UY Aur A", "UYAur", "04:51:47.38 +30:47:13.6"),
    ("IRAS04489+3042", "IRAS 04489+3042", "IRAS04489+3042", "04:52:06.69 +30:47:17.0"),
    ("V347_Aur", "V347 Aur", "V347_Aur", "04:56:57.02 +51:30:50.8"),
    ("IRAS04591-0856", "IRAS 04591-0856", "IRAS04591-0856", "05:01:29.64 -08:52:16.9"),
    ("IRAS05379-0758", "IRAS 05379-0758(2)", "IRAS05379-0758", "05:40:20.31 -07:56:24.8"),
    ("IRAS05555-1405", "IRAS 05555-1405(S)", "IRAS05555-1405", "05:57:49.18 -14:06:08.0"),
    ("DoAr25", "DoAr 25", "DoAr25", "16:26:23.69 -24:43:13.8"),
    ("SR24", "SR 24 S", "SR24", "16:26:58.51 -24:45:36.7"),
    ("GY92-235", "GY92 235", "GY92-235", "16:27:13.82 -24:43:31.6"),
    ("WLY2-42", "WLY 2-42", "WLY2-42", "16:27:21.47 -24:41:43.1"),
    ("Elia32", "Elia 2-32", "Elia32", "16:27:28.45 -24:27:21.0"),
    ("Elia33", "Elia 2-33", "Elia33", "16:27:30.18 -24:27:43.3"),
    ("YLW58", "YLW 58", "YLW58", "16:28:16.51 -24:36:58.1"),
    ("DoAr43", "DoAr 43", "DoAr43", "16:31:30.88 -24:24:39.9"),
    ("OphIRS63", "IRAS 16285-2355", "OphIRS63", "16:31:35.65 -24:01:29.5"),
    ("EC92", "EC92 92", "EC92", "18:29:57.85 +01:12:51.4"),
    ("IRS5", "TS84 IRS 5", "IRS5", "19:01:48.06 -36:57:22.0"),
    ("IRAS19247+2238(1)", "IRAS 19247+2238 (1)", "IRAS19247+2238", "19:26:51.33 +22:45:13.4"),
    ("IRAS19247+2238(2)", "IRAS 19247+2238 (2)", "IRAS19247+2238", "19:26:51.62 +22:45:03.3"),
]


def fnum(value):
    try:
        ans = float(value)
        return ans if math.isfinite(ans) else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_two_column(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if p and not p[0].startswith("#") and len(p) >= 2:
            value = fnum(p[1])
            if math.isfinite(value):
                out[p[0]] = value
    return out


def read_legacy_c(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if p and not p[0].startswith("#") and len(p) >= 5:
            out[p[0]] = fnum(p[4])
    return out


def read_spectrum(path: Path) -> dict[str, tuple[float, float, float]]:
    out = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if p and not p[0].startswith("#") and len(p) >= 11:
            # moment-zero noise estimate, YSO-centered integral, uncertainty
            noise = fnum(p[1]) * math.sqrt(0.2 * 6.0 * abs(fnum(p[8])))
            out[p[0]] = (noise, fnum(p[9]), fnum(p[10]))
    return out


def gaussian_fixed_center(data, x0, y0, pix_x, pix_y):
    """Return a YSO-fixed elliptical Gaussian radius for map visualization."""
    yy, xx = np.mgrid[: data.shape[0], : data.shape[1]]
    valid = np.isfinite(data)
    if valid.sum() < 20 or not (0 <= x0 < data.shape[1] and 0 <= y0 < data.shape[0]):
        return {"fit_status": "failed", "fwhm_radius_arcsec": math.nan,
                "fit_x_pix": x0, "fit_y_pix": y0, "fit_r2": math.nan}
    xv, yv, zv = xx[valid], yy[valid], data[valid]
    max_sigma = max(10.0, max(data.shape) / 2.0)
    p0 = np.array([max(float(np.nanmax(data)) * 0.5, 1e-6), 3.4, 3.4, 0.0])

    def model(p):
        amp, sx, sy, theta = p
        ct, st = math.cos(theta), math.sin(theta)
        dx, dy = xv - x0, yv - y0
        xr, yr = ct * dx + st * dy, -st * dx + ct * dy
        return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

    result = least_squares(lambda p: model(p) - zv, p0,
                           bounds=([0.0, 3.4, 3.4, -math.pi / 2],
                                   [np.inf, max_sigma, max_sigma, math.pi / 2]),
                           max_nfev=3000)
    amp, sx, sy, theta = result.x
    pred = model(result.x)
    sse = float(np.sum((zv - pred) ** 2))
    sst = float(np.sum((zv - np.mean(zv)) ** 2))
    fwhm_x = 2.354820045 * sx * pix_x
    fwhm_y = 2.354820045 * sy * pix_y
    return {
        "fit_status": "ok" if result.success else "failed",
        "fwhm_radius_arcsec": math.sqrt(fwhm_x * fwhm_y),
        "fit_x_pix": x0, "fit_y_pix": y0,
        "fit_r2": 1.0 - sse / sst if sst > 0 else math.nan,
        "fit_fwhm_x_arcsec": fwhm_x, "fit_fwhm_y_arcsec": fwhm_y,
        "fit_theta_rad": theta, "fit_amplitude": amp,
    }


def peak_candidates(data, xsrc, ysrc, pix_x, pix_y, beam, noise, search_radius=60.0):
    yy, xx = np.mgrid[: data.shape[0], : data.shape[1]]
    rr = np.hypot((xx - xsrc) * pix_x, (yy - ysrc) * pix_y)
    clean = np.where(np.isfinite(data), data, -np.inf)
    pix = math.sqrt(pix_x * pix_y)
    rad = max(1, int(math.ceil((beam / 2.0) / pix)))
    fy, fx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    footprint = ((fx * pix_x) ** 2 + (fy * pix_y) ** 2) <= (beam / 2.0) ** 2
    local_max = maximum_filter(clean, footprint=footprint, mode="constant", cval=-np.inf)
    significant = (np.isfinite(data) & np.isclose(data, local_max, rtol=0, atol=1e-12)
                   & (rr <= search_radius) & (data >= 3.0 * noise))
    candidates = []
    for y, x in np.argwhere(significant):
        candidates.append({"x": int(x), "y": int(y), "offset": float(rr[y, x]),
                           "value": float(data[y, x]), "snr": float(data[y, x] / noise)})
    candidates.sort(key=lambda p: (p["offset"], -p["value"]))
    wide = (rr <= search_radius) & np.isfinite(data)
    if not np.any(wide):
        return candidates, None
    idx = np.nanargmax(np.where(wide, data, np.nan))
    yg, xg = np.unravel_index(idx, data.shape)
    global_peak = {"x": int(xg), "y": int(yg), "offset": float(rr[yg, xg]),
                   "value": float(data[yg, xg]), "snr": float(data[yg, xg] / noise)}
    return candidates, global_peak


def usable_noise(data, tabulated_noise):
    """Use the spectrum-derived moment-0 noise, with a robust map fallback."""
    if math.isfinite(tabulated_noise) and tabulated_noise > 0:
        return tabulated_noise, "spectrum_window"
    finite = data[np.isfinite(data)]
    if finite.size:
        median = float(np.median(finite))
        robust = 1.4826 * float(np.median(np.abs(finite - median)))
        if math.isfinite(robust) and robust > 0:
            return robust, "map_MAD_fallback"
    return math.nan, "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--simple-out", type=Path,
                    help="Optional one-row-per-source table containing only classification inputs")
    args = ap.parse_args()
    repo = args.repo.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    distances = read_two_column(repo / "text_files/source_distances_updated.txt")
    legacy = {
        "HCO+": read_legacy_c(repo / "text_files/concentrations_5arsrc_max_new_rad_2025-10-31_12_HCO+.txt"),
        "C18O": read_legacy_c(repo / "text_files/concentrations_5arsrc_max_new_rad_2025-10-31_13_C18O.txt"),
    }
    spectra = {m: read_spectrum(repo / f"spectrum_parameters_{m}.txt") for m in ("HCO+", "C18O")}
    rows = []
    for source, display, map_source, position in SOURCES:
        coord = SkyCoord(position, unit=("hourangle", "deg"), frame="icrs")
        distance = distances.get(map_source, distances.get(source, math.nan))
        for molecule in ("HCO+", "C18O"):
            row = {
                "source": source, "display_name": display, "map_source": map_source,
                "molecule": molecule, "ra_dec_icrs": position, "distance_pc": distance,
                "legacy_C": legacy[molecule].get(map_source, math.nan),
                "beam_arcsec": BEAM[molecule], "peak_threshold_snr": 3.0,
                "association_threshold_arcsec": 15.0,
            }
            noise, w_yso, w_err = spectra[molecule].get(map_source, (math.nan, math.nan, math.nan))
            row.update(tabulated_moment0_noise=noise, W_YSO_Kkms=w_yso, W_YSO_err_Kkms=w_err)
            path = repo / "moment_maps" / map_source / f"{map_source}_{molecule}_mom0.fits"
            row["map_path"] = str(path.relative_to(repo))
            try:
                with fits.open(path) as hdul:
                    data = np.squeeze(hdul[0].data).astype(float)
                    wcs = WCS(hdul[0].header).celestial
                xsrc, ysrc = wcs.world_to_pixel(coord)
                pix_x = abs(float(wcs.wcs.cdelt[0])) * 3600.0
                pix_y = abs(float(wcs.wcs.cdelt[1])) * 3600.0
                row.update(xsrc_pix=xsrc, ysrc_pix=ysrc, pixscale_x_arcsec=pix_x,
                           pixscale_y_arcsec=pix_y)
                fit = gaussian_fixed_center(data, xsrc, ysrc, pix_x, pix_y)
                row.update(**fit)
                noise, noise_method = usable_noise(data, noise)
                row.update(moment0_noise=noise, moment0_noise_method=noise_method)
                if math.isfinite(noise):
                    candidates, global_peak = peak_candidates(
                        data, xsrc, ysrc, pix_x, pix_y, BEAM[molecule], noise)
                else:
                    candidates, global_peak = [], None
                selected = candidates[0] if candidates else None
                row.update(candidate_count=len(candidates),
                           candidate_offsets_arcsec=";".join(f"{p['offset']:.3f}" for p in candidates),
                           candidate_snrs=";".join(f"{p['snr']:.3f}" for p in candidates),
                           candidate_x_pix=";".join(str(p["x"]) for p in candidates),
                           candidate_y_pix=";".join(str(p["y"]) for p in candidates))
                if selected:
                    row.update(selected_peak_x_pix=selected["x"], selected_peak_y_pix=selected["y"],
                               selected_peak_offset_arcsec=selected["offset"],
                               selected_peak_offset_au=selected["offset"] * distance,
                               selected_peak_offset_pc=selected["offset"] * distance / 206265.0,
                               selected_peak_snr=selected["snr"], selected_peak_value=selected["value"],
                               selection_rule="nearest_local_max_gt3sigma")
                if global_peak:
                    row.update(global_peak_x_pix=global_peak["x"], global_peak_y_pix=global_peak["y"],
                               global_peak_offset_arcsec=global_peak["offset"],
                               global_peak_snr=global_peak["snr"], global_peak_value=global_peak["value"])
                row["measurement_status"] = ("ok" if selected else
                                               ("no_significant_local_peak" if math.isfinite(noise)
                                                else "noise_unavailable"))
            except Exception as exc:
                row.update(measurement_status="failed", measurement_error=str(exc))
            rows.append(row)

    fields = sorted({key for row in rows for key in row})
    preferred = ["source", "display_name", "map_source", "molecule", "ra_dec_icrs", "distance_pc",
                 "legacy_C", "W_YSO_Kkms", "W_YSO_err_Kkms", "selected_peak_offset_arcsec",
                 "selected_peak_offset_au", "selected_peak_offset_pc", "selected_peak_snr",
                 "global_peak_offset_arcsec", "global_peak_snr", "candidate_count",
                 "candidate_offsets_arcsec", "candidate_snrs", "fwhm_radius_arcsec",
                 "fit_fwhm_x_arcsec", "fit_fwhm_y_arcsec", "fit_r2", "fit_status",
                 "measurement_status", "selection_rule", "map_path"]
    fields = preferred + [f for f in fields if f not in preferred]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if args.simple_out:
        args.simple_out.parent.mkdir(parents=True, exist_ok=True)
        by = {(r["source"], r["molecule"]): r for r in rows}
        simple = []
        for source, display, map_source, _ in SOURCES:
            h, c = by[(source, "HCO+")], by[(source, "C18O")]
            distance = fnum(h.get("distance_pc"))
            simple.append({
                "source": source,
                "display_name": display,
                "distance_pc": distance,
                "C_HCO": h.get("legacy_C", math.nan),
                "C_C18O": c.get("legacy_C", math.nan),
                "W_HCO_Kkms": h.get("W_YSO_Kkms", math.nan),
                "W_HCO_threshold_Kkms": 0.4 * 140.0 / distance if distance > 0 else math.nan,
                "HCO_peak_significant": bool(h.get("selected_peak_offset_arcsec") not in (None, "")),
                "HCO_peak_offset_arcsec": h.get("selected_peak_offset_arcsec", math.nan),
                "C18O_peak_significant": bool(c.get("selected_peak_offset_arcsec") not in (None, "")),
                "C18O_peak_offset_arcsec": c.get("selected_peak_offset_arcsec", math.nan),
            })
        with args.simple_out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(simple[0]))
            writer.writeheader()
            writer.writerows(simple)


if __name__ == "__main__":
    main()
