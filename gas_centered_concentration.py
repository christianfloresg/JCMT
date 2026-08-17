#!/usr/bin/env python3
"""Recompute JCMT molecular-line concentration factors with fit diagnostics.

This is a read-only audit of the JCMT repository.  It deliberately does not
import concentration_factors.py, because that module queries SIMBAD at run
time and its Gaussian helper does not return the diagnostics needed here.
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
from photutils.aperture import CircularAperture
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares


REPO = Path(__file__).resolve().parent

# Coordinates supplied by the user (ICRS/J2000).  IRAS19247+2238 uses
# component 1; the second component is reported separately in the notes.
COORDS = {
    "IRAS03260+3111": "03:29:10.40 +31:21:59.3",
    "IRAS03301+3111": "03:33:12.84 +31:21:24.1",
    "IRAS04108+2803": "04:13:54.72 +28:11:32.9",
    "IRAS04113+2758S": "04:14:26.40 +28:05:59.6",
    "FPTau": "04:14:47.30 +26:46:26.4",
    "IRAS04181+2655M": "04:21:10.38 +27:01:37.3",
    "IRAS04181+2655S": "04:21:11.46 +27:01:09.4",
    "T-Tauri": "04:21:59.43 +19:32:06.4",
    "DG-Tau": "04:27:04.69 +26:06:16.1",
    "GV_Tau": "04:29:23.73 +24:33:00.2",
    "HK-Tau": "04:31:50.57 +24:24:17.8",
    "Haro6-13": "04:32:15.41 +24:28:59.6",
    "IRAS04295+2251": "04:32:32.05 +22:57:26.7",
    "Haro6-28": "04:35:56.85 +22:54:36.1",
    "Haro6-33": "04:41:38.82 +25:56:26.7",
    "UYAur": "04:51:47.38 +30:47:13.6",
    "IRAS04489+3042": "04:52:06.69 +30:47:17.0",
    "V347_Aur": "04:56:57.02 +51:30:50.8",
    "IRAS04591-0856": "05:01:29.64 -08:52:16.9",
    "IRAS05379-0758": "05:40:20.31 -07:56:24.8",
    "IRAS05555-1405": "05:57:49.18 -14:06:08.0",
    "DoAr25": "16:26:23.69 -24:43:13.8",
    "SR24": "16:26:58.51 -24:45:36.7",
    "GY92-235": "16:27:13.82 -24:43:31.6",
    "WLY2-42": "16:27:21.47 -24:41:43.1",
    "Elia32": "16:27:28.45 -24:27:21.0",
    "Elia33": "16:27:30.18 -24:27:43.3",
    "YLW58": "16:28:16.51 -24:36:58.1",
    "DoAr43": "16:31:30.88 -24:24:39.9",
    "OphIRS63": "16:31:35.65 -24:01:29.5",
    "EC92": "18:29:57.85 +01:12:51.4",
    "IRS5": "19:01:48.06 -36:57:22.0",
    "IRAS19247+2238(1)": "19:26:51.33 +22:45:13.4",
    "IRAS19247+2238(2)": "19:26:51.62 +22:45:03.3",
}

# Two entries in the observing table share one map directory/product.
MAP_SOURCE = {
    "IRAS19247+2238(1)": "IRAS19247+2238",
    "IRAS19247+2238(2)": "IRAS19247+2238",
}

# Adopted distances used only to translate the angular fitted size to AU.
# They never enter C.  status=provisional retains the former value because a
# defensible source/cloud distance was not established in the distance audit.
DIST = {
    "IRAS03260+3111": (299.0, 15.0, "NGC1333 association"),
    "IRAS03301+3111": (291.0, 8.0, "B1 association"),
    "IRAS04108+2803": (129.9, 0.4, "L1495 association"),
    "IRAS04113+2758S": (132.4, 4.0, "Gaia DR2 source"),
    "FPTau": (130.0, 5.0, "Taurus association; verify Gaia source"),
    "IRAS04181+2655M": (128.5, 1.6, "B215 association"),
    "IRAS04181+2655S": (128.5, 1.6, "B215 association"),
    "T-Tauri": (146.7, 0.6, "VLBA system parallax"),
    "DG-Tau": (121.2, 1.2, "Gaia DR2 source"),
    "GV_Tau": (180.0, 17.0, "L1524 association"),
    "HK-Tau": (133.0, 5.0, "Taurus association; verify Gaia source"),
    "Haro6-13": (130.0, 5.0, "Taurus association; verify Gaia source"),
    "IRAS04295+2251": (160.0, 5.0, "Taurus filament association"),
    "Haro6-28": (160.0, 5.0, "Taurus filament association"),
    "Haro6-33": (130.0, 5.0, "Taurus association; verify Gaia source"),
    "UYAur": (155.0, 10.0, "Taurus association; verify Gaia source"),
    "IRAS04489+3042": (155.0, 10.0, "Taurus association; verify Gaia source"),
    "V347_Aur": (209.0, 10.0, "Gaia-based; high RUWE"),
    "IRAS04591-0856": (210.0, 20.0, "MBM21 association"),
    "IRAS05379-0758": (400.0, 40.0, "provisional former value"),
    "IRAS05555-1405": (400.0, 40.0, "provisional former value"),
    "DoAr25": (138.0, 2.0, "Gaia source"),
    "SR24": (114.4, 4.8, "Gaia DR2 system"),
    "GY92-235": (137.3, 1.2, "L1688 association"),
    "WLY2-42": (137.3, 1.2, "L1688 association"),
    "Elia32": (137.3, 1.2, "L1688 association"),
    "Elia33": (137.3, 1.2, "L1688 association"),
    "YLW58": (137.3, 1.2, "L1688 association"),
    "DoAr43": (137.3, 1.2, "L1688 association"),
    "OphIRS63": (144.2, 1.3, "Ophiuchus eastern streamer association"),
    "EC92": (436.0, 9.2, "Serpens/EC95 VLBA association"),
    "IRS5": (149.4, 0.4, "Corona Australis association"),
    "IRAS19247+2238(1)": (300.0, 30.0, "provisional former value; no association secured"),
    "IRAS19247+2238(2)": (300.0, 30.0, "provisional former value; no association secured"),
}

BEAM = {"HCO+": 14.1, "C18O": 15.27}
OLD_FILES = {
    "HCO+": REPO / "text_files/concentrations_5arsrc_max_new_rad_2025-10-31_12_HCO+.txt",
    "C18O": REPO / "text_files/concentrations_5arsrc_max_new_rad_2025-10-31_13_C18O.txt",
}


def read_old(path: Path) -> dict[str, float]:
    ans = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        try:
            ans[p[0]] = float(p[4])
        except (ValueError, IndexError):
            ans[p[0]] = math.nan
    return ans


def read_noise(molecule: str) -> dict[str, float]:
    path = REPO / f"spectrum_parameters_{molecule}.txt"
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        # image RMS per channel, velocity sigma, and dv=0.2 km/s.
        try:
            out[p[0]] = float(p[1]) * math.sqrt(0.2 * 6.0 * abs(float(p[8])))
        except (ValueError, IndexError):
            pass
    return out


def circular_model(p, x, y):
    amp, x0, y0, sigma, bg = p
    return bg + amp * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


def aperture_sum(data, x0, y0, r_pix):
    finite = np.isfinite(data)
    clean = np.where(finite, data, 0.0)
    ap = CircularAperture((x0, y0), r=r_pix)
    mask = ap.to_mask(method="exact")
    cut = mask.multiply(clean)
    coverage = mask.multiply(finite.astype(float))
    if cut is None or coverage is None:
        return math.nan, 0.0
    return float(np.sum(cut)), float(np.sum(coverage) / np.sum(mask.data))


def fit_one(data, wcs, coord, molecule, noise, rng=None):
    ny, nx = data.shape
    xsrc, ysrc = wcs.world_to_pixel(coord)
    pix_x = abs(float(wcs.wcs.cdelt[0])) * 3600.0
    pix_y = abs(float(wcs.wcs.cdelt[1])) * 3600.0
    pix = math.sqrt(pix_x * pix_y)
    beam = BEAM[molecule]
    beam_sigma_pix = beam / 2.354820045 / pix

    yy, xx = np.mgrid[:ny, :nx]
    rsrc = np.hypot((xx - xsrc) * pix_x, (yy - ysrc) * pix_y)
    # Identify both an on-source component and the brightest component in the
    # wider field.  A significant on-source component is fitted preferentially;
    # otherwise the wide-field component is retained to diagnose confusion.
    wide_peak_region = (rsrc <= 60.0) & np.isfinite(data)
    onsource_region = (rsrc <= 14.0) & np.isfinite(data)
    if wide_peak_region.sum() == 0 or onsource_region.sum() == 0:
        raise ValueError("insufficient finite map coverage near source")

    global_idx = np.nanargmax(np.where(wide_peak_region, data, np.nan))
    yg, xg = np.unravel_index(global_idx, data.shape)
    global_peak = float(data[yg, xg])
    global_offset = math.hypot((xg - xsrc) * pix_x, (yg - ysrc) * pix_y)
    onsource_idx = np.nanargmax(np.where(onsource_region, data, np.nan))
    yo, xo = np.unravel_index(onsource_idx, data.shape)
    onsource_peak = float(data[yo, xo])
    onsource_peak_snr = onsource_peak / noise if noise > 0 else math.nan
    global_peak_snr = global_peak / noise if noise > 0 else math.nan
    has_onsource_peak = bool(np.isfinite(onsource_peak_snr) and onsource_peak_snr >= 3)
    xp, yp = (xo, yo) if has_onsource_peak else (xg, yg)
    peak_offset = math.hypot((xp - xsrc) * pix_x, (yp - ysrc) * pix_y)
    rpeak = np.hypot((xx - xp) * pix_x, (yy - yp) * pix_y)
    fit_region = (rpeak <= 45.0) & np.isfinite(data)
    if fit_region.sum() < 20:
        raise ValueError("insufficient finite map coverage around gas peak")
    bg0 = float(np.nanmedian(data[fit_region]))
    amp0 = max(float(data[yp, xp] - bg0), max(noise, 1e-4))
    p0 = np.array([amp0, xp, yp, max(beam_sigma_pix * 1.15, 2.0), bg0])

    # Refine the gas center locally; the bounds are relative to the gas peak,
    # not the infrared source.
    shift_x = beam / pix_x
    shift_y = beam / pix_y
    lower = np.array([0.0, max(0, xp - shift_x), max(0, yp - shift_y),
                      beam_sigma_pix, -np.inf])
    upper = np.array([np.inf, min(nx - 1, xp + shift_x), min(ny - 1, yp + shift_y),
                      45.0 / 2.354820045 / pix, np.inf])
    p0 = np.minimum(np.maximum(p0, lower + 1e-7), upper - 1e-7)
    xv, yv, zv = xx[fit_region], yy[fit_region], data[fit_region]

    def resid(p):
        return (circular_model(p, xv, yv) - zv) / max(noise, 1e-4)

    result = least_squares(resid, p0, bounds=(lower, upper), max_nfev=2500)
    amp, xfit, yfit, sigma, bg = result.x
    fwhm = 2.354820045 * sigma * pix
    gaussian_offset = math.hypot((xfit - xsrc) * pix_x, (yfit - ysrc) * pix_y)

    # Peak integrated intensity is measured around the fitted gas peak.  The
    # source-to-peak displacement remains a separate association diagnostic.
    rfit = np.hypot((xx - xfit) * pix_x, (yy - yfit) * pix_y)
    local = (rfit <= beam / 2.0) & np.isfinite(data)
    peak = float(np.nanmax(data[local])) if np.any(local) else math.nan

    raw_sum, coverage = aperture_sum(data, xfit, yfit, fwhm / pix)
    bgsub_sum, _ = aperture_sum(data - bg, xfit, yfit, fwhm / pix)
    pix_per_beam = (math.pi / (4.0 * math.log(2.0)) * beam**2) / (pix_x * pix_y)
    total = raw_sum / pix_per_beam
    total_bgsub = bgsub_sum / pix_per_beam
    if np.isfinite(peak) and peak > 0 and np.isfinite(total) and fwhm > 0:
        c = 1.0 - (math.pi / (4.0 * math.log(2.0)) * beam**2) / (math.pi * fwhm**2) * total / peak
        peak_bgsub = peak - bg
        c_bgsub = (1.0 - (math.pi / (4.0 * math.log(2.0)) * beam**2) /
                   (math.pi * fwhm**2) * total_bgsub / peak_bgsub) if peak_bgsub > 0 else math.nan
    else:
        c = math.nan
        c_bgsub = math.nan

    model = circular_model(result.x, xv, yv)
    sse = float(np.sum((zv - model) ** 2))
    sst = float(np.sum((zv - np.mean(zv)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else math.nan
    at_center_bound = any(abs(result.x[i] - lower[i]) < 1e-3 or abs(result.x[i] - upper[i]) < 1e-3 for i in (1, 2))
    at_width_bound = abs(sigma - lower[3]) < 1e-3 or abs(sigma - upper[3]) < 1e-3
    return {
        "c": c, "c_bgsub": c_bgsub, "fwhm": fwhm,
        "offset": peak_offset, "gaussian_offset": gaussian_offset,
        "global_offset": global_offset, "global_peak": global_peak,
        "global_peak_snr": global_peak_snr,
        "onsource_peak": onsource_peak, "onsource_peak_snr": onsource_peak_snr,
        "has_onsource_peak": has_onsource_peak,
        "selected_component": "onsource" if has_onsource_peak else "wide_field",
        "peak_x_pix": float(xp), "peak_y_pix": float(yp),
        "fit_x_pix": float(xfit), "fit_y_pix": float(yfit),
        "peak": peak, "total": total, "total_bgsub": total_bgsub,
        "coverage": coverage, "r2": r2, "bg": bg,
        "peak_snr": peak / noise if noise > 0 else math.nan,
        "center_bound": at_center_bound, "width_bound": at_width_bound,
    }


def analyze_map(map_source, molecule, coord, noise, nmc, seed):
    path = REPO / "moment_maps" / map_source / f"{map_source}_{molecule}_mom0.fits"
    with fits.open(path) as hdul:
        data = np.squeeze(hdul[0].data).astype(float)
        wcs = WCS(hdul[0].header).celestial
    base = fit_one(data, wcs, coord, molecule, noise)

    draws = []
    rng = np.random.default_rng(seed)
    if nmc > 0 and np.isfinite(noise) and noise > 0:
        pix = math.sqrt(abs(wcs.wcs.cdelt[0] * wcs.wcs.cdelt[1])) * 3600.0
        corr_sigma = BEAM[molecule] / 2.354820045 / pix
        for _ in range(nmc):
            white = rng.normal(size=data.shape)
            corr = gaussian_filter(white, corr_sigma, mode="reflect")
            sd = np.nanstd(corr)
            perturbed = data + (corr / sd * noise if sd > 0 else 0.0)
            try:
                draws.append(fit_one(perturbed, wcs, coord, molecule, noise))
            except Exception:
                continue

    for key in ("c", "c_bgsub", "fwhm", "offset", "gaussian_offset"):
        vals = np.array([
            d[key] for d in draws
            if np.isfinite(d[key])
            and (key not in ("c", "c_bgsub") or
                 (d["peak_snr"] >= 3 and 0 <= d[key] <= 1 and d["coverage"] >= 0.8))
        ])
        if len(vals) >= max(10, nmc // 3):
            base[key + "_p16"], base[key + "_p84"] = np.percentile(vals, [16, 84])
        else:
            base[key + "_p16"] = base[key + "_p84"] = math.nan
    base["mc_success"] = len(draws)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nmc", type=int, default=100)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    old = {m: read_old(p) for m, p in OLD_FILES.items()}
    noise = {m: read_noise(m) for m in ("HCO+", "C18O")}
    rows = []
    for sidx, (source, pos) in enumerate(COORDS.items()):
        map_source = MAP_SOURCE.get(source, source)
        coord = SkyCoord(pos, unit=("hourangle", "deg"), frame="icrs")
        dpc, edpc, dbasis = DIST[source]
        for midx, molecule in enumerate(("HCO+", "C18O")):
            path = REPO / "moment_maps" / map_source / f"{map_source}_{molecule}_mom0.fits"
            old_c = old[molecule].get(map_source, math.nan)
            if not np.isfinite(old_c):
                old_c = math.nan
            row = {"source": source, "map_source": map_source, "molecule": molecule, "distance_pc": dpc,
                   "distance_err_pc": edpc, "distance_basis": dbasis,
                   "old_c_factor_gaussian": old_c}
            if not path.exists():
                row.update(status="missing_map")
                rows.append(row)
                continue
            try:
                res = analyze_map(map_source, molecule, coord, noise[molecule].get(map_source, math.nan),
                                  args.nmc, 10000 + sidx * 2 + midx)
                row.update(res)
                numeric_valid = (np.isfinite(res["c"]) and 0 <= res["c"] <= 1
                                 and res["peak_snr"] >= 3 and res["coverage"] >= 0.8)
                row["c_validated"] = row["c"] if numeric_valid else math.nan
                row["delta_c"] = (row["c"] - old_c) if np.isfinite(row["c"]) and np.isfinite(old_c) else math.nan
                row["delta_c_validated"] = (row["c_validated"] - old_c) if numeric_valid and np.isfinite(old_c) else math.nan
                row["fwhm_radius_au"] = row["fwhm"] * dpc
                fit_err = 0.5 * (row["fwhm_p84"] - row["fwhm_p16"])
                row["fwhm_radius_err_au"] = math.sqrt((dpc * fit_err) ** 2 + (row["fwhm"] * edpc) ** 2)
                flags = []
                confused = (not res["has_onsource_peak"] and res["global_peak_snr"] >= 3
                            and res["global_offset"] > 20)
                if confused: flags.append("confused_no_onsource_peak_global_gt20")
                elif res["offset"] > 14: flags.append("association_offset_14to20")
                if res["has_onsource_peak"] and res["global_peak_snr"] >= 3 and res["global_offset"] > 20:
                    flags.append("additional_offsource_peak_gt20")
                if res["center_bound"]: flags.append("center_at_fit_bound")
                if res["width_bound"]: flags.append("width_at_fit_bound")
                if res["peak_snr"] < 3: flags.append("peak_snr_lt3")
                if not (0 <= res["c"] <= 1): flags.append("c_outside_0to1")
                if res["coverage"] < 0.8: flags.append("aperture_coverage_lt80pct")
                elif res["coverage"] < 0.95: flags.append("aperture_coverage_80to95pct")
                if res["r2"] < 0.5: flags.append("poor_gaussian_r2")
                if "provisional" in dbasis: flags.append("distance_provisional")
                row["flags"] = ";".join(flags)
                severe = (not numeric_valid or res["center_bound"] or confused)
                row["status"] = "invalid" if not numeric_valid else ("review" if severe or flags else "usable")
            except Exception as exc:
                row.update(status="fit_failed", flags=str(exc))
            rows.append(row)

    fields = sorted({k for r in rows for k in r})
    preferred = ["source", "map_source", "molecule", "status", "flags", "old_c_factor_gaussian", "c",
                 "c_validated", "c_p16", "c_p84", "c_bgsub", "c_bgsub_p16", "c_bgsub_p84",
                 "delta_c", "delta_c_validated", "offset", "offset_p16", "offset_p84",
                 "gaussian_offset", "gaussian_offset_p16", "gaussian_offset_p84",
                 "fwhm", "fwhm_p16", "fwhm_p84", "fwhm_radius_au", "fwhm_radius_err_au",
                 "distance_pc", "distance_err_pc", "distance_basis", "peak", "peak_snr",
                 "total", "coverage", "r2", "bg", "center_bound", "width_bound", "mc_success"]
    fields = preferred + [f for f in fields if f not in preferred]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
