from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import regionprops_table

from .background import BackgroundPatch
from .projections import cell_z_range, max_projection, mean_projection


@dataclass
class CellRow:
    image_name: str
    cell_id: int
    area_um2: float
    area_px: int
    centroid_x: float
    centroid_y: float
    solidity: float
    eccentricity: float
    focus_score: float
    qc_pass: bool
    qc_reason: str
    z_start: int
    z_end: int
    # Per channel per source (mip / zspan) numbers land in this dict:
    intensities: Dict[str, float] = field(default_factory=dict)


def _stats(pixels: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """mean, median, min, max, integrated (sum), stdev"""
    if pixels.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(pixels.mean()),
        float(np.median(pixels)),
        float(pixels.min()),
        float(pixels.max()),
        float(pixels.sum()),
        float(pixels.std(ddof=0)),
    )


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Standard focus metric: variance of the image Laplacian."""
    from scipy.ndimage import laplace
    return float(laplace(gray.astype(np.float32)).var())


def measure_cells(
    image_name: str,
    cell_mask: np.ndarray,                 # (Y, X) int label image
    dapi_mip: np.ndarray,                  # (Y, X) for focus + z-range anchor
    stacks_by_channel: Dict[str, np.ndarray],   # channel -> (Z, Y, X) raw stack
    mips_by_channel: Dict[str, np.ndarray],     # channel -> (Y, X) MIP
    background: BackgroundPatch,
    pixel_size_um: float,
    measure_channels: List[str],
    bg_ring_masks: Optional[Dict[int, np.ndarray]] = None,
) -> List[CellRow]:
    """Measure every labelled cell and return a list of CellRow.

    QC (area/solidity/eccentricity/focus/DAPI) is applied downstream by qc.apply_qc.
    """
    props = regionprops_table(
        cell_mask,
        intensity_image=dapi_mip,
        properties=("label", "area", "centroid", "solidity", "eccentricity",
                    "bbox", "coords"),
    )

    rows: List[CellRow] = []
    n = len(props["label"])
    for i in range(n):
        lbl = int(props["label"][i])
        area_px = int(props["area"][i])
        area_um2 = area_px * (pixel_size_um ** 2)
        cy = float(props["centroid-0"][i])
        cx = float(props["centroid-1"][i])
        sol = float(props["solidity"][i])
        ecc = float(props["eccentricity"][i])

        # DAPI focus on the bbox crop
        r0, c0, r1, c1 = (int(props["bbox-0"][i]), int(props["bbox-1"][i]),
                          int(props["bbox-2"][i]), int(props["bbox-3"][i]))
        crop = dapi_mip[r0:r1, c0:c1]
        focus = variance_of_laplacian(crop)

        # this cell's mask (Y, X)
        this = cell_mask == lbl

        # z-range using DAPI stack
        dapi_stack = stacks_by_channel.get("DAPI")
        if dapi_stack is None:
            dapi_stack = stacks_by_channel.get("dapi")
        if dapi_stack is not None:
            z0, z1 = cell_z_range(dapi_stack, this)
        else:
            # No DAPI stack -> use full range
            any_stack = next(iter(stacks_by_channel.values()))
            z0, z1 = 0, any_stack.shape[0]

        row = CellRow(
            image_name=image_name,
            cell_id=lbl,
            area_um2=area_um2,
            area_px=area_px,
            centroid_x=cx,
            centroid_y=cy,
            solidity=sol,
            eccentricity=ecc,
            focus_score=focus,
            qc_pass=True,
            qc_reason="",
            z_start=z0,
            z_end=z1,
        )

        # Ring background (if configured) may override the shared patch bg per cell
        ring_mask = bg_ring_masks.get(lbl) if bg_ring_masks else None

        for ch in measure_channels:
            mip = mips_by_channel[ch]
            stack = stacks_by_channel[ch]

            # ---- MIP-based measurements ----
            roi_px_mip = mip[this]
            mean_, med_, min_, max_, sum_, sd_ = _stats(roi_px_mip)

            if ring_mask is not None and ring_mask.any():
                bg_pixels_mip = mip[ring_mask]
            else:
                bg_pixels_mip = mip[background.mask]
            bg_mean_mip = float(bg_pixels_mip.mean()) if bg_pixels_mip.size else 0.0
            bg_med_mip = float(np.median(bg_pixels_mip)) if bg_pixels_mip.size else 0.0

            corrected_mean_mip = mean_ - bg_mean_mip
            corrected_int_mip = corrected_mean_mip * area_px
            sob_mip = (mean_ / bg_mean_mip) if bg_mean_mip > 0 else 0.0

            row.intensities.update({
                f"{ch}_mip_roi_mean": mean_,
                f"{ch}_mip_roi_median": med_,
                f"{ch}_mip_roi_min": min_,
                f"{ch}_mip_roi_max": max_,
                f"{ch}_mip_roi_integrated": sum_,
                f"{ch}_mip_roi_stdev": sd_,
                f"{ch}_mip_bg_mean": bg_mean_mip,
                f"{ch}_mip_bg_median": bg_med_mip,
                f"{ch}_mip_corrected_mean": corrected_mean_mip,
                f"{ch}_mip_corrected_integrated": corrected_int_mip,
                f"{ch}_mip_signal_over_bg": sob_mip,
            })

            # ---- z-span mean projection measurements ----
            if stack.shape[0] > 1 and z1 > z0:
                sub = stack[z0:z1]
                zproj = mean_projection(sub)
            else:
                zproj = mip
            roi_px_z = zproj[this]
            zmean, zmed, zmin, zmax, zsum, zsd = _stats(roi_px_z)
            if ring_mask is not None and ring_mask.any():
                bg_pixels_z = zproj[ring_mask]
            else:
                bg_pixels_z = zproj[background.mask]
            zbg_mean = float(bg_pixels_z.mean()) if bg_pixels_z.size else 0.0
            zbg_med = float(np.median(bg_pixels_z)) if bg_pixels_z.size else 0.0
            zcorr_mean = zmean - zbg_mean
            zcorr_int = zcorr_mean * area_px
            zsob = (zmean / zbg_mean) if zbg_mean > 0 else 0.0

            row.intensities.update({
                f"{ch}_zspan_roi_mean": zmean,
                f"{ch}_zspan_roi_median": zmed,
                f"{ch}_zspan_roi_min": zmin,
                f"{ch}_zspan_roi_max": zmax,
                f"{ch}_zspan_roi_integrated": zsum,
                f"{ch}_zspan_roi_stdev": zsd,
                f"{ch}_zspan_bg_mean": zbg_mean,
                f"{ch}_zspan_bg_median": zbg_med,
                f"{ch}_zspan_corrected_mean": zcorr_mean,
                f"{ch}_zspan_corrected_integrated": zcorr_int,
                f"{ch}_zspan_signal_over_bg": zsob,
            })

        rows.append(row)

    return rows
