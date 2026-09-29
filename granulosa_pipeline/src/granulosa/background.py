from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)


@dataclass
class BackgroundPatch:
    """Rectangular background patch defined by (row_start, col_start, size_rows, size_cols)."""
    row: int
    col: int
    rows: int
    cols: int
    mask: np.ndarray                    # boolean array same shape as image
    per_channel_mean: Dict[str, float]
    per_channel_median: Dict[str, float]

    def as_slice(self) -> Tuple[slice, slice]:
        return slice(self.row, self.row + self.rows), slice(self.col, self.col + self.cols)


def _speck_threshold(gfp_yx: np.ndarray, dapi_empty_mask: np.ndarray, percentile: float) -> float:
    vals = gfp_yx[dapi_empty_mask]
    if vals.size == 0:
        return float("inf")
    return float(np.percentile(vals, percentile))


def build_background_pool(
    mip_by_channel: Dict[str, np.ndarray],
    cell_mask: np.ndarray,
    pixel_size_um: float,
    speck_percentile: float,
    cell_buffer_um: float,
    reference_gfp_channel: Optional[str] = "GFP",
) -> np.ndarray:
    """Return a boolean mask of pixels safe to use as background:
       DAPI-empty (via cell mask + buffer) AND below the GFP-speck cutoff.
    """
    cells_bin = cell_mask > 0
    if cell_buffer_um > 0:
        radius_px = max(1, int(round(cell_buffer_um / pixel_size_um)))
        # Euclidean buffer: exclude any pixel within radius_px of a cell.
        edt = ndi.distance_transform_edt(~cells_bin)
        dapi_empty = edt > radius_px
    else:
        dapi_empty = ~cells_bin

    if reference_gfp_channel and reference_gfp_channel in mip_by_channel:
        gfp = mip_by_channel[reference_gfp_channel]
        thr = _speck_threshold(gfp, dapi_empty, speck_percentile)
        pool = dapi_empty & (gfp <= thr)
        logger.debug(
            "background pool: %d / %d pixels (speck threshold %.2f on %s)",
            int(pool.sum()), int(dapi_empty.sum()), thr, reference_gfp_channel,
        )
    else:
        pool = dapi_empty
    return pool


def find_shared_patch(
    pool_mask: np.ndarray,
    target_area_px: int,
    mip_by_channel: Dict[str, np.ndarray],
) -> BackgroundPatch:
    """Pick a square patch of area ~target_area_px whose pixels are all in the pool.

    Strategy: choose a square side = ceil(sqrt(area)); slide with an integral image and
    pick the highest-count square (== fully valid if count == side*side). Among candidates
    with all pixels in the pool, pick the one with the lowest median GFP (or whichever
    reference channel comes first alphabetically) to bias toward truly-empty area.
    """
    side = int(np.ceil(np.sqrt(target_area_px)))
    H, W = pool_mask.shape
    if side >= H or side >= W:
        raise RuntimeError(
            f"Cannot fit a {side}x{side} background patch inside a {H}x{W} image."
        )

    # Integer image for fast rolling window sums.
    ii = np.zeros((H + 1, W + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(pool_mask.astype(np.int64), axis=0), axis=1)
    counts = (
        ii[side:, side:]
        - ii[:-side, side:]
        - ii[side:, :-side]
        + ii[:-side, :-side]
    )
    full = side * side
    valid_positions = np.argwhere(counts == full)
    if valid_positions.size == 0:
        # Fall back: take the position with the highest count (largest overlap with pool)
        best = np.unravel_index(counts.argmax(), counts.shape)
        valid_positions = np.array([best])
        logger.warning(
            "No fully-valid %dx%d background patch found; using best-effort location "
            "(%d/%d valid pixels).", side, side, counts[best], full,
        )

    # Pick the patch with lowest median of the highest-signal channel (usually GFP)
    ref_channel = None
    for cand in ("GFP", "gfp"):
        if cand in mip_by_channel:
            ref_channel = cand
            break
    if ref_channel is None:
        ref_channel = sorted(mip_by_channel.keys())[0]
    ref = mip_by_channel[ref_channel]

    best_row, best_col = valid_positions[0]
    best_score = float("inf")
    # limit examined positions to at most 5000 to keep this fast on large images
    if len(valid_positions) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(valid_positions), size=5000, replace=False)
        valid_positions = valid_positions[idx]
    for r, c in valid_positions:
        patch = ref[r:r + side, c:c + side]
        m = float(np.median(patch))
        if m < best_score:
            best_score = m
            best_row, best_col = int(r), int(c)

    mask = np.zeros_like(pool_mask, dtype=bool)
    mask[best_row:best_row + side, best_col:best_col + side] = True

    per_mean = {ch: float(mip[best_row:best_row + side, best_col:best_col + side].mean())
                for ch, mip in mip_by_channel.items()}
    per_med = {ch: float(np.median(mip[best_row:best_row + side, best_col:best_col + side]))
               for ch, mip in mip_by_channel.items()}

    logger.info(
        "shared background patch: (%d, %d) size=%dx%d px  ref=%s median=%.2f",
        best_row, best_col, side, side, ref_channel, best_score,
    )
    return BackgroundPatch(
        row=best_row, col=best_col, rows=side, cols=side,
        mask=mask, per_channel_mean=per_mean, per_channel_median=per_med,
    )


def build_ring_masks(
    cell_mask: np.ndarray,
    pool_mask: np.ndarray,
    pixel_size_um: float,
    inner_um: float,
    outer_um: float,
) -> Dict[int, np.ndarray]:
    """Per-cell annulus masks; ring pixels must also lie in pool_mask (i.e. no other
    cell and no GFP speck).
    """
    from skimage.morphology import binary_dilation, disk

    inner_px = max(1, int(round(inner_um / pixel_size_um)))
    outer_px = max(inner_px + 1, int(round(outer_um / pixel_size_um)))
    all_cells = cell_mask > 0

    out: Dict[int, np.ndarray] = {}
    for lbl in np.unique(cell_mask):
        if lbl == 0:
            continue
        this = cell_mask == lbl
        outer = binary_dilation(this, footprint=disk(outer_px))
        inner = binary_dilation(this, footprint=disk(inner_px))
        ring = outer & ~inner & ~all_cells & pool_mask
        out[int(lbl)] = ring
    return out
