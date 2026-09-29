from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage.measure import find_contours

from .background import BackgroundPatch
from .measurement import CellRow


def _normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.5])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def save_overlay(
    out_path: Path,
    dapi_mip: np.ndarray,
    gfp_mip: np.ndarray,
    cell_mask: np.ndarray,
    rows: List[CellRow],
    background: BackgroundPatch,
    dpi: int = 150,
) -> None:
    """Overlay: passed cells green, rejected red, background patch yellow, labels white."""
    rgb = np.zeros((*dapi_mip.shape, 3), dtype=np.float32)
    rgb[..., 2] = _normalize(dapi_mip)         # blue = DAPI
    rgb[..., 1] = _normalize(gfp_mip)          # green = GFP

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")

    # cell outlines
    for r in rows:
        this = cell_mask == r.cell_id
        for contour in find_contours(this.astype(float), 0.5):
            colour = "#3ADB63" if r.qc_pass else "#F03A47"
            ax.plot(contour[:, 1], contour[:, 0], color=colour, lw=1.2)
        ax.text(r.centroid_x, r.centroid_y, str(r.cell_id),
                color="white", fontsize=8, ha="center", va="center",
                weight="bold")

    # background patch box
    ax.add_patch(mpatches.Rectangle(
        (background.col, background.row),
        background.cols, background.rows,
        edgecolor="#FFD700", facecolor="none", lw=1.5,
    ))
    ax.text(background.col, background.row - 4, "BG",
            color="#FFD700", fontsize=9, weight="bold")

    ax.axis("off")
    ax.set_title(out_path.stem, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
